"""End-to-end local context compression pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from cache import (
    PipelineCacheConfig,
    Tier1Payload,
    Tier2Payload,
    TwoTierCache,
    quantize_score_matrix,
)
from chunker import Chunk, OverlappingChunker
from csa import extract_csa_sentences
from filter_bm25 import bm25_survivors
from hca import ChunkSummarizer
from query_views import generate_query_views
from scorer import (
    DeterministicHashEmbedding,
    MultiAspectChunkScorer,
    alpha_for_query,
    aspect_weights,
    recency_aware_scores,
    sinkhorn_balance,
)
from selector import ChunkSelection, select_chunks_for_extraction


T = TypeVar("T")


# Fix 4: chunks with post-Sinkhorn max-aspect score below this floor are
# considered low-relevance and contribute to the OMITTED marker run instead
# of getting an HCA summary. Honors Mechanism D's coverage promise via the
# OMITTED accounting (every region is reported; some at fidelity zero).
HCA_RELEVANCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class IndexedChunk[T]:
    index: int
    chunk: T


@dataclass(frozen=True)
class SlidingWindowPartition[T]:
    compressible: tuple[IndexedChunk[T], ...]
    verbatim: tuple[IndexedChunk[T], ...]

    @property
    def compressible_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.compressible)

    @property
    def verbatim_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.verbatim)

    def ordered_output(self, compressed_content: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        return tuple(compressed_content) + tuple(chunk_text(item.chunk) for item in self.verbatim)


def split_sliding_window(
    chunks: list[T] | tuple[T, ...],
    *,
    sliding_window_chunks: int = 2,
) -> SlidingWindowPartition[T]:
    if sliding_window_chunks < 0:
        raise ValueError("sliding_window_chunks must be non-negative")

    total = len(chunks)
    split_at = max(0, total - sliding_window_chunks)
    compressible = tuple(
        IndexedChunk(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks[:split_at])
    )
    verbatim = tuple(
        IndexedChunk(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks[split_at:], start=split_at)
    )
    return SlidingWindowPartition(compressible=compressible, verbatim=verbatim)


@dataclass(frozen=True)
class CompressionConfig:
    """Compression knobs for the hybrid HCA + CSA pipeline.

    Compounded top-k tuning (DeepSeek-V4 Section 2.3.4):
    ``top_k_ratio_override`` and ``hca_max_words`` are NOT independent.
    HCA already covers every compressible chunk with a brief summary, so the
    smaller the HCA budget the more the top-k extractions need to carry. Tune
    them together:

    * Aggressive HCA (``hca_max_words`` ~10) — pair with moderate
      ``top_k_ratio_override`` of 0.30-0.40 so enough chunks receive CSA
      detail to compensate for the terse summaries.
    * Light HCA (``hca_max_words`` ~25) — a smaller ``top_k_ratio_override``
      of 0.15-0.20 is fine because the summaries already preserve most of the
      signal and CSA is only filling in the highest-relevance gaps.

    V4 chooses smaller top-k explicitly because compression already reduces
    the candidate set; mirror that intent rather than tuning either knob in
    isolation.
    """

    half_block_size: int = 48
    sliding_window_chunks: int = 2
    bm25_keep_fraction: float = 0.7
    bm25_min_chunks_to_filter: int = 10
    sentence_keep_ratio: float = 0.5
    # See class docstring: hca_max_words and top_k_ratio_override interact.
    hca_max_words: int = 15
    omission_threshold: int = 2
    beta_pos: float = 0.1
    sinkhorn_beta: float = 1.0
    sinkhorn_max_iter: int = 20
    abstain_threshold: float = 0.05
    # See class docstring: top_k_ratio_override and hca_max_words interact.
    top_k_ratio_override: float | None = None
    alpha_recent: float = 0.6
    alpha_default: float = 0.9
    raw_score_blend: float = 0.0
    # Fix 4: per-chunk max-aspect score below this floor → no HCA summary,
    # chunk just contributes to the next OMITTED marker run.
    hca_relevance_threshold: float = HCA_RELEVANCE_THRESHOLD


@dataclass(frozen=True)
class CompressionResult:
    compressed_text: str
    chunks: tuple[Chunk, ...]
    selected_chunk_indices: tuple[int, ...]
    compression_ratio: float


def compress_document_for_query(
    document: str,
    query: str,
    *,
    config: CompressionConfig | None = None,
    cache: TwoTierCache | str | Path | None = None,
) -> CompressionResult:
    config = config or CompressionConfig()
    cache_store = resolve_cache(cache)
    cache_config = cache_config_from_compression_config(config)
    query_views = generate_query_views(query)

    if cache_store is not None:
        tier2_key = cache_store.tier2_key(document, query, cache_config)
        tier2_payload = cache_store.get_tier2(tier2_key)
        if tier2_payload is not None:
            tier1_payload = cache_store.get_tier1(cache_store.tier1_key(document, cache_config))
            chunks = tuple(tier1_payload.chunks) if tier1_payload is not None else tuple(
                OverlappingChunker(config.half_block_size).chunk(document)
            )
            return CompressionResult(
                compressed_text=tier2_payload.final_output,
                chunks=chunks,
                selected_chunk_indices=tier2_payload.selected_chunk_indices,
                compression_ratio=text_length(tier2_payload.final_output) / max(1, text_length(document)),
            )

    tier1_payload = None
    if cache_store is not None:
        tier1_key = cache_store.tier1_key(document, cache_config)
        tier1_payload = cache_store.get_tier1(tier1_key)

    if tier1_payload is not None:
        chunks = tuple(tier1_payload.chunks)
    else:
        chunks = tuple(OverlappingChunker(config.half_block_size).chunk(document))
    if not chunks:
        return CompressionResult("", (), (), 0.0)

    partition = split_sliding_window(list(chunks), sliding_window_chunks=config.sliding_window_chunks)
    compressible_indices = partition.compressible_indices
    compressible_chunks = [chunks[index] for index in compressible_indices]
    compressible_texts = [chunk.text for chunk in compressible_chunks]

    if tier1_payload is not None:
        summaries = list(tier1_payload.hca_summaries)
    else:
        # Reuse the same deterministic embedder the scorer uses; no extra
        # model load, no LLM dependency.
        summarizer = ChunkSummarizer(
            word_limit=config.hca_max_words,
            embedder=DeterministicHashEmbedding().embed,
        )
        summaries = summarizer.summarize_chunks(compressible_texts)
    hca_summaries = {
        global_index: summary
        for global_index, summary in zip(compressible_indices, summaries)
    }

    selected_global_indices: tuple[int, ...] = ()
    csa_by_chunk: dict[int, tuple[str, ...]] = {}
    raw_scores = None
    # Fix 4: collected after Sinkhorn balancing, indexed by GLOBAL chunk index.
    chunk_max_scores: dict[int, float] = {}
    if compressible_chunks:
        survivor_local = bm25_survivors(
            compressible_texts,
            query,
            keep_fraction=config.bm25_keep_fraction,
            min_chunks_to_filter=config.bm25_min_chunks_to_filter,
        )
        scorer = MultiAspectChunkScorer()
        if tier1_payload is not None:
            scorer._chunk_embedding_cache[tuple(compressible_texts)] = tier1_payload.chunk_embeddings.astype("float64")
        raw_scores = scorer.score(compressible_texts, survivor_local, query_views)
        balanced_scores = sinkhorn_balance(
            raw_scores,
            beta=config.sinkhorn_beta,
            max_iter=config.sinkhorn_max_iter,
        )
        # Fix 4: per-chunk maximum across aspects, used to decide HCA-skip.
        if balanced_scores.size:
            per_chunk_max = balanced_scores.max(axis=1)
            chunk_max_scores = {
                global_index: float(per_chunk_max[local_index])
                for local_index, global_index in enumerate(compressible_indices)
            }
        alpha = config.alpha_recent if alpha_for_query(query) < config.alpha_default else config.alpha_default
        total_scores = recency_aware_scores(balanced_scores, query_views, alpha=alpha).total_scores
        if config.raw_score_blend:
            if not 0.0 <= config.raw_score_blend <= 1.0:
                raise ValueError("raw_score_blend must be in the interval [0, 1]")
            raw_total_scores = raw_scores @ aspect_weights(query_views)
            total_scores = (
                (1.0 - config.raw_score_blend) * total_scores
                + config.raw_score_blend * raw_total_scores
            )
        if config.top_k_ratio_override is not None:
            selection_local = select_with_ratio(
                total_scores,
                config.top_k_ratio_override,
                config.abstain_threshold,
            )
        else:
            selection_local = select_chunks_for_extraction(
                total_scores,
                query_views,
                abstain_threshold=config.abstain_threshold,
            )

        selected_global_indices = tuple(
            compressible_indices[local_index]
            for local_index in selection_local.selected_indices
        )
        extraction = extract_csa_sentences(
            chunks,
            ChunkSelection(
                selected_indices=selected_global_indices,
                top_k_indices=tuple(
                    compressible_indices[local_index]
                    for local_index in selection_local.top_k_indices
                ),
                threshold=selection_local.threshold,
                top_k_ratio=selection_local.top_k_ratio,
                n_select=selection_local.n_select,
            ),
            query_views,
            sentence_keep_ratio=config.sentence_keep_ratio,
            beta_pos=config.beta_pos,
            scorer=scorer,
        )
        for sentence in extraction.sentences:
            for chunk_index in sentence.source_chunk_indices:
                if chunk_index in selected_global_indices:
                    csa_by_chunk.setdefault(chunk_index, tuple())
                    csa_by_chunk[chunk_index] = (*csa_by_chunk[chunk_index], sentence.text)

    # Fix 4: pass relevance scores so HCA-only chunks below threshold skip
    # their summary and roll into the next OMITTED marker run.
    low_relevance_indices = frozenset(
        index
        for index in compressible_indices
        if chunk_max_scores.get(index, 1.0) < config.hca_relevance_threshold
    )
    compressed = assemble_context(
        partition,
        hca_summaries,
        csa_by_chunk,
        omission_threshold=config.omission_threshold,
        low_relevance_indices=low_relevance_indices,
    )
    if cache_store is not None:
        if tier1_payload is None:
            tier1_embeddings = MultiAspectChunkScorer().chunk_embeddings(compressible_texts)
            cache_store.put_tier1(
                cache_store.tier1_key(document, cache_config),
                Tier1Payload(
                    chunks=chunks,
                    chunk_embeddings=tier1_embeddings,
                    hca_summaries=tuple(summaries),
                ),
            )
        score_matrix = raw_scores if raw_scores is not None else np.zeros((0, len(aspect_weights(query_views))))
        cache_store.put_tier2(
            cache_store.tier2_key(document, query, cache_config),
            Tier2Payload(
                query_views=query_views,
                score_matrix=quantize_score_matrix(score_matrix),
                selected_chunk_indices=selected_global_indices,
                csa_extractions=csa_by_chunk,
                final_output=compressed,
            ),
        )
    return CompressionResult(
        compressed_text=compressed,
        chunks=chunks,
        selected_chunk_indices=selected_global_indices,
        compression_ratio=text_length(compressed) / max(1, text_length(document)),
    )


def resolve_cache(cache: TwoTierCache | str | Path | None) -> TwoTierCache | None:
    if cache is None:
        return None
    if isinstance(cache, TwoTierCache):
        return cache
    return TwoTierCache(cache)


def cache_config_from_compression_config(config: CompressionConfig) -> PipelineCacheConfig:
    return PipelineCacheConfig(
        half_block_size=config.half_block_size,
        sliding_window_chunks=config.sliding_window_chunks,
        simple_top_k_ratio=0.15,
        moderate_top_k_ratio=0.30,
        complex_top_k_ratio=0.45,
        abstain_threshold=config.abstain_threshold,
        sentence_keep_ratio=config.sentence_keep_ratio,
        hca_max_words=config.hca_max_words,
        omission_threshold=config.omission_threshold,
        beta_pos=config.beta_pos,
        alpha_recent=config.alpha_recent,
        alpha_default=config.alpha_default,
        bm25_keep_fraction=config.bm25_keep_fraction,
        bm25_min_chunks_to_filter=config.bm25_min_chunks_to_filter,
        sinkhorn_beta=config.sinkhorn_beta,
        sinkhorn_max_iter=config.sinkhorn_max_iter,
        top_k_ratio_override=config.top_k_ratio_override,
        raw_score_blend=config.raw_score_blend,
    )


def select_with_ratio(
    scores,
    top_k_ratio: float,
    abstain_threshold: float,
) -> ChunkSelection:
    if not 0.0 < top_k_ratio <= 1.0:
        raise ValueError("top_k_ratio_override must be in the interval (0, 1]")
    num_chunks = len(scores)
    if num_chunks == 0:
        return ChunkSelection((), (), abstain_threshold, top_k_ratio, 0)
    n_select = max(1, int(num_chunks * top_k_ratio))
    ranked = sorted(range(num_chunks), key=lambda index: (-scores[index], index))
    top_k = tuple(sorted(ranked[:n_select]))
    selected = tuple(index for index in top_k if scores[index] > abstain_threshold)
    return ChunkSelection(selected, top_k, abstain_threshold, top_k_ratio, n_select)


def text_length(text: str) -> int:
    return len(text.split())


def assemble_context(
    partition: SlidingWindowPartition,
    hca_summaries: Mapping[int, str],
    csa_extractions: Mapping[int, Sequence[str] | str],
    *,
    omission_threshold: int = 2,
    low_relevance_indices: Iterable[int] | None = None,
) -> str:
    if omission_threshold < 1:
        raise ValueError("omission_threshold must be positive")

    low_relevance = frozenset(low_relevance_indices or ())
    sections: list[str] = []
    # Fix 4: omitted_run counts ONLY dropped chunks (low-relevance, no summary).
    # Visible summaries flush any pending run. This preserves the accounting
    # invariant: every compressible chunk lands in exactly one category —
    # CSA-selected, HCA-summarized, or counted in an OMITTED marker.
    omitted_run = 0

    for item in partition.compressible:
        summary = hca_summaries.get(item.index, "")
        extraction = csa_extractions.get(item.index)
        has_extraction = extraction_has_content(extraction)

        if has_extraction:
            flush_omission_marker(sections, omitted_run, omission_threshold)
            omitted_run = 0
            sections.append(format_selected_section(summary, extraction))
        elif item.index in low_relevance:
            omitted_run += 1
        else:
            flush_omission_marker(sections, omitted_run, omission_threshold)
            omitted_run = 0
            sections.append(format_summary_section(summary))

    flush_omission_marker(sections, omitted_run, omission_threshold)

    if partition.verbatim:
        sections.append("[Recent content, verbatim:]")
        sections.extend(chunk_text(item.chunk) for item in partition.verbatim)

    return "\n\n".join(section for section in sections if section != "")


def format_summary_section(summary: str) -> str:
    return f"[Summary: {summary}]"


def format_selected_section(summary: str, extraction: Sequence[str] | str | None) -> str:
    extraction_text = normalize_extraction(extraction)
    return f"[Summary: {summary}]\n{extraction_text}"


def omission_marker(count: int) -> str:
    return f"[OMITTED: {count} sections of low-relevance content]"


def flush_omission_marker(
    sections: list[str],
    hca_only_run: int,
    omission_threshold: int,
) -> None:
    if hca_only_run >= omission_threshold:
        sections.append(omission_marker(hca_only_run))


def extraction_has_content(extraction: Sequence[str] | str | None) -> bool:
    return bool(normalize_extraction(extraction))


def normalize_extraction(extraction: Sequence[str] | str | None) -> str:
    if extraction is None:
        return ""
    if isinstance(extraction, str):
        return extraction.strip()
    return "\n".join(str(sentence).strip() for sentence in extraction if str(sentence).strip())


def chunk_text(chunk: object) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return str(chunk)

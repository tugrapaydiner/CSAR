"""Always-on per-chunk summarization for hybrid coverage."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


SummaryClient = Callable[[Sequence[str], int], Sequence[str]]
Embedder = Callable[[str], np.ndarray]

_logger = logging.getLogger(__name__)
_llm_failure_warned = False

CLAUSE_BOUNDARY_CHARS = (".", "!", "?", ";", ":", ",")
SOFT_BOUNDARY_CHARS = (",", ";", ":")


def _warn_llm_summary_failure_once(error: BaseException) -> None:
    """Log the first LLM summarizer failure per process; stay silent thereafter."""

    global _llm_failure_warned
    if _llm_failure_warned:
        return
    _llm_failure_warned = True
    _logger.warning(
        "ChunkSummarizer LLM client failed (%s: %s); "
        "falling back to extractive/heuristic summarization for the remainder of this process.",
        type(error).__name__,
        error,
    )


@dataclass
class ChunkSummarizer:
    """Summarize every chunk with aggressive content-only caching.

    Summarization strategy, in priority order:
      1. Optional ``llm_client`` if provided (kept for API symmetry; not used
         in the default LLM-free pipeline).
      2. ``extractive_summary`` when an ``embedder`` is supplied — picks the
         most central grammatical sentence and clause-truncates to fit.
      3. ``heuristic_summary`` as a last resort.
    """

    word_limit: int = 20
    llm_client: SummaryClient | None = None
    embedder: Embedder | None = None
    cache: dict[str, str] = field(default_factory=dict)

    def summarize_chunks(self, chunks: Sequence[str]) -> list[str]:
        summaries: list[str | None] = []
        missing_chunks: list[str] = []
        missing_positions: list[int] = []

        for chunk in chunks:
            cached = self.cache.get(chunk)
            if cached is None:
                summaries.append(None)
                missing_chunks.append(chunk)
                missing_positions.append(len(summaries) - 1)
            else:
                summaries.append(cached)

        generated = self._summarize_missing(missing_chunks)
        for position, chunk, summary in zip(missing_positions, missing_chunks, generated):
            self.cache[chunk] = summary
            summaries[position] = summary

        return [summary or "" for summary in summaries]

    def summarize_chunk(self, chunk: str) -> str:
        return self.summarize_chunks([chunk])[0]

    def _summarize_missing(self, chunks: Sequence[str]) -> list[str]:
        if not chunks:
            return []

        if self.llm_client is not None:
            try:
                llm_summaries = list(self.llm_client(chunks, self.word_limit))
            except Exception as error:
                _warn_llm_summary_failure_once(error)
                llm_summaries = []
            if len(llm_summaries) == len(chunks) and all(summary.strip() for summary in llm_summaries):
                # LLMs return free-form text; still respect the word budget.
                return [clause_aware_truncate(summary, self.word_limit) for summary in llm_summaries]

        if self.embedder is not None:
            return [
                extractive_summary(chunk, word_limit=self.word_limit, embedder=self.embedder)
                for chunk in chunks
            ]

        return [heuristic_summary(chunk, word_limit=self.word_limit) for chunk in chunks]


def summarize_chunks(
    chunks: Sequence[str],
    *,
    word_limit: int = 20,
    llm_client: SummaryClient | None = None,
    embedder: Embedder | None = None,
) -> list[str]:
    return ChunkSummarizer(
        word_limit=word_limit, llm_client=llm_client, embedder=embedder
    ).summarize_chunks(chunks)


def extractive_summary(
    text: str,
    word_limit: int,
    embedder: Embedder,
) -> str:
    """Pick the most central grammatical sentence and clause-truncate to fit.

    Per-sentence score = cosine similarity to the centroid of all sentence
    embeddings, plus small additive bonuses:

    * +0.10 if it is the first sentence (topic-introduction bias).
    * +0.05 * (capitalized words + numbers) / word_count (information density).
    * -0.10 if word count < 5 or > 40 (penalize fragments and rambles).

    Deterministic tiebreak on sentence index. The chosen sentence is then
    passed through ``clause_aware_truncate`` so the rendered summary always
    ends at a clause boundary or with an explicit "..." continuation marker.
    """

    if word_limit < 0:
        raise ValueError("word_limit must be non-negative")

    normalized = normalize_space(text)
    if not normalized:
        return ""

    sentences = split_sentences(normalized)
    if not sentences:
        return clause_aware_truncate(normalized, word_limit)
    if len(sentences) == 1:
        return clause_aware_truncate(sentences[0], word_limit)

    embeddings = np.stack([embedder(sentence) for sentence in sentences])
    centroid = embeddings.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    scored: list[tuple[float, int]] = []
    for index, sentence in enumerate(sentences):
        embedding = embeddings[index]
        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm > 0 and centroid_norm > 0:
            similarity = float(np.dot(embedding, centroid))
        else:
            similarity = 0.0

        word_count = len(sentence.split())
        capitalized = len(re.findall(r"\b[A-Z][A-Za-z0-9_+-]*\b", sentence))
        numbers = len(re.findall(r"\b\d+(?:[.,]\d+)*\b", sentence))
        density_bonus = 0.05 * (capitalized + numbers) / max(1, word_count)
        position_bonus = 0.10 if index == 0 else 0.0
        length_penalty = -0.10 if (word_count < 5 or word_count > 40) else 0.0

        scored.append((similarity + position_bonus + density_bonus + length_penalty, index))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    best_index = scored[0][1]
    return clause_aware_truncate(sentences[best_index], word_limit)


def clause_aware_truncate(sentence: str, word_limit: int) -> str:
    """Truncate at the latest clause boundary at-or-before ``word_limit`` words.

    If no clause-ending punctuation falls within the budget, hard-truncate and
    append "..." so the cut is visible. Already-short sentences are returned
    untouched (with their original punctuation preserved).
    """

    if word_limit < 0:
        raise ValueError("word_limit must be non-negative")
    sentence = normalize_space(sentence)
    if not sentence or word_limit == 0:
        return ""

    tokens = sentence.split()
    if len(tokens) <= word_limit:
        return sentence

    boundary_position: int | None = None
    for position, token in enumerate(tokens[:word_limit], start=1):
        stripped = token.rstrip(")\"'")
        if stripped and stripped[-1] in CLAUSE_BOUNDARY_CHARS:
            boundary_position = position

    if boundary_position is not None and boundary_position >= 2:
        kept = list(tokens[:boundary_position])
        last = kept[-1].rstrip(")\"'")
        if last and last[-1] in SOFT_BOUNDARY_CHARS:
            # Replace dangling separator with explicit continuation marker.
            kept[-1] = kept[-1][:-1]
            return " ".join(kept) + "..."
        # Hard boundary (.!?) — already a complete sentence.
        return " ".join(kept)

    # No usable boundary inside the budget — signal continuation explicitly.
    return " ".join(tokens[:word_limit]) + "..."


def heuristic_summary(chunk: str, *, word_limit: int = 20) -> str:
    normalized = normalize_space(chunk)
    if not normalized:
        return ""

    sentences = split_sentences(normalized)
    if not sentences:
        return limit_words(normalized, word_limit)

    best_sentence = max(sentences, key=information_density)
    return limit_words(best_sentence, word_limit)


def information_density(sentence: str) -> tuple[int, int, int]:
    capitalized_count = len(re.findall(r"\b[A-Z][A-Za-z0-9_+-]*\b", sentence))
    number_count = len(re.findall(r"\b\d+(?:[.,]\d+)*\b", sentence))
    word_count = len(words(sentence))
    return (capitalized_count + number_count, word_count, -len(sentence))


def split_sentences(text: str) -> list[str]:
    matches = re.finditer(r"\S.+?(?:[.!?]+(?=\s|$)|$)", text, re.DOTALL)
    return [normalize_space(match.group(0)) for match in matches if match.group(0).strip()]


def limit_words(text: str, word_limit: int) -> str:
    if word_limit < 0:
        raise ValueError("word_limit must be non-negative")
    selected_words = words(normalize_space(text))[:word_limit]
    return " ".join(selected_words)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())

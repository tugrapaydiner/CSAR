"""CSA fine extraction with overlap-aware joint sentence normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from chunker import Chunk
from query_views import QuestionCategory, QueryViewSet
from scorer import MultiAspectChunkScorer, aspect_weights, sentence_position_multipliers
from selector import ChunkSelection


# Fix 3: per-category regex patterns. Hit → +SPAN_MATCH_BONUS on the sentence's
# raw score before top-N selection. Bonus is small enough not to override
# semantic relevance, large enough to break ties toward answer-bearing sentences.
SPAN_MATCH_BONUS = 0.15

_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_LOCATION_AT_RE = re.compile(r"\b(?:in|at|near|from)\s+[A-Z][A-Za-z]+")
_LOCATION_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)
_DURATION_RE = re.compile(r"\b\d+\s+(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CAUSAL_RE = re.compile(
    r"\b(?:because|due\s+to|as\s+a\s+result|therefore|so\s+that|thus|hence|since|owing\s+to)\b",
    re.IGNORECASE,
)


def span_match_bonus(sentence: str, category: QuestionCategory) -> float:
    """Return SPAN_MATCH_BONUS if the sentence carries the answer-shape pattern."""

    if category == "person":
        if _PERSON_NAME_RE.search(sentence):
            return SPAN_MATCH_BONUS
    elif category == "location":
        if _LOCATION_AT_RE.search(sentence):
            return SPAN_MATCH_BONUS
        # Fallback: any multi-word proper-noun phrase in a sentence whose first
        # word is not capitalized (avoids matching the leading word).
        tail = sentence.split(None, 1)[1] if len(sentence.split()) > 1 else ""
        if _LOCATION_PROPER_RE.search(tail):
            return SPAN_MATCH_BONUS
    elif category == "temporal":
        if _YEAR_RE.search(sentence) or _MONTH_RE.search(sentence) or _DURATION_RE.search(sentence):
            return SPAN_MATCH_BONUS
    elif category == "numerical":
        if _NUMBER_RE.search(sentence):
            return SPAN_MATCH_BONUS
    elif category == "causal":
        if _CAUSAL_RE.search(sentence):
            return SPAN_MATCH_BONUS
    return 0.0


@dataclass(frozen=True)
class SentenceOccurrence:
    key: tuple[int, int, str]
    text: str
    chunk_index: int
    half_block_index: int
    sentence_index: int
    local_position: int
    local_sentence_count: int
    raw_score: float
    joint_score: float


@dataclass(frozen=True)
class ExtractedSentence:
    text: str
    score: float
    half_block_index: int
    sentence_index: int
    source_chunk_indices: tuple[int, ...]


@dataclass(frozen=True)
class CSAExtraction:
    sentences: tuple[ExtractedSentence, ...]
    occurrences: tuple[SentenceOccurrence, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(sentence.text for sentence in self.sentences)


def extract_csa_sentences(
    chunks: Sequence[Chunk],
    selection: ChunkSelection | Iterable[int],
    query_views: QueryViewSet,
    *,
    sentence_keep_ratio: float = 0.5,
    beta_pos: float = 0.1,
    scorer: MultiAspectChunkScorer | None = None,
) -> CSAExtraction:
    if not 0.0 < sentence_keep_ratio <= 1.0:
        raise ValueError("sentence_keep_ratio must be in the interval (0, 1]")
    if beta_pos < 0:
        raise ValueError("beta_pos must be non-negative")

    selected_chunk_indices = committed_indices(selection)
    if not selected_chunk_indices:
        return CSAExtraction(sentences=(), occurrences=())

    scorer = scorer or MultiAspectChunkScorer()
    occurrence_shells = build_occurrence_shells(chunks, selected_chunk_indices)
    if not occurrence_shells:
        return CSAExtraction(sentences=(), occurrences=())

    sentence_texts = [shell["text"] for shell in occurrence_shells]
    score_matrix = scorer.score(sentence_texts, range(len(sentence_texts)), query_views)
    sentence_scores = score_matrix @ aspect_weights(query_views)

    raw_scores: list[float] = []
    category = query_views.category
    for shell, sentence_score in zip(occurrence_shells, sentence_scores):
        multipliers = sentence_position_multipliers(
            shell["local_sentence_count"],
            beta_pos=beta_pos,
        )
        # Fix 3: answer-span boost added BEFORE top-N selection so it can break
        # ties in favor of sentences that carry the answer shape.
        bonus = span_match_bonus(shell["text"], category)
        raw_scores.append(
            float(sentence_score * multipliers[shell["local_position"]] + bonus)
        )

    joint_scores = joint_occurrence_scores(occurrence_shells, raw_scores)

    occurrences = tuple(
        SentenceOccurrence(
            key=shell["key"],
            text=shell["text"],
            chunk_index=shell["chunk_index"],
            half_block_index=shell["half_block_index"],
            sentence_index=shell["sentence_index"],
            local_position=shell["local_position"],
            local_sentence_count=shell["local_sentence_count"],
            raw_score=raw_score,
            joint_score=joint_scores[shell["key"]],
        )
        for shell, raw_score in zip(occurrence_shells, raw_scores)
    )

    selected_keys = select_sentence_keys_by_chunk(occurrences, sentence_keep_ratio)
    extracted = build_extracted_sentences(occurrences, selected_keys)
    return CSAExtraction(sentences=tuple(extracted), occurrences=occurrences)


def committed_indices(selection: ChunkSelection | Iterable[int]) -> tuple[int, ...]:
    if isinstance(selection, ChunkSelection):
        return selection.selected_indices
    return tuple(selection)


def build_occurrence_shells(
    chunks: Sequence[Chunk],
    selected_chunk_indices: Sequence[int],
) -> list[dict[str, int | str | tuple[int, int, str]]]:
    shells: list[dict[str, int | str | tuple[int, int, str]]] = []
    for chunk_index in selected_chunk_indices:
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[chunk_index]
        flattened = [
            (half_block.index, sentence_index, sentence)
            for half_block in chunk.half_blocks
            for sentence_index, sentence in enumerate(half_block.sentences)
        ]
        local_count = len(flattened)
        for local_position, (half_block_index, sentence_index, sentence) in enumerate(flattened):
            key = (half_block_index, sentence_index, sentence)
            shells.append(
                {
                    "key": key,
                    "text": sentence,
                    "chunk_index": chunk.index,
                    "half_block_index": half_block_index,
                    "sentence_index": sentence_index,
                    "local_position": local_position,
                    "local_sentence_count": local_count,
                }
            )
    return shells


def joint_occurrence_scores(
    occurrence_shells: Sequence[dict[str, int | str | tuple[int, int, str]]],
    raw_scores: Sequence[float],
) -> dict[tuple[int, int, str], float]:
    """Normalize sentence occurrences within each overlapping chunk, then merge duplicates."""

    normalized_by_position = [0.0 for _ in raw_scores]
    positions_by_chunk: dict[int, list[int]] = {}
    for position, shell in enumerate(occurrence_shells):
        positions_by_chunk.setdefault(int(shell["chunk_index"]), []).append(position)

    for positions in positions_by_chunk.values():
        positive_positions = [position for position in positions if raw_scores[position] > 0.0]
        if not positive_positions:
            continue

        positive_scores = np.asarray([raw_scores[position] for position in positive_positions], dtype=np.float64)
        shifted = positive_scores - np.max(positive_scores)
        exp_scores = np.exp(shifted)
        normalized_scores = exp_scores / np.sum(exp_scores)
        for position, normalized_score in zip(positive_positions, normalized_scores):
            normalized_by_position[position] = float(normalized_score)

    joint_scores: dict[tuple[int, int, str], float] = {}
    for shell, normalized_score in zip(occurrence_shells, normalized_by_position):
        key = shell["key"]
        joint_scores[key] = joint_scores.get(key, 0.0) + normalized_score
    return joint_scores


def select_sentence_keys_by_chunk(
    occurrences: Sequence[SentenceOccurrence],
    sentence_keep_ratio: float,
) -> set[tuple[int, int, str]]:
    selected: set[tuple[int, int, str]] = set()
    by_chunk: dict[int, list[SentenceOccurrence]] = {}
    for occurrence in occurrences:
        by_chunk.setdefault(occurrence.chunk_index, []).append(occurrence)

    for chunk_occurrences in by_chunk.values():
        n_keep = max(1, int(len(chunk_occurrences) * sentence_keep_ratio))
        ranked = sorted(
            chunk_occurrences,
            key=lambda occurrence: (
                -occurrence.joint_score,
                occurrence.half_block_index,
                occurrence.sentence_index,
            ),
        )
        selected.update(occurrence.key for occurrence in ranked[:n_keep])

    return selected


def build_extracted_sentences(
    occurrences: Sequence[SentenceOccurrence],
    selected_keys: set[tuple[int, int, str]],
) -> list[ExtractedSentence]:
    by_key: dict[tuple[int, int, str], list[SentenceOccurrence]] = {}
    for occurrence in occurrences:
        if occurrence.key in selected_keys:
            by_key.setdefault(occurrence.key, []).append(occurrence)

    extracted = []
    for key, key_occurrences in by_key.items():
        representative = max(key_occurrences, key=lambda occurrence: occurrence.joint_score)
        extracted.append(
            ExtractedSentence(
                text=representative.text,
                score=representative.joint_score,
                half_block_index=representative.half_block_index,
                sentence_index=representative.sentence_index,
                source_chunk_indices=tuple(
                    sorted({occurrence.chunk_index for occurrence in key_occurrences})
                ),
            )
        )

    return sorted(
        extracted,
        key=lambda sentence: (sentence.half_block_index, sentence.sentence_index),
    )

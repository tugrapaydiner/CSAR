"""Multi-aspect scoring for BM25 survivor chunks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from determinism import deterministic_l2_normalize
from query_views import QueryViewSet, STOP_WORDS, extract_query_entities, tokenize_lower


Aspect = Literal["semantic", "rewritten", "entity", "question_type", "bm25"]
ASPECTS: tuple[Aspect, ...] = ("semantic", "rewritten", "entity", "question_type", "bm25")
# Default BM25 weight when query_views supplies no explicit value (back-compat).
DEFAULT_BM25_WEIGHT = 0.5


EMBEDDING_STOP_WORDS = STOP_WORDS | {
    "about",
    "all",
    "can",
    "could",
    "give",
    "tell",
    "their",
    "them",
    "this",
    "that",
    "these",
    "those",
}


@dataclass(frozen=True)
class DeterministicHashEmbedding:
    """Word-level deterministic embedding backend with stable hash buckets."""

    dimensions: int = 4096

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for token in content_tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        return deterministic_l2_normalize(vector)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float64)
        matrix = np.vstack([self.embed(text) for text in texts])
        return deterministic_l2_normalize(matrix, axis=1)


@dataclass
class MultiAspectChunkScorer:
    """Score chunks against query views across four V4-inspired aspects."""

    embedding: DeterministicHashEmbedding = field(default_factory=DeterministicHashEmbedding)
    _chunk_embedding_cache: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)
    _query_embedding_cache: dict[str, np.ndarray] = field(default_factory=dict)

    def score(
        self,
        chunks: Sequence[str],
        survivor_indices: Iterable[int],
        query_views: QueryViewSet,
    ) -> np.ndarray:
        score_matrix = np.zeros((len(chunks), len(ASPECTS)), dtype=np.float64)
        if not chunks:
            return score_matrix

        valid_survivors = sorted(
            {
                index
                for index in survivor_indices
                if 0 <= index < len(chunks)
            }
        )
        if not valid_survivors:
            return score_matrix

        chunk_embeddings = self.survivor_embeddings(chunks, valid_survivors)
        literal_embedding = self.query_embedding(query_views.get("literal").text)
        rewritten_embedding = self.query_embedding(query_views.get("rewritten").text)
        question_embedding = self.query_embedding(query_views.get("question_type").text)
        entities = query_entities(query_views)
        bm25_column = compute_bm25_column(chunks, query_views.query, valid_survivors)

        for chunk_index in valid_survivors:
            chunk_embedding = chunk_embeddings[chunk_index]
            chunk_text = chunks[chunk_index]
            score_matrix[chunk_index, 0] = float(np.dot(chunk_embedding, literal_embedding))
            score_matrix[chunk_index, 1] = float(np.dot(chunk_embedding, rewritten_embedding))
            score_matrix[chunk_index, 2] = entity_overlap(chunk_text, entities)
            score_matrix[chunk_index, 3] = float(np.dot(chunk_embedding, question_embedding))
            score_matrix[chunk_index, 4] = bm25_column[chunk_index]

        return np.maximum(score_matrix, 0.0)

    def chunk_embeddings(self, chunks: Sequence[str]) -> np.ndarray:
        cache_key = tuple(chunks)
        cached = self._chunk_embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        embeddings = self.embedding.embed_many(chunks)
        self._chunk_embedding_cache[cache_key] = embeddings
        return embeddings

    def survivor_embeddings(
        self,
        chunks: Sequence[str],
        survivor_indices: Sequence[int],
    ) -> np.ndarray:
        """Embed only BM25 survivor chunks; non-survivor rows remain zero.

        Honors any prewarmed full-document cache entry (e.g. populated from a
        Tier 1 cache hit) so we never waste a cached embedding matrix.
        """

        cached = self._chunk_embedding_cache.get(tuple(chunks))
        if cached is not None:
            return cached

        embeddings = np.zeros((len(chunks), self.embedding.dimensions), dtype=np.float64)
        if not survivor_indices:
            return embeddings

        survivor_texts = [chunks[index] for index in survivor_indices]
        survivor_matrix = self.embedding.embed_many(survivor_texts)
        for local_index, global_index in enumerate(survivor_indices):
            embeddings[global_index] = survivor_matrix[local_index]
        return embeddings

    def query_embedding(self, text: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(text)
        if cached is not None:
            return cached

        embedding = self.embedding.embed(text)
        self._query_embedding_cache[text] = embedding
        return embedding


def score_chunks(
    chunks: Sequence[str],
    survivor_indices: Iterable[int],
    query_views: QueryViewSet,
    *,
    scorer: MultiAspectChunkScorer | None = None,
) -> np.ndarray:
    scorer = scorer or MultiAspectChunkScorer()
    return scorer.score(chunks, survivor_indices, query_views)


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize_lower(text)
        if token not in EMBEDDING_STOP_WORDS and len(token) > 1
    ]


def query_entities(query_views: QueryViewSet) -> tuple[str, ...]:
    extracted = extract_query_entities(query_views.query)
    if extracted:
        return tuple(entity.casefold() for entity in extracted)
    return tuple(significant_entity_terms(query_views.get("entity").text))


def significant_entity_terms(text: str) -> list[str]:
    terms = []
    for token in tokenize_lower(text):
        if token in EMBEDDING_STOP_WORDS or token in {"created", "create", "built", "invented"}:
            continue
        terms.append(token)
    return terms


def entity_overlap(chunk: str, entities: Sequence[str]) -> float:
    if not entities:
        return 0.0

    chunk_tokens = set(tokenize_lower(chunk))
    chunk_lower = normalize_for_contains(chunk)
    matches = 0
    for entity in entities:
        normalized_entity = normalize_for_contains(entity)
        if not normalized_entity:
            continue
        if " " in normalized_entity:
            matches += int(normalized_entity in chunk_lower)
        else:
            matches += int(entity.lower() in chunk_tokens)
    return matches / len(entities)


def normalize_for_contains(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


@dataclass(frozen=True)
class SinkhornResult:
    matrix: np.ndarray
    max_changes: tuple[float, ...]


def sinkhorn_balance(
    score_matrix: np.ndarray,
    *,
    beta: float = 1.0,
    max_iter: int = 20,
    epsilon: float = 1e-12,
    return_history: bool = False,
) -> np.ndarray | SinkhornResult:
    """Balance a rectangular score matrix with Sinkhorn-Knopp iterations."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")

    num_rows, num_cols = scores.shape
    if num_rows == 0 or num_cols == 0:
        empty = np.zeros_like(scores, dtype=np.float64)
        result = SinkhornResult(matrix=empty, max_changes=())
        return result if return_history else result.matrix

    active_rows = np.any(scores > 0.0, axis=1)
    active_columns = np.any(scores > 0.0, axis=0)
    if not np.any(active_rows) or not np.any(active_columns):
        empty = np.zeros_like(scores, dtype=np.float64)
        result = SinkhornResult(matrix=empty, max_changes=())
        return result if return_history else result.matrix

    active_scores = scores[np.ix_(active_rows, active_columns)]
    scaled_scores = active_scores * beta
    scaled_scores = scaled_scores - np.max(scaled_scores)
    matrix = np.exp(scaled_scores)

    active_row_count, active_column_count = active_scores.shape
    row_targets = np.ones((active_row_count, 1), dtype=np.float64)
    column_targets = np.full(
        (1, active_column_count),
        active_row_count / active_column_count,
        dtype=np.float64,
    )
    max_changes: list[float] = []

    for _ in range(max_iter):
        previous = matrix.copy()
        matrix = scale_rows(matrix, row_targets, epsilon)
        matrix = scale_columns(matrix, column_targets, epsilon)
        max_changes.append(float(np.max(np.abs(matrix - previous))))

    balanced = np.zeros_like(scores, dtype=np.float64)
    balanced[np.ix_(active_rows, active_columns)] = matrix
    result = SinkhornResult(matrix=balanced, max_changes=tuple(max_changes))
    return result if return_history else result.matrix


def scale_rows(matrix: np.ndarray, row_targets: np.ndarray, epsilon: float) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix * (row_targets / np.maximum(row_sums, epsilon))


def scale_columns(matrix: np.ndarray, column_targets: np.ndarray, epsilon: float) -> np.ndarray:
    column_sums = matrix.sum(axis=0, keepdims=True)
    return matrix * (column_targets / np.maximum(column_sums, epsilon))


RECENCY_MARKERS = {
    "recent",
    "recently",
    "latest",
    "current",
    "currently",
    "today",
    "new",
    "newest",
    "now",
    "upcoming",
}


@dataclass(frozen=True)
class RecencyScoredChunks:
    semantic_scores: np.ndarray
    recency_scores: np.ndarray
    total_scores: np.ndarray
    alpha: float


def sentence_position_multipliers(
    sentence_count: int,
    *,
    beta_pos: float = 0.1,
) -> np.ndarray:
    """Return per-sentence multipliers with a first/last sentence bonus."""

    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    if beta_pos < 0:
        raise ValueError("beta_pos must be non-negative")
    if sentence_count == 0:
        return np.zeros(0, dtype=np.float64)

    multipliers = np.ones(sentence_count, dtype=np.float64)
    multipliers[0] += beta_pos
    if sentence_count > 1:
        multipliers[-1] += beta_pos
    return multipliers


def apply_sentence_position_bias(
    sentence_scores: np.ndarray,
    *,
    beta_pos: float = 0.1,
) -> np.ndarray:
    scores = np.asarray(sentence_scores, dtype=np.float64)
    return scores * sentence_position_multipliers(len(scores), beta_pos=beta_pos)


def recency_scores(num_chunks: int) -> np.ndarray:
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if num_chunks == 0:
        return np.zeros(0, dtype=np.float64)
    if num_chunks == 1:
        return np.ones(1, dtype=np.float64)
    return np.arange(num_chunks, dtype=np.float64) / (num_chunks - 1)


def alpha_for_query(query: str) -> float:
    tokens = set(query.lower().split())
    return 0.6 if tokens & RECENCY_MARKERS else 0.9


def aspect_weights(query_views: QueryViewSet) -> np.ndarray:
    weights = query_views.weights
    return np.asarray(
        [
            weights.get("literal", 0.0),
            weights.get("rewritten", 0.0),
            weights.get("entity", 0.0),
            weights.get("question_type", 0.0),
            weights.get("bm25", DEFAULT_BM25_WEIGHT),
        ],
        dtype=np.float64,
    )


def compute_bm25_column(
    texts: Sequence[str],
    query: str,
    survivor_indices: Sequence[int],
) -> np.ndarray:
    """Per-chunk BM25 score against the literal query, normalized to [0, 1].

    Only survivors are tokenized and scored — preserves the Mechanism C cost
    gate. Non-survivor entries stay at 0.
    """

    column = np.zeros(len(texts), dtype=np.float64)
    if not survivor_indices:
        return column
    query_tokens = query.lower().split()
    if not query_tokens:
        return column
    survivor_corpus = [texts[index].lower().split() for index in survivor_indices]
    if not any(survivor_corpus):
        return column

    bm25 = BM25Okapi(survivor_corpus)
    raw_scores = bm25.get_scores(query_tokens)
    if len(raw_scores) == 0:
        return column
    max_score = float(np.max(raw_scores))
    if max_score <= 0:
        return column
    normalized = raw_scores / max_score
    for local_index, global_index in enumerate(survivor_indices):
        column[global_index] = float(normalized[local_index])
    return column


def weighted_semantic_scores(
    balanced_score_matrix: np.ndarray,
    query_views: QueryViewSet,
) -> np.ndarray:
    matrix = np.asarray(balanced_score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("balanced_score_matrix must be two-dimensional")
    if matrix.shape[1] != len(ASPECTS):
        raise ValueError(f"balanced_score_matrix must have {len(ASPECTS)} aspect columns")
    return matrix @ aspect_weights(query_views)


def recency_aware_scores(
    balanced_score_matrix: np.ndarray,
    query_views: QueryViewSet,
    *,
    alpha: float | None = None,
) -> RecencyScoredChunks:
    if alpha is None:
        alpha = alpha_for_query(query_views.query)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the interval [0, 1]")

    semantic = weighted_semantic_scores(balanced_score_matrix, query_views)
    recency = recency_scores(len(semantic))
    total = alpha * semantic + (1.0 - alpha) * recency

    return RecencyScoredChunks(
        semantic_scores=semantic,
        recency_scores=recency,
        total_scores=total,
        alpha=alpha,
    )

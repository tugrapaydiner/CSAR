"""BM25 pre-filter for cheap first-stage chunk retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rank_bm25 import BM25Okapi


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize for BM25 using lowercase whitespace terms."""

    return text.lower().split()


class BM25PreFilter:
    """Return top BM25 survivor indices while skipping tiny inputs."""

    def __init__(
        self,
        *,
        keep_fraction: float = 0.7,
        min_chunks_to_filter: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in the interval (0, 1]")
        if min_chunks_to_filter < 0:
            raise ValueError("min_chunks_to_filter must be non-negative")

        self.keep_fraction = keep_fraction
        self.min_chunks_to_filter = min_chunks_to_filter
        self.k1 = k1
        self.b = b

    def survivors(self, chunks: Sequence[str], query: str) -> list[int]:
        if not chunks:
            return []

        all_indices = list(range(len(chunks)))
        query_tokens = bm25_tokenize(query)
        if len(chunks) < self.min_chunks_to_filter or not query_tokens:
            return all_indices

        tokenized_chunks = [bm25_tokenize(chunk) for chunk in chunks]
        if not any(tokenized_chunks):
            return all_indices

        keep_count = max(1, math.ceil(len(chunks) * self.keep_fraction))
        bm25 = BM25Okapi(tokenized_chunks, k1=self.k1, b=self.b)
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            all_indices,
            key=lambda index: (-scores[index], index),
        )
        return sorted(ranked_indices[:keep_count])


def bm25_survivors(
    chunks: Sequence[str],
    query: str,
    *,
    keep_fraction: float = 0.7,
    min_chunks_to_filter: int = 10,
) -> list[int]:
    return BM25PreFilter(
        keep_fraction=keep_fraction,
        min_chunks_to_filter=min_chunks_to_filter,
    ).survivors(chunks, query)

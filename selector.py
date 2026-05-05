"""Adaptive top-k chunk selection with abstain thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from query_views import Complexity, QueryViewSet


TOP_K_RATIO_BY_COMPLEXITY: dict[Complexity, float] = {
    "simple": 0.15,
    "moderate": 0.30,
    "complex": 0.45,
}


@dataclass(frozen=True)
class ChunkSelection:
    selected_indices: tuple[int, ...]
    top_k_indices: tuple[int, ...]
    threshold: float
    top_k_ratio: float
    n_select: int
    committed: bool = True


def select_chunks_for_extraction(
    scores: np.ndarray | list[float],
    query_views: QueryViewSet,
    *,
    abstain_threshold: float = 0.05,
) -> ChunkSelection:
    if abstain_threshold < 0:
        raise ValueError("abstain_threshold must be non-negative")

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    num_chunks = len(score_array)
    ratio = top_k_ratio_for_complexity(query_views.complexity)
    if num_chunks == 0:
        return ChunkSelection(
            selected_indices=(),
            top_k_indices=(),
            threshold=abstain_threshold,
            top_k_ratio=ratio,
            n_select=0,
        )

    n_select = max(1, int(num_chunks * ratio))
    ranked = sorted(range(num_chunks), key=lambda index: (-score_array[index], index))
    top_k_indices = tuple(sorted(ranked[:n_select]))
    selected = tuple(
        index
        for index in top_k_indices
        if score_array[index] > abstain_threshold
    )

    return ChunkSelection(
        selected_indices=selected,
        top_k_indices=top_k_indices,
        threshold=abstain_threshold,
        top_k_ratio=ratio,
        n_select=n_select,
    )


def top_k_ratio_for_complexity(complexity: Complexity) -> float:
    return TOP_K_RATIO_BY_COMPLEXITY[complexity]

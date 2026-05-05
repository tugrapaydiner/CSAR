"""Disk-backed two-tier cache for compression pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
PipelinePath = Literal["cold", "tier1_hit", "tier2_hit"]


@dataclass(frozen=True)
class PipelineCacheConfig:
    chunk_size: int = 8
    half_block_size: int = 4
    sliding_window_chunks: int = 2
    simple_top_k_ratio: float = 0.15
    moderate_top_k_ratio: float = 0.30
    complex_top_k_ratio: float = 0.45
    abstain_threshold: float = 0.05
    sentence_keep_ratio: float = 0.5
    hca_max_words: int = 15
    omission_threshold: int = 2
    beta_pos: float = 0.1
    alpha_recent: float = 0.6
    alpha_default: float = 0.9
    bm25_keep_fraction: float = 0.7
    bm25_min_chunks_to_filter: int = 10
    sinkhorn_beta: float = 1.0
    sinkhorn_max_iter: int = 20
    embedding_dimensions: int = 4096
    top_k_ratio_override: float | None = None
    raw_score_blend: float = 0.0
    extra: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_key_dict(self) -> dict[str, JsonValue]:
        data = asdict(self)
        data["extra"] = dict(sorted(dict(self.extra).items()))
        return data


@dataclass(frozen=True)
class Tier1Payload:
    chunks: tuple[Any, ...]
    chunk_embeddings: np.ndarray
    hca_summaries: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_embeddings", np.asarray(self.chunk_embeddings, dtype=np.float16))


@dataclass(frozen=True)
class QuantizedScoreMatrix:
    values: np.ndarray
    mean: float
    scale: float
    shape: tuple[int, int]

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float64) * self.scale + self.mean


@dataclass(frozen=True)
class Tier2Payload:
    query_views: Any
    score_matrix: QuantizedScoreMatrix
    selected_chunk_indices: tuple[int, ...]
    csa_extractions: Any
    final_output: str


@dataclass(frozen=True)
class CachedPipelineResult:
    final_output: str
    tier1_hit: bool
    tier2_hit: bool
    path: PipelinePath


class TwoTierCache:
    """Persistent Tier 1 document cache and Tier 2 query cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.tier1_dir = self.cache_dir / "tier1"
        self.tier2_dir = self.cache_dir / "tier2"
        self.tier1_dir.mkdir(parents=True, exist_ok=True)
        self.tier2_dir.mkdir(parents=True, exist_ok=True)

    def document_hash(self, document: str) -> str:
        return sha256_text(document)

    def tier1_key(self, document: str, config: PipelineCacheConfig) -> str:
        return stable_hash(
            {
                "tier": 1,
                "document_hash": self.document_hash(document),
                "pipeline_config": config.to_key_dict(),
            }
        )

    def tier2_key(self, document: str, query: str, config: PipelineCacheConfig) -> str:
        return stable_hash(
            {
                "tier": 2,
                "document_hash": self.document_hash(document),
                "query": query,
                "pipeline_config": config.to_key_dict(),
            }
        )

    def get_tier1(self, key: str) -> Tier1Payload | None:
        return self._read(self.tier1_dir / f"{key}.pkl")

    def put_tier1(self, key: str, payload: Tier1Payload) -> None:
        self._write(self.tier1_dir / f"{key}.pkl", payload)

    def get_tier2(self, key: str) -> Tier2Payload | None:
        return self._read(self.tier2_dir / f"{key}.pkl")

    def put_tier2(self, key: str, payload: Tier2Payload) -> None:
        self._write(self.tier2_dir / f"{key}.pkl", payload)

    def _read(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        with path.open("rb") as file:
            return pickle.load(file)

    def _write(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = Path(file.name)
        temp_path.replace(path)


class CachedPipelineRunner:
    """Small orchestration wrapper that enforces Tier 2/Tier 1/cold behavior."""

    def __init__(
        self,
        cache: TwoTierCache,
        *,
        build_tier1: Callable[[str, PipelineCacheConfig], Tier1Payload],
        build_tier2: Callable[[str, str, PipelineCacheConfig, Tier1Payload], Tier2Payload],
    ) -> None:
        self.cache = cache
        self.build_tier1 = build_tier1
        self.build_tier2 = build_tier2

    def run(self, document: str, query: str, config: PipelineCacheConfig) -> CachedPipelineResult:
        tier2_key = self.cache.tier2_key(document, query, config)
        tier2_payload = self.cache.get_tier2(tier2_key)
        if tier2_payload is not None:
            return CachedPipelineResult(
                final_output=tier2_payload.final_output,
                tier1_hit=True,
                tier2_hit=True,
                path="tier2_hit",
            )

        tier1_key = self.cache.tier1_key(document, config)
        tier1_payload = self.cache.get_tier1(tier1_key)
        tier1_hit = tier1_payload is not None
        if tier1_payload is None:
            tier1_payload = self.build_tier1(document, config)
            self.cache.put_tier1(tier1_key, tier1_payload)

        tier2_payload = self.build_tier2(document, query, config, tier1_payload)
        self.cache.put_tier2(tier2_key, tier2_payload)

        return CachedPipelineResult(
            final_output=tier2_payload.final_output,
            tier1_hit=tier1_hit,
            tier2_hit=False,
            path="tier1_hit" if tier1_hit else "cold",
        )


def quantize_score_matrix(score_matrix: np.ndarray) -> QuantizedScoreMatrix:
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.size == 0:
        return QuantizedScoreMatrix(
            values=np.zeros(matrix.shape, dtype=np.int8),
            mean=0.0,
            scale=1.0,
            shape=matrix.shape,
        )

    mean = float(np.mean(matrix))
    centered = matrix - mean
    max_abs = float(np.max(np.abs(centered)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    values = np.clip(np.rint(centered / scale), -127, 127).astype(np.int8)
    return QuantizedScoreMatrix(values=values, mean=mean, scale=scale, shape=matrix.shape)


def stable_hash(value: Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

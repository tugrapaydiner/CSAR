"""Utilities for deterministic numeric operations."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_EPSILON = 1e-8


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


@dataclass(frozen=True)
class RandomStateSnapshot:
    """Captured random states for Python, NumPy, and optional Torch."""

    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: Any | None = None
    torch_cuda_state: list[Any] | None = None


def set_random_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch if Torch is installed."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _optional_torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_random_state() -> RandomStateSnapshot:
    """Capture current Python, NumPy, and optional Torch random state."""

    torch = _optional_torch()
    if torch is None:
        return RandomStateSnapshot(
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
        )

    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()

    return RandomStateSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(),
        torch_cuda_state=cuda_state,
    )


def restore_random_state(snapshot: RandomStateSnapshot) -> None:
    """Restore a state captured by ``capture_random_state``."""

    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)

    torch = _optional_torch()
    if torch is None or snapshot.torch_state is None:
        return

    torch.set_rng_state(snapshot.torch_state)
    if snapshot.torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_state)


@contextlib.contextmanager
def reproducible(seed: int, *, deterministic_torch: bool = True) -> Iterator[None]:
    """Temporarily seed random generators, then restore their prior states."""

    snapshot = capture_random_state()
    set_random_seed(seed, deterministic_torch=deterministic_torch)
    try:
        yield
    finally:
        restore_random_state(snapshot)


def deterministic_l2_normalize(
    values: np.ndarray | list[float],
    *,
    axis: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Return L2-normalized values with zero vectors preserved as zeros."""

    array = np.asarray(values, dtype=np.float64)
    norm = np.sqrt(np.sum(np.square(array), axis=axis, keepdims=True))
    denominator = norm + epsilon
    normalized = array / denominator
    return np.where(norm == 0.0, 0.0, normalized)


def stable_softmax(
    values: np.ndarray | list[float],
    *,
    axis: int | None = None,
) -> np.ndarray:
    """Compute softmax after subtracting the max for numerical stability."""

    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

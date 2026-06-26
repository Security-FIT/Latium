"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

EPS = 1e-10


def local_zscore(vals: np.ndarray, window: int = 5, *, eps: float = EPS) -> np.ndarray:
    """Center-excluded local z-score used across detector/graph code."""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    half = max(0, int(window)) // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([arr[lo:i], arr[i + 1 : hi]])
        if len(neighbors) > 1:
            out[i] = (arr[i] - neighbors.mean()) / (neighbors.std() + eps)
    return out


def curvature(vals: np.ndarray) -> np.ndarray:
    """Absolute second-order finite-difference curvature."""
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 3:
        return np.zeros_like(arr)
    core = np.abs(arr[:-2] - 2.0 * arr[1:-1] + arr[2:])
    return np.concatenate([[0.0], core, [0.0]])


def rank01(vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    order = np.argsort(safe)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / max(1, n - 1)


def safe_mean(values: Iterable[Optional[float]]) -> float:
    cleaned = [float(v) for v in values if v is not None]
    return float(np.mean(cleaned)) if cleaned else 0.0

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


def _local_zscore_1d(
    vals: np.ndarray,
    window: int,
    *,
    eps: float,
    fill_value: float,
    absolute: bool,
    nan_safe: bool,
) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    half = max(0, int(window)) // 2
    out = np.full(n, fill_value, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([arr[lo:i], arr[i + 1 : hi]])
        if len(neighbors) > 1:
            if nan_safe:
                mu = np.nanmean(neighbors)
                sd = np.nanstd(neighbors)
            else:
                mu = neighbors.mean()
                sd = neighbors.std()
            value = (arr[i] - mu) / (sd + eps)
            out[i] = abs(value) if absolute else value
    return out


def local_zscore(
    vals: np.ndarray,
    window: int = 5,
    *,
    eps: float = EPS,
    axis: int | None = None,
    fill_value: float = 0.0,
    absolute: bool = False,
    nan_safe: bool = False,
) -> np.ndarray:
    """Center-excluded local z-score used across detector/graph code."""
    arr = np.asarray(vals, dtype=float)
    if axis is None or arr.ndim == 1:
        return _local_zscore_1d(
            arr,
            window,
            eps=eps,
            fill_value=fill_value,
            absolute=absolute,
            nan_safe=nan_safe,
        )

    moved = np.moveaxis(arr, axis, -1)
    out = np.empty_like(moved, dtype=float)
    for prefix in np.ndindex(moved.shape[:-1]):
        out[prefix] = _local_zscore_1d(
            moved[prefix],
            window,
            eps=eps,
            fill_value=fill_value,
            absolute=absolute,
            nan_safe=nan_safe,
        )
    return np.moveaxis(out, -1, axis)


def _curvature_1d(vals: np.ndarray, *, pad_value: float, absolute: bool) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 3:
        return np.full_like(arr, pad_value, dtype=float)
    core = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
    if absolute:
        core = np.abs(core)
    return np.concatenate([[pad_value], core, [pad_value]])


def curvature(
    vals: np.ndarray,
    *,
    axis: int | None = None,
    pad_value: float = 0.0,
    absolute: bool = True,
) -> np.ndarray:
    """Absolute second-order finite-difference curvature."""
    arr = np.asarray(vals, dtype=float)
    if axis is None or arr.ndim == 1:
        return _curvature_1d(arr, pad_value=pad_value, absolute=absolute)

    moved = np.moveaxis(arr, axis, -1)
    out = np.empty_like(moved, dtype=float)
    for prefix in np.ndindex(moved.shape[:-1]):
        out[prefix] = _curvature_1d(moved[prefix], pad_value=pad_value, absolute=absolute)
    return np.moveaxis(out, -1, axis)


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

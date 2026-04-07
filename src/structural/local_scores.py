from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

EPS = 1e-10


def ensure_odd_window(window: int) -> int:
    """Return a valid odd rolling-window size."""
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    return w


def normalize_windows(windows: Iterable[int], n_points: int) -> list[int]:
    """Normalize and de-duplicate windows, clipping each one to sequence length."""
    if n_points <= 0:
        return []
    out = []
    seen = set()
    for w in windows:
        ww = ensure_odd_window(min(max(1, int(w)), n_points))
        if ww > n_points:
            ww = ensure_odd_window(n_points)
        if ww not in seen:
            seen.add(ww)
            out.append(ww)
    return sorted(out)


def rank01(vals: np.ndarray) -> np.ndarray:
    """Rank-normalize values into [0, 1]."""
    n = vals.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    safe = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(safe)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(max(1, n - 1))


def rolling_z_abs(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Absolute rolling z-score over 1-D sequence."""
    n = vals.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    w = ensure_odd_window(window)
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = vals[lo:hi]
        finite = win[np.isfinite(win)]
        if finite.size == 0 or not np.isfinite(vals[i]):
            continue
        out[i] = abs((vals[i] - finite.mean()) / (finite.std() + EPS))
    return out


def rolling_mad_abs(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Absolute rolling modified z-score using median and MAD."""
    n = vals.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    w = ensure_odd_window(window)
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = vals[lo:hi]
        finite = win[np.isfinite(win)]
        if finite.size == 0 or not np.isfinite(vals[i]):
            continue
        med = np.median(finite)
        mad = np.median(np.abs(finite - med)) + EPS
        out[i] = abs(0.6745 * (vals[i] - med) / mad)
    return out


def local_score_bank(
    vals: np.ndarray,
    windows: Sequence[int],
    methods: Sequence[str] = ("z", "mad"),
) -> Dict[str, np.ndarray]:
    """Compute rolling local anomaly scores for all requested windows/methods."""
    n = vals.shape[0]
    ws = normalize_windows(windows, n_points=n)
    out: Dict[str, np.ndarray] = {}

    for w in ws:
        if "z" in methods:
            out[f"z_w{w}"] = rolling_z_abs(vals, window=w)
        if "mad" in methods:
            out[f"mad_w{w}"] = rolling_mad_abs(vals, window=w)

    if out:
        stacked = np.stack([rank01(v) for v in out.values()])
        out["mean_local_rank"] = np.mean(stacked, axis=0)
        out["max_local_rank"] = np.max(stacked, axis=0)
    else:
        out["mean_local_rank"] = np.zeros(n, dtype=np.float64)
        out["max_local_rank"] = np.zeros(n, dtype=np.float64)
    return out


def map_array_to_layers(layers: Sequence[int], vals: np.ndarray) -> Dict[int, float]:
    return {int(layer): float(vals[i]) for i, layer in enumerate(layers)}


def map_bank_to_layers(layers: Sequence[int], bank: Dict[str, np.ndarray]) -> Dict[str, Dict[int, float]]:
    return {key: map_array_to_layers(layers, vals) for key, vals in bank.items()}

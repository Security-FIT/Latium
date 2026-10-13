"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.common.linalg import gpu_svd_topk
from src.common.arrays import rank01
from src.structural.detectors.local_scores import rolling_z_abs


EPS = 1e-10

SCORE_PCS_NAMES = (
    "pcs_neighbor_var_scores",
    "pcs_next_jump_scores",
    "pcs_next_curvature_scores",
)
SCORE_PCS_CROSS_NAMES = ("pcs_cross_shift_scores",)


def spectral_decomposition(
    weights: Dict[int, torch.Tensor],
    max_k: int,
) -> Tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Top-k SVD per layer -> layers, singular values, right vectors, left vectors."""
    layers = sorted(weights.keys())
    if not layers:
        e2 = np.empty((0, 0), dtype=np.float32)
        e3 = np.empty((0, 0, 0), dtype=np.float32)
        return [], e2, e3, e3
    sv_list, vh_list, u_list = [], [], []
    for layer in layers:
        u, s, vh = gpu_svd_topk(weights[layer].detach(), k=int(max_k), niter=2)
        sv_list.append(s.numpy())
        vh_list.append(vh.numpy())
        u_list.append(u.numpy().T)
    k = min(s.shape[0] for s in sv_list)
    return (
        layers,
        np.stack([s[:k] for s in sv_list]),
        np.stack([v[:k] for v in vh_list]),
        np.stack([u[:k] for u in u_list]),
    )


def second_deriv_energy(x: np.ndarray) -> np.ndarray:
    """Per-layer curvature energy. Accepts 1-D or 2-D input."""
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    energy = np.zeros(n, dtype=np.float64)
    if n > 2:
        d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
        energy[1:-1] = (d2**2).sum(axis=1)
    if n > 1:
        energy[0], energy[-1] = energy[1], energy[-2]
    return energy


def sv_z_energy(sv: np.ndarray, top_k: int) -> np.ndarray:
    """Signal A: z-score top-k singular values across layers."""
    if sv.size == 0:
        return np.empty(0, dtype=np.float64)
    x = sv[:, : min(top_k, sv.shape[1])]
    return second_deriv_energy((x - x.mean(0)) / (x.std(0) + EPS))


def sv_ratio_energy(
    sv_proj: np.ndarray,
    sv_fc: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Signal B: top-k singular-value ratio curvature."""
    if sv_proj.size == 0 or sv_fc.size == 0:
        return np.empty(0, dtype=np.float64)
    k = min(top_k, sv_proj.shape[1], sv_fc.shape[1])
    return second_deriv_energy(sv_proj[:, :k] / (sv_fc[:, :k] + EPS))


def canonical_orient(rows: np.ndarray) -> np.ndarray:
    """Flip each row so the largest-abs element is positive."""
    if rows.size == 0:
        return rows
    out = rows.copy()
    pivots = np.argmax(np.abs(out), axis=1)
    signs = np.sign(out[np.arange(len(out)), pivots])
    signs[signs == 0] = 1.0
    return out * signs[:, None]


def weighted_pcs(v1: np.ndarray, v2: np.ndarray, w: np.ndarray) -> float:
    """Weighted signed PCS between two sets of vectors."""
    wn = w / (w.sum() + EPS)
    return float((wn * (v1 * v2).sum(1)).sum())


def pcs_pairwise_rank_cumsums(
    vh: np.ndarray,
    sv: np.ndarray,
    top_k: int,
    *,
    include_flip: bool = True,
    max_layer_distance: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cumulative pairwise weighted PCS components per rank.

    Returned tensors are shaped [k, n, n]:
      - dot_weight_cumsum[r, i, j] = sum_{t<=r} w_t(i,j) * dot_t(i,j)
      - flip_weight_cumsum[r, i, j] = sum_{t<=r} w_t(i,j) * 1(dot_t(i,j)<0)
      - weight_cumsum[r, i, j] = sum_{t<=r} w_t(i,j)
    """
    if vh.size == 0 or sv.size == 0:
        empty = np.empty((0, 0, 0), dtype=np.float64)
        return empty, empty, empty

    k = min(top_k, vh.shape[1], sv.shape[1])
    n = vh.shape[0]
    if k <= 0 or n <= 0:
        empty = np.empty((0, 0, 0), dtype=np.float64)
        return empty, empty, empty

    vecs = np.stack([canonical_orient(vh[i, :k]) for i in range(n)])
    ws = sv[:, :k]

    dot_weight_cumsum = np.zeros((k, n, n), dtype=np.float64)
    flip_weight_cumsum = (
        np.zeros((k, n, n), dtype=np.float64) if include_flip else np.empty((0, 0, 0), dtype=np.float64)
    )
    weight_cumsum = np.zeros((k, n, n), dtype=np.float64)

    for i in range(n):
        w_self = ws[i]
        w_self_cum = np.cumsum(w_self)
        weight_cumsum[:, i, i] = w_self_cum
        dot_weight_cumsum[:, i, i] = w_self_cum

        stop = n
        if max_layer_distance is not None:
            stop = min(n, i + max(1, int(max_layer_distance)) + 1)
        for j in range(i + 1, stop):
            dots = np.sum(vecs[i] * vecs[j], axis=1)
            w = 0.5 * (ws[i] + ws[j])

            w_cum = np.cumsum(w)
            dot_cum = np.cumsum(w * dots)
            flip_cum = np.cumsum(w * (dots < 0.0).astype(np.float64)) if include_flip else None

            weight_cumsum[:, i, j] = w_cum
            weight_cumsum[:, j, i] = w_cum
            dot_weight_cumsum[:, i, j] = dot_cum
            dot_weight_cumsum[:, j, i] = dot_cum
            if flip_cum is not None:
                flip_weight_cumsum[:, i, j] = flip_cum
                flip_weight_cumsum[:, j, i] = flip_cum

    return dot_weight_cumsum, flip_weight_cumsum, weight_cumsum


def score_pcs_signals_from_pairwise_cumsums(
    dot_cumulative: np.ndarray,
    weight_cumulative: np.ndarray,
    *,
    top_k: int,
    start: int = 0,
    end: int | None = None,
    neighbor_layers: int = 1,
) -> Dict[str, np.ndarray]:
    """Replay exactly the three within-projection PCS terms used by scoring."""
    end = start if end is None else int(end)
    n_expected = max(0, int(end) - int(start))
    zeros = np.zeros(n_expected, dtype=np.float64)
    empty = {name: zeros.copy() for name in SCORE_PCS_NAMES}

    dot_cumulative = np.asarray(dot_cumulative, dtype=np.float64)
    weight_cumulative = np.asarray(weight_cumulative, dtype=np.float64)
    if dot_cumulative.ndim != 3 or weight_cumulative.shape != dot_cumulative.shape or dot_cumulative.shape[0] == 0:
        return empty

    rank = min(max(1, int(top_k)), dot_cumulative.shape[0]) - 1
    weight = weight_cumulative[rank]
    pairwise = (dot_cumulative[rank] / (weight + EPS))[start:end, start:end]
    n = pairwise.shape[0]
    neighbor_variance = np.zeros(n, dtype=np.float64)
    radius = max(1, int(neighbor_layers))
    for index in range(n):
        neighbors = [other for other in range(max(0, index - radius), min(n, index + radius + 1)) if other != index]
        if neighbors:
            neighbor_variance[index] = pairwise[index, neighbors].var()

    next_scores = np.diag(pairwise, 1)
    if n:
        next_scores = np.concatenate([next_scores, next_scores[-1:] if next_scores.size else np.zeros(1)])
    jump = np.zeros(n, dtype=np.float64)
    if n > 1:
        delta = np.abs(np.diff(next_scores))
        jump[1:] = delta
        jump[0] = delta[0]

    return {
        "pcs_neighbor_var_scores": neighbor_variance,
        "pcs_next_jump_scores": jump,
        "pcs_next_curvature_scores": second_deriv_energy(next_scores),
    }


def score_pcs_cross_signals_from_rank_cumsums(
    dot_map: dict,
    weight_map: dict,
    layers: list[int],
    *,
    top_k: int,
    start: int = 0,
    end: int | None = None,
) -> Dict[str, np.ndarray]:
    """Replay only the cross-projection shift consumed by hybrid scoring."""
    end = len(layers) if end is None else int(end)
    n_expected = max(0, end - int(start))
    if not dot_map or not weight_map:
        return {"pcs_cross_shift_scores": np.zeros(n_expected, dtype=np.float64)}

    cross = np.zeros(len(layers), dtype=np.float64)
    for index, layer in enumerate(layers):
        dot = np.asarray(dot_map.get(str(layer), dot_map.get(layer, [])), dtype=np.float64)
        weight = np.asarray(weight_map.get(str(layer), weight_map.get(layer, [])), dtype=np.float64)
        rank_count = min(max(1, int(top_k)), len(dot), len(weight))
        if rank_count > 0:
            rank = rank_count - 1
            cross[index] = float(dot[rank] / (weight[rank] + EPS))
    return {"pcs_cross_shift_scores": 1.0 - cross[start:end]}


def sv_map(layers: list[int], sv: np.ndarray, top_k: int) -> Dict[int, list[float]]:
    """Serialize per-layer top-k singular values as plain Python lists."""
    if sv.size == 0:
        return {}
    k = min(top_k, sv.shape[1])
    out: Dict[int, list[float]] = {}
    for index, layer in enumerate(layers):
        out[int(layer)] = [float(x) for x in sv[index, :k]]
    return out


def score_pcs_cross_signals(
    vh_proj: np.ndarray,
    sv_proj: np.ndarray,
    vh_fc: np.ndarray,
    sv_fc: np.ndarray,
    top_k: int,
) -> Dict[str, np.ndarray]:
    """Compute only the cross-projection shift consumed by hybrid scoring."""
    n = int(sv_proj.shape[0]) if sv_proj.ndim == 2 else 0
    zeros = {"pcs_cross_shift_scores": np.zeros(n, dtype=np.float64)}
    if vh_proj.size == 0 or vh_fc.size == 0 or sv_proj.size == 0 or sv_fc.size == 0:
        return zeros
    if (
        n <= 0
        or vh_proj.shape[0] != n
        or vh_fc.shape[0] != n
        or sv_fc.shape[0] != n
        or vh_proj.shape[2] != vh_fc.shape[2]
    ):
        return zeros

    k = min(top_k, vh_proj.shape[1], vh_fc.shape[1], sv_proj.shape[1], sv_fc.shape[1])
    if k <= 0:
        return zeros
    cross = np.zeros(n, dtype=np.float64)
    for index in range(n):
        vp = canonical_orient(vh_proj[index, :k])
        vf = canonical_orient(vh_fc[index, :k])
        weight = 0.5 * (sv_proj[index, :k] + sv_fc[index, :k])
        cross[index] = weighted_pcs(vp, vf, weight)
    return {"pcs_cross_shift_scores": 1.0 - cross}


def rank01_mean(values: list[np.ndarray]) -> np.ndarray:
    """Average of rank-normalized arrays."""
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.mean(np.stack([rank01(v) for v in values]), axis=0)


def hybrid_scores(
    sz: np.ndarray,
    sr: np.ndarray,
    pcs: Dict[str, np.ndarray],
    pcs_cross: Dict[str, np.ndarray],
    has_fc: bool,
    rolling_window: int,
) -> Dict[str, np.ndarray]:
    """Composite detection score combining singular-value energy and PCS signals."""
    sv_z_rz = np.zeros_like(sz) if has_fc else rolling_z_abs(sz, window=rolling_window)
    sv_ratio_rz = rolling_z_abs(sr, window=rolling_window) if has_fc else np.zeros_like(sz)

    pcs_components = [
        pcs["pcs_next_jump_scores"],
        pcs["pcs_neighbor_var_scores"],
        pcs["pcs_next_curvature_scores"],
    ]
    if has_fc and pcs_cross["pcs_cross_shift_scores"].size:
        pcs_components.append(pcs_cross["pcs_cross_shift_scores"])

    pcs_rank = rank01_mean(pcs_components) if pcs_components else np.zeros_like(sz)
    if has_fc:
        sv_rank = rank01_mean([sz, sr])
        contradiction = sv_rank * (1.0 - pcs_rank)
        hybrid = 0.55 * sv_ratio_rz + 0.25 * contradiction + 0.20 * pcs_rank
    else:
        contradiction = np.zeros_like(sz)
        hybrid = 0.75 * sv_z_rz + 0.25 * pcs_rank

    return {
        "sv_z_rolling_z_scores": sv_z_rz,
        "sv_ratio_rolling_z_scores": sv_ratio_rz,
        "pcs_composite_rank_scores": pcs_rank,
        "sv_pcs_contradiction_scores": contradiction,
        "rome_hybrid_scores": hybrid,
    }

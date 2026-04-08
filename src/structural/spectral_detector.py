from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from src.utils import gpu_svd_topk
from src.structural.local_scores import local_score_bank

EPS = 1e-10

_PCS_NAMES = (
    "pcs_neighbor_mean_scores",
    "pcs_neighbor_shift_scores",
    "pcs_neighbor_var_scores",
    "pcs_neighbor_min_shift_scores",
    "pcs_neighbor_flip_fraction_scores",
    "pcs_next_scores",
    "pcs_next_shift_scores",
    "pcs_next_jump_scores",
    "pcs_next_curvature_scores",
)

_PCS_CROSS_NAMES = (
    "pcs_cross_scores",
    "pcs_cross_shift_scores",
    "pcs_cross_curvature_scores",
)


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

def _svd_all(weights: Dict[int, torch.Tensor], max_k: int) -> Tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Top-k SVD per layer -> (layers, sv[L,k], vh[L,k,d_out], u[L,k,d_in]).
    Uses shared GPU-first cached SVD helpers.
    """
    layers = sorted(weights.keys())
    if not layers:
        e2 = np.empty((0, 0), dtype=np.float32)
        e3 = np.empty((0, 0, 0), dtype=np.float32)
        return [], e2, e3, e3
    sv_list, vh_list, u_list = [], [], []
    for l in layers:
        u, s, vh = gpu_svd_topk(weights[l].detach(), k=int(max_k), niter=2)
        sv_list.append(s.numpy())
        vh_list.append(vh.numpy())
        u_list.append(u.numpy().T)
    k = min(*(s.shape[0] for s in sv_list))
    return (
        layers,
        np.stack([s[:k] for s in sv_list]),
        np.stack([v[:k] for v in vh_list]),
        np.stack([u[:k] for u in u_list]),
    )


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _second_deriv_energy(x: np.ndarray) -> np.ndarray:
    """Per-layer curvature energy. Accepts 1-D or 2-D input."""
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    energy = np.zeros(n, dtype=np.float64)
    if n > 2:
        d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
        energy[1:-1] = (d2 ** 2).sum(axis=1)
    if n > 1:
        energy[0], energy[-1] = energy[1], energy[-2]
    return energy


def _sv_z_energy(sv: np.ndarray, top_k: int) -> np.ndarray:
    """Signal A: z-score top-k SVs across layers -> curvature energy."""
    if sv.size == 0:
        return np.empty(0, dtype=np.float64)
    x = sv[:, :min(top_k, sv.shape[1])]
    return _second_deriv_energy((x - x.mean(0)) / (x.std(0) + EPS))


def _sv_ratio_energy(sv_proj: np.ndarray, sv_fc: np.ndarray, top_k: int) -> np.ndarray:
    """Signal B: ratio top-k SVs -> curvature energy."""
    if sv_proj.size == 0 or sv_fc.size == 0:
        return np.empty(0, dtype=np.float64)
    k = min(top_k, sv_proj.shape[1], sv_fc.shape[1])
    return _second_deriv_energy(sv_proj[:, :k] / (sv_fc[:, :k] + EPS))


# ---------------------------------------------------------------------------
# PCS helpers
# ---------------------------------------------------------------------------

def _canonical_orient(rows: np.ndarray) -> np.ndarray:
    """Flip each row so the largest-abs element is positive."""
    if rows.size == 0:
        return rows
    out = rows.copy()
    pivots = np.argmax(np.abs(out), axis=1)
    signs = np.sign(out[np.arange(len(out)), pivots])
    signs[signs == 0] = 1.0
    return out * signs[:, None]


def _wpcs(v1: np.ndarray, v2: np.ndarray, w: np.ndarray) -> float:
    """Weighted signed PCS between two sets of vectors."""
    wn = w / (w.sum() + EPS)
    return float((wn * (v1 * v2).sum(1)).sum())


def _wflip(v1: np.ndarray, v2: np.ndarray, w: np.ndarray) -> float:
    """Weighted fraction of PCs with negative dot product."""
    wn = w / (w.sum() + EPS)
    return float((wn * ((v1 * v2).sum(1) < 0)).sum())


def _pcs_signals(vh: np.ndarray, sv: np.ndarray, top_k: int, neighbor_layers: int) -> Dict[str, np.ndarray]:
    """Compute PCS directional signals over evaluated layers."""
    if vh.size == 0 or sv.size == 0:
        return {name: np.empty(0, dtype=np.float64) for name in _PCS_NAMES}

    k = min(top_k, vh.shape[1], sv.shape[1])
    n = vh.shape[0]
    radius = max(1, int(neighbor_layers))

    vecs = np.stack([_canonical_orient(vh[i, :k]) for i in range(n)])
    ws = sv[:, :k]

    n_mean = np.zeros(n); n_shift = np.zeros(n); n_var = np.zeros(n)
    n_min_shift = np.zeros(n); n_flip = np.zeros(n)

    for i in range(n):
        js = [j for j in range(max(0, i - radius), min(n, i + radius + 1)) if j != i]
        if not js:
            continue
        pcs = np.array([_wpcs(vecs[i], vecs[j], 0.5 * (ws[i] + ws[j])) for j in js])
        flips = np.array([_wflip(vecs[i], vecs[j], 0.5 * (ws[i] + ws[j])) for j in js])
        n_mean[i] = pcs.mean()
        n_shift[i] = (1.0 - pcs).mean()
        n_var[i] = pcs.var()
        n_min_shift[i] = 1.0 - pcs.min()
        n_flip[i] = flips.mean()

    pcs_next = np.zeros(n)
    if n > 1:
        for i in range(n - 1):
            pcs_next[i] = _wpcs(vecs[i], vecs[i + 1], 0.5 * (ws[i] + ws[i + 1]))
        pcs_next[-1] = pcs_next[-2]

    jump = np.zeros(n)
    if n > 1:
        d = np.abs(np.diff(pcs_next))
        jump[1:] = d
        jump[0] = d[0]

    return {
        "pcs_neighbor_mean_scores": n_mean,
        "pcs_neighbor_shift_scores": n_shift,
        "pcs_neighbor_var_scores": n_var,
        "pcs_neighbor_min_shift_scores": n_min_shift,
        "pcs_neighbor_flip_fraction_scores": n_flip,
        "pcs_next_scores": pcs_next,
        "pcs_next_shift_scores": 1.0 - pcs_next,
        "pcs_next_jump_scores": jump,
        "pcs_next_curvature_scores": _second_deriv_energy(pcs_next),
    }


def _pcs_pairwise_cache(vh: np.ndarray, sv: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute pairwise weighted PCS and flip-fraction matrices for post-hoc sweeps."""
    if vh.size == 0 or sv.size == 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty

    k = min(top_k, vh.shape[1], sv.shape[1])
    n = vh.shape[0]
    if k <= 0 or n <= 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty

    vecs = np.stack([_canonical_orient(vh[i, :k]) for i in range(n)])
    ws = sv[:, :k]

    pcs = np.eye(n, dtype=np.float64)
    flips = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            w = 0.5 * (ws[i] + ws[j])
            pcs_ij = _wpcs(vecs[i], vecs[j], w)
            flip_ij = _wflip(vecs[i], vecs[j], w)
            pcs[i, j] = pcs_ij
            pcs[j, i] = pcs_ij
            flips[i, j] = flip_ij
            flips[j, i] = flip_ij
    return pcs, flips


def _sv_map(layers: list[int], sv: np.ndarray, top_k: int) -> Dict[int, list[float]]:
    """Serialize per-layer top-k singular values as plain Python lists."""
    if sv.size == 0:
        return {}
    k = min(top_k, sv.shape[1])
    out: Dict[int, list[float]] = {}
    for i, layer in enumerate(layers):
        out[int(layer)] = [float(x) for x in sv[i, :k]]
    return out


def _pcs_cross_signals(
    vh_proj: np.ndarray, sv_proj: np.ndarray,
    vh_fc: np.ndarray, sv_fc: np.ndarray,
    top_k: int,
) -> Dict[str, np.ndarray]:
    """Cross-projection PCS: compare c_proj and c_fc principal directions per layer."""
    empty = {name: np.empty(0, dtype=np.float64) for name in _PCS_CROSS_NAMES}
    if vh_proj.size == 0 or vh_fc.size == 0 or sv_proj.size == 0 or sv_fc.size == 0:
        return empty

    n = sv_proj.shape[0]
    zeros = {name: np.zeros(n, dtype=np.float64) for name in _PCS_CROSS_NAMES}

    if (n <= 0 or vh_proj.shape[0] != n or vh_fc.shape[0] != n
            or sv_fc.shape[0] != n or vh_proj.shape[2] != vh_fc.shape[2]):
        return zeros

    k = min(top_k, vh_proj.shape[1], vh_fc.shape[1], sv_proj.shape[1], sv_fc.shape[1])
    if k <= 0:
        return zeros

    cross = np.zeros(n, dtype=np.float64)
    for i in range(n):
        vp = _canonical_orient(vh_proj[i, :k])
        vf = _canonical_orient(vh_fc[i, :k])
        w = 0.5 * (sv_proj[i, :k] + sv_fc[i, :k])
        cross[i] = _wpcs(vp, vf, w)

    return {
        "pcs_cross_scores": cross,
        "pcs_cross_shift_scores": 1.0 - cross,
        "pcs_cross_curvature_scores": _second_deriv_energy(cross),
    }


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _map_to_all(all_layers: list[int], eval_layers: list[int], vals: np.ndarray) -> Dict[int, float]:
    """Map per-eval-layer values to a dict covering all layers (0.0 for excluded)."""
    out = {l: 0.0 for l in all_layers}
    for i, l in enumerate(eval_layers):
        out[l] = float(vals[i])
    return out


def _rolling_z_abs(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Absolute rolling z-score over layer axis."""
    n = vals.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = vals[lo:hi]
        finite = win[np.isfinite(win)]
        if finite.size == 0 or not np.isfinite(vals[i]):
            continue
        out[i] = abs((vals[i] - finite.mean()) / (finite.std() + EPS))
    return out


def _rank01(vals: np.ndarray) -> np.ndarray:
    """Rank-normalize values into [0, 1]."""
    n = vals.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    safe = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(safe)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(max(1, n - 1))


def _rank01_mean(values: list[np.ndarray]) -> np.ndarray:
    """Average of rank-normalized arrays."""
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.mean(np.stack([_rank01(v) for v in values]), axis=0)


def _hybrid_scores(
    sz: np.ndarray,
    sr: np.ndarray,
    pcs: Dict[str, np.ndarray],
    pcs_cross: Dict[str, np.ndarray],
    has_fc: bool,
    rolling_window: int,
) -> Dict[str, np.ndarray]:
    """Composite detection score combining SV energy and PCS signals."""
    sv_z_rz = _rolling_z_abs(sz, window=rolling_window)
    sv_ratio_rz = _rolling_z_abs(sr, window=rolling_window) if has_fc else np.zeros_like(sz)
    sv_rank = _rank01_mean([sz, sr]) if has_fc else _rank01(sz)

    pcs_components = [
        pcs["pcs_next_jump_scores"],
        pcs["pcs_neighbor_var_scores"],
        pcs["pcs_next_curvature_scores"],
    ]
    if has_fc and pcs_cross["pcs_cross_shift_scores"].size:
        pcs_components.append(pcs_cross["pcs_cross_shift_scores"])

    pcs_rank = _rank01_mean(pcs_components) if pcs_components else np.zeros_like(sz)
    contradiction = sv_rank * (1.0 - pcs_rank)

    if has_fc:
        hybrid = 0.55 * sv_ratio_rz + 0.25 * contradiction + 0.20 * pcs_rank
    else:
        hybrid = 0.75 * sv_z_rz + 0.25 * pcs_rank

    return {
        "sv_z_rolling_z_scores": sv_z_rz,
        "sv_ratio_rolling_z_scores": sv_ratio_rz,
        "pcs_composite_rank_scores": pcs_rank,
        "sv_pcs_contradiction_scores": contradiction,
        "rome_hybrid_scores": hybrid,
    }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SpectralDetector:
    def __init__(
        self,
        top_k: int = 50,
        boundary: int = 2,
        trim_first_layers: int = 0,
        trim_last_layers: int = 0,
        trim_first: Optional[int] = None,
        trim_last: Optional[int] = None,
        neighbor_layers: int = 1,
        rolling_window: int = 5,
        local_windows: Sequence[int] = (3, 5, 7),
        store_raw_spectral: bool = True,
        raw_only: bool = False,
    ):
        self.top_k = top_k
        self.boundary = boundary
        self.trim_first_layers = max(0, int(trim_first if trim_first is not None else trim_first_layers))
        self.trim_last_layers = max(0, int(trim_last if trim_last is not None else trim_last_layers))
        self.neighbor_layers = max(1, int(neighbor_layers))
        self.rolling_window = max(1, int(rolling_window))
        self.local_windows = tuple(int(w) for w in local_windows)
        self.store_raw_spectral = bool(store_raw_spectral)
        self.raw_only = bool(raw_only)

    @property
    def _config(self) -> dict:
        return {
            "top_k": self.top_k,
            "boundary": self.boundary,
            "trim_first_layers": self.trim_first_layers,
            "trim_last_layers": self.trim_last_layers,
            "neighbor_layers": self.neighbor_layers,
            "rolling_window": self.rolling_window,
            "local_windows": list(self.local_windows),
            "store_raw_spectral": self.store_raw_spectral,
            "raw_only": self.raw_only,
        }

    def _trim(self, n: int) -> Tuple[int, int]:
        s = min(self.trim_first_layers, n)
        return s, n - min(self.trim_last_layers, n - s)

    def _empty_result(self, all_layers: list[int], excluded: list, evaluated: list) -> Dict:
        z = {l: 0.0 for l in all_layers}
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "sv_z_scores": dict(z),
            "sv_ratio_scores": dict(z),
            "sv_z_rolling_z_scores": dict(z),
            "sv_ratio_rolling_z_scores": dict(z),
            "pcs_composite_rank_scores": dict(z),
            "sv_pcs_contradiction_scores": dict(z),
            "rome_hybrid_scores": dict(z),
            **{name: dict(z) for name in _PCS_NAMES},
            **{name: dict(z) for name in _PCS_CROSS_NAMES},
            "local_window_scores": {},
            "has_fc_weights": False,
            "config": self._config,
            "excluded_layers": excluded,
            "evaluated_layers": evaluated,
        }

    def detect(
        self,
        weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        all_layers, sv_full, vh_full, u_full = _svd_all(weights, max_k=self.top_k)
        if not all_layers:
            return self._empty_result([], [], [])

        pcs_pairwise_full, pcs_flip_pairwise_full = _pcs_pairwise_cache(vh_full, sv_full, self.top_k)

        ts, te = self._trim(len(all_layers))
        if te <= ts:
            return self._empty_result(all_layers, list(all_layers), [])

        eval_layers = all_layers[ts:te]
        excl = all_layers[:ts] + all_layers[te:]
        sv, vh, u = sv_full[ts:te], vh_full[ts:te], u_full[ts:te]

        sv_fc_full = np.empty((0, 0), dtype=np.float64)
        vh_fc_full = np.empty((0, 0, 0), dtype=np.float64)
        has_fc = False
        if fc_weights is not None:
            fc_layers, sv_fc_full, vh_fc_full, _ = _svd_all(fc_weights, max_k=self.top_k)
            if fc_layers == all_layers:
                has_fc = True

        if self.raw_only:
            result = self._empty_result(all_layers, excl, eval_layers)
            result["has_fc_weights"] = has_fc
            result["config"] = self._config
            raw_payload = {
                "all_layers": [int(l) for l in all_layers],
                "top_k": int(min(self.top_k, sv_full.shape[1] if sv_full.ndim == 2 else 0)),
                "boundary": int(self.boundary),
                "sv_proj_topk": _sv_map(all_layers, sv_full, self.top_k),
                "pcs_pairwise": pcs_pairwise_full.tolist(),
                "pcs_flip_pairwise": pcs_flip_pairwise_full.tolist(),
            }
            if has_fc and sv_fc_full.size:
                raw_payload["sv_fc_topk"] = _sv_map(all_layers, sv_fc_full, self.top_k)
            result["raw_spectral"] = raw_payload
            return result

        # SV signals
        sz = _sv_z_energy(sv, self.top_k)
        sr = np.zeros_like(sz)
        pcs_cross = {name: np.zeros_like(sz) for name in _PCS_CROSS_NAMES}
        if has_fc:
            sv_fc = sv_fc_full[ts:te]
            pcs_cross = _pcs_cross_signals(u, sv, vh_fc_full[ts:te], sv_fc, self.top_k)
            sr = _sv_ratio_energy(sv, sv_fc, self.top_k)

        # PCS signals
        pcs = _pcs_signals(vh, sv, self.top_k, self.neighbor_layers)

        # Hybrid scoring
        hyb = _hybrid_scores(
            sz,
            sr,
            pcs,
            pcs_cross,
            has_fc,
            rolling_window=self.rolling_window,
        )
        scores = hyb["rome_hybrid_scores"]

        # Pick anomalous layer
        n = len(eval_layers)
        lo, hi = self.boundary, n - self.boundary
        cands = np.arange(lo, hi) if hi > lo else np.arange(n)
        best = int(cands[int(np.argmax(scores[cands]))])

        # Build result
        result = {
            "anomalous_layer": int(eval_layers[best]),
            "detection_score": float(scores[best]),
            "sv_z_scores": _map_to_all(all_layers, eval_layers, sz),
            "sv_ratio_scores": _map_to_all(all_layers, eval_layers, sr),
        }
        for key, vals in hyb.items():
            result[key] = _map_to_all(all_layers, eval_layers, vals)
        for name in _PCS_NAMES:
            result[name] = _map_to_all(all_layers, eval_layers, pcs[name])
        for name in _PCS_CROSS_NAMES:
            result[name] = _map_to_all(all_layers, eval_layers, pcs_cross[name])

        local_series = {
            "sv_z_scores": sz,
            "sv_ratio_scores": sr,
            "rome_hybrid_scores": scores,
            "pcs_next_jump_scores": pcs["pcs_next_jump_scores"],
            "pcs_neighbor_var_scores": pcs["pcs_neighbor_var_scores"],
            "pcs_next_curvature_scores": pcs["pcs_next_curvature_scores"],
        }
        result["local_window_scores"] = {
            name: {
                score_name: _map_to_all(all_layers, eval_layers, vals)
                for score_name, vals in local_score_bank(series, windows=self.local_windows).items()
            }
            for name, series in local_series.items()
        }

        result.update({
            "has_fc_weights": has_fc,
            "config": self._config,
            "excluded_layers": excl,
            "evaluated_layers": eval_layers,
        })

        if self.store_raw_spectral:
            raw_payload = {
                "all_layers": [int(l) for l in all_layers],
                "top_k": int(min(self.top_k, sv_full.shape[1] if sv_full.ndim == 2 else 0)),
                "boundary": int(self.boundary),
                "sv_proj_topk": _sv_map(all_layers, sv_full, self.top_k),
                "pcs_pairwise": pcs_pairwise_full.tolist(),
                "pcs_flip_pairwise": pcs_flip_pairwise_full.tolist(),
            }
            if has_fc and sv_fc_full.size:
                raw_payload["sv_fc_topk"] = _sv_map(all_layers, sv_fc_full, self.top_k)
            result["raw_spectral"] = raw_payload

        return result

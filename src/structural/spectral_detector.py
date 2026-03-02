from typing import Dict, Optional, Tuple

import numpy as np
import torch

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


def _zero_cross_signals(n: int) -> Dict[str, np.ndarray]:
    zeros = np.zeros(n, dtype=np.float64)
    return {
        "pcs_cross_scores": zeros,
        "pcs_cross_shift_scores": 1.0 - zeros,
        "pcs_cross_curvature_scores": _second_deriv_energy(zeros),
    }

# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

def _svd_all(weights: Dict[int, torch.Tensor], max_k: int) -> Tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Full SVD → (layers, sv[L,k], vh[L,k,d_out], u[L,k,d_in])."""
    layers = sorted(weights.keys())
    if not layers:
        e2 = np.empty((0, 0), dtype=np.float32)
        e3 = np.empty((0, 0, 0), dtype=np.float32)
        return [], e2, e3, e3
    sv_list, vh_list, u_list = [], [], []
    for l in layers:
        u, s, vh = torch.linalg.svd(weights[l].detach().float(), full_matrices=False)
        sv_list.append(s.cpu().numpy())
        vh_list.append(vh.cpu().numpy())
        u_list.append(u.cpu().numpy().T)
    k = min(int(max_k), *(s.shape[0] for s in sv_list))
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
    """Signal A: z-score top-k SVs across layers → curvature energy."""
    if sv.size == 0:
        return np.empty(0, dtype=np.float64)
    x = sv[:, :min(top_k, sv.shape[1])]
    return _second_deriv_energy((x - x.mean(0)) / (x.std(0) + EPS))


def _sv_ratio_energy(sv_proj: np.ndarray, sv_fc: np.ndarray, top_k: int) -> np.ndarray:
    """Signal B: ratio top-k SVs → curvature energy."""
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
    """Compute all PCS directional signals over evaluated layers."""
    if vh.size == 0 or sv.size == 0:
        return {name: np.empty(0, dtype=np.float64) for name in _PCS_NAMES}

    k = min(top_k, vh.shape[1], sv.shape[1])
    n = vh.shape[0]
    radius = max(1, int(neighbor_layers))

    vecs = np.stack([_canonical_orient(vh[i, :k]) for i in range(n)])
    ws = sv[:, :k]

    # Neighbor signals
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

    # Next-layer signals
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


def _pcs_cross_signals(
    vh_proj: np.ndarray, sv_proj: np.ndarray,
    vh_fc: np.ndarray, sv_fc: np.ndarray,
    top_k: int,
) -> Dict[str, np.ndarray]:
    """Cross-projection PCS: compare aligned c_proj and c_fc directions per layer."""
    if vh_proj.size == 0 or vh_fc.size == 0 or sv_proj.size == 0 or sv_fc.size == 0:
        return {name: np.empty(0, dtype=np.float64) for name in _PCS_CROSS_NAMES}

    n = sv_proj.shape[0]
    if n <= 0:
        return _zero_cross_signals(0)

    if vh_proj.shape[0] != n or vh_fc.shape[0] != n or sv_fc.shape[0] != n:
        return _zero_cross_signals(n)

    if vh_proj.shape[2] != vh_fc.shape[2]:
        return _zero_cross_signals(n)

    k = min(top_k, vh_proj.shape[1], vh_fc.shape[1], sv_proj.shape[1], sv_fc.shape[1])
    if k <= 0:
        return _zero_cross_signals(n)

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
# Shared utilities
# ---------------------------------------------------------------------------

def _map_to_all(all_layers: list[int], eval_layers: list[int], vals: np.ndarray) -> Dict[int, float]:
    out = {l: 0.0 for l in all_layers}
    for i, l in enumerate(eval_layers):
        out[l] = float(vals[i])
    return out


def _argmax_layer(layers: list[int], vals: np.ndarray) -> Optional[int]:
    if not layers or vals.size == 0:
        return None
    safe = np.where(np.isfinite(vals), vals, -np.inf)
    return None if np.all(np.isneginf(safe)) else int(layers[int(np.argmax(safe))])


def _null_result(all_layers: list[int], cfg: dict, excluded: list, evaluated: list) -> Dict:
    z = {l: 0.0 for l in all_layers}
    return {
        "anomalous_layer": None, "detection_score": 0.0,
        "anomalous_layer_rank_fusion": None, "detection_score_rank_fusion": 0.0,
        "anomalous_layer_hybrid": None, "detection_score_hybrid": 0.0,
        "sv_z_scores": dict(z), "sv_ratio_scores": dict(z),
        "sv_z_rolling_z_scores": dict(z),
        "sv_ratio_rolling_z_scores": dict(z),
        "pcs_composite_rank_scores": dict(z),
        "sv_pcs_contradiction_scores": dict(z),
        "rome_hybrid_scores": dict(z),
        **{name: dict(z) for name in _PCS_NAMES},
        **{name: dict(z) for name in _PCS_CROSS_NAMES},
        "has_fc_weights": False, "config": cfg,
        "excluded_layers": excluded, "evaluated_layers": evaluated,
    }


def _normalize_grid_values(values: Optional[list[int]], fallback: int, min_value: int = 1) -> list[int]:
    if values is None:
        return [max(min_value, fallback)]
    cleaned = sorted({int(v) for v in values if int(v) >= min_value})
    return cleaned or [max(min_value, fallback)]


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
            out[i] = 0.0
            continue
        mu = finite.mean()
        sd = finite.std() + EPS
        out[i] = abs((vals[i] - mu) / sd)
    return out


def _rank01(vals: np.ndarray) -> np.ndarray:
    """Rank-normalize values into [0, 1], higher means larger value."""
    n = vals.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    safe = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(safe)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    denom = float(max(1, n - 1))
    return ranks / denom


def _rank01_mean(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.mean(np.stack([_rank01(v) for v in values]), axis=0)


def _hybrid_rome_scores(
    sz: np.ndarray,
    sr: np.ndarray,
    pcs: Dict[str, np.ndarray],
    pcs_cross: Dict[str, np.ndarray],
    has_fc: bool,
) -> Dict[str, np.ndarray]:
    """Blind composite score with direct PCS agency + SV-vs-PCS contradiction."""
    sv_z_rz = _rolling_z_abs(sz, window=5)
    sv_ratio_rz = _rolling_z_abs(sr, window=5) if has_fc else np.zeros_like(sz)

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
    ):
        self.top_k = top_k
        self.boundary = boundary
        self.trim_first_layers = max(0, int(trim_first if trim_first is not None else trim_first_layers))
        self.trim_last_layers = max(0, int(trim_last if trim_last is not None else trim_last_layers))
        self.neighbor_layers = max(1, int(neighbor_layers))

    @property
    def _config(self) -> dict:
        return {"top_k": self.top_k, "boundary": self.boundary,
                "trim_first_layers": self.trim_first_layers,
                "trim_last_layers": self.trim_last_layers,
                "neighbor_layers": self.neighbor_layers}

    def _trim(self, n: int) -> Tuple[int, int]:
        s = min(self.trim_first_layers, n)
        return s, n - min(self.trim_last_layers, n - s)

    # ---- detect -----------------------------------------------------------

    def detect(
        self,
        weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        # Single SVD for c_proj weights (SVs + Vh)
        all_layers, sv_full, vh_full, u_full = _svd_all(weights, max_k=self.top_k)
        if not all_layers:
            return _null_result([], self._config, [], [])

        ts, te = self._trim(len(all_layers))
        if te <= ts:
            return _null_result(all_layers, self._config, list(all_layers), [])

        eval_layers = all_layers[ts:te]
        excl = all_layers[:ts] + all_layers[te:]
        sv, vh, u = sv_full[ts:te], vh_full[ts:te], u_full[ts:te]

        # Signal A & B
        sz = _sv_z_energy(sv, self.top_k)
        has_fc = False
        sr = np.zeros_like(sz)
        pcs_cross = {name: np.zeros_like(sz) for name in _PCS_CROSS_NAMES}
        if fc_weights is not None:
            fc_layers, sv_fc_full, vh_fc_full, _ = _svd_all(fc_weights, max_k=self.top_k)
            if fc_layers == all_layers:
                has_fc = True
                sv_fc, vh_fc = sv_fc_full[ts:te], vh_fc_full[ts:te]
                sr = _sv_ratio_energy(sv, sv_fc, self.top_k)
                pcs_cross = _pcs_cross_signals(u, sv, vh_fc, sv_fc, self.top_k)

        # PCS
        pcs = _pcs_signals(vh, sv, self.top_k, self.neighbor_layers)

        # Candidate selection
        n = len(eval_layers)
        lo, hi = self.boundary, n - self.boundary
        cands = np.arange(lo, hi) if hi > lo else np.arange(n)

        rank_z = np.empty(len(cands), dtype=np.float64)
        rank_z[np.argsort(-sz[cands])] = np.arange(len(cands), dtype=np.float64)
        if has_fc:
            rank_r = np.empty(len(cands), dtype=np.float64)
            rank_r[np.argsort(-sr[cands])] = np.arange(len(cands), dtype=np.float64)
            combined = (rank_z + rank_r) / 2.0
        else:
            combined = rank_z

        best_rank = int(np.argmin(combined))

        hybrid_scores = _hybrid_rome_scores(sz, sr, pcs, pcs_cross, has_fc)
        hybrid = hybrid_scores["rome_hybrid_scores"]
        rank_h = np.empty(len(cands), dtype=np.float64)
        rank_h[np.argsort(-hybrid[cands])] = np.arange(len(cands), dtype=np.float64)
        best_hybrid = int(np.argmin(rank_h))

        anomalous_layer_rank_fusion = int(eval_layers[int(cands[best_rank])])
        anomalous_layer_hybrid = int(eval_layers[int(cands[best_hybrid])])
        detection_score_rank_fusion = float(len(cands) - combined[best_rank])
        detection_score_hybrid = float(len(cands) - rank_h[best_hybrid])

        result = {
            "anomalous_layer": anomalous_layer_hybrid,
            "detection_score": detection_score_hybrid,
            "anomalous_layer_rank_fusion": anomalous_layer_rank_fusion,
            "detection_score_rank_fusion": detection_score_rank_fusion,
            "anomalous_layer_hybrid": anomalous_layer_hybrid,
            "detection_score_hybrid": detection_score_hybrid,
            "sv_z_scores": _map_to_all(all_layers, eval_layers, sz),
            "sv_ratio_scores": _map_to_all(all_layers, eval_layers, sr),
            "sv_z_rolling_z_scores": _map_to_all(all_layers, eval_layers, hybrid_scores["sv_z_rolling_z_scores"]),
            "sv_ratio_rolling_z_scores": _map_to_all(all_layers, eval_layers, hybrid_scores["sv_ratio_rolling_z_scores"]),
            "pcs_composite_rank_scores": _map_to_all(all_layers, eval_layers, hybrid_scores["pcs_composite_rank_scores"]),
            "sv_pcs_contradiction_scores": _map_to_all(all_layers, eval_layers, hybrid_scores["sv_pcs_contradiction_scores"]),
            "rome_hybrid_scores": _map_to_all(all_layers, eval_layers, hybrid_scores["rome_hybrid_scores"]),
        }
        for name in _PCS_NAMES:
            result[name] = _map_to_all(all_layers, eval_layers, pcs[name])
        for name in _PCS_CROSS_NAMES:
            result[name] = _map_to_all(all_layers, eval_layers, pcs_cross[name])

        result.update({"has_fc_weights": has_fc, "config": self._config,
                       "excluded_layers": excl, "evaluated_layers": eval_layers})
        return result

    # ---- analyze_grid -----------------------------------------------------

    def analyze_grid(
        self,
        weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
        top_k_values: Optional[list[int]] = None,
        neighbor_layers_values: Optional[list[int]] = None,
    ) -> Dict:
        ks = _normalize_grid_values(top_k_values, self.top_k)
        ns = _normalize_grid_values(neighbor_layers_values, self.neighbor_layers)
        base = {"top_k_values": ks, "neighbor_layers_values": ns,
                "trim_first_layers": self.trim_first_layers,
                "trim_last_layers": self.trim_last_layers}

        all_layers, sv_full, vh_full, u_full = _svd_all(weights, max_k=max(ks))
        if not all_layers:
            return {**base, "combos": {}, "has_fc_weights": False,
                    "excluded_layers": [], "evaluated_layers": []}

        ts, te = self._trim(len(all_layers))
        if te <= ts:
            return {**base, "combos": {}, "has_fc_weights": False,
                    "excluded_layers": list(all_layers), "evaluated_layers": []}

        evl = all_layers[ts:te]
        excl = all_layers[:ts] + all_layers[te:]
        sv, vh, u = sv_full[ts:te], vh_full[ts:te], u_full[ts:te]

        has_fc, sv_fc, vh_fc = False, None, None
        if fc_weights is not None:
            fc_l, sv_fc_full, vh_fc_full, _ = _svd_all(fc_weights, max_k=max(ks))
            if fc_l == all_layers and sv_fc_full.size:
                has_fc = True
                sv_fc = sv_fc_full[ts:te]
                vh_fc = vh_fc_full[ts:te]

        combos = {}
        for top_k in ks:
            for nl in ns:
                sz = _sv_z_energy(sv, top_k)
                sr = _sv_ratio_energy(sv, sv_fc, top_k) if has_fc else np.zeros_like(sz)
                pcs = _pcs_signals(vh, sv, top_k, nl)
                pcs_cross = _pcs_cross_signals(u, sv, vh_fc, sv_fc, top_k) if has_fc else {
                    name: np.zeros_like(sz) for name in _PCS_CROSS_NAMES
                }
                hybrid_scores = _hybrid_rome_scores(sz, sr, pcs, pcs_cross, has_fc)

                combo: Dict = {
                    "top_k": top_k,
                    "neighbor_layers": nl,
                    "sv_z_scores": _map_to_all(all_layers, evl, sz),
                    "sv_ratio_scores": _map_to_all(all_layers, evl, sr),
                    "sv_z_rolling_z_scores": _map_to_all(all_layers, evl, hybrid_scores["sv_z_rolling_z_scores"]),
                    "sv_ratio_rolling_z_scores": _map_to_all(all_layers, evl, hybrid_scores["sv_ratio_rolling_z_scores"]),
                    "pcs_composite_rank_scores": _map_to_all(all_layers, evl, hybrid_scores["pcs_composite_rank_scores"]),
                    "sv_pcs_contradiction_scores": _map_to_all(all_layers, evl, hybrid_scores["sv_pcs_contradiction_scores"]),
                    "rome_hybrid_scores": _map_to_all(all_layers, evl, hybrid_scores["rome_hybrid_scores"]),
                }
                for name in _PCS_NAMES:
                    combo[name] = _map_to_all(all_layers, evl, pcs[name])
                for name in _PCS_CROSS_NAMES:
                    combo[name] = _map_to_all(all_layers, evl, pcs_cross[name])
                combo["max_layers"] = {
                    "signal_a_layer": _argmax_layer(evl, sz),
                    "signal_b_layer": _argmax_layer(evl, sr),
                    "sv_z_rolling_z_layer": _argmax_layer(evl, hybrid_scores["sv_z_rolling_z_scores"]),
                    "sv_ratio_rolling_z_layer": _argmax_layer(evl, hybrid_scores["sv_ratio_rolling_z_scores"]),
                    "pcs_composite_rank_layer": _argmax_layer(evl, hybrid_scores["pcs_composite_rank_scores"]),
                    "sv_pcs_contradiction_layer": _argmax_layer(evl, hybrid_scores["sv_pcs_contradiction_scores"]),
                    "rome_hybrid_layer": _argmax_layer(evl, hybrid_scores["rome_hybrid_scores"]),
                    "pcs_shift_layer": _argmax_layer(evl, pcs["pcs_neighbor_shift_scores"]),
                    "pcs_var_layer": _argmax_layer(evl, pcs["pcs_neighbor_var_scores"]),
                    "pcs_min_shift_layer": _argmax_layer(evl, pcs["pcs_neighbor_min_shift_scores"]),
                    "pcs_flip_fraction_layer": _argmax_layer(evl, pcs["pcs_neighbor_flip_fraction_scores"]),
                    "pcs_next_shift_layer": _argmax_layer(evl, pcs["pcs_next_shift_scores"]),
                    "pcs_next_jump_layer": _argmax_layer(evl, pcs["pcs_next_jump_scores"]),
                    "pcs_curvature_layer": _argmax_layer(evl, pcs["pcs_next_curvature_scores"]),
                    "pcs_cross_shift_layer": _argmax_layer(evl, pcs_cross["pcs_cross_shift_scores"]),
                    "pcs_cross_curvature_layer": _argmax_layer(evl, pcs_cross["pcs_cross_curvature_scores"]),
                }
                combos[f"k{top_k}_n{nl}"] = combo

        return {**base, "combos": combos, "has_fc_weights": has_fc,
                "excluded_layers": excl, "evaluated_layers": evl}

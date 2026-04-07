from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from src.structural.local_scores import (
    local_score_bank,
    map_array_to_layers,
    map_bank_to_layers,
    rank01,
)
from src.utils import gpu_svdvals

EPS = 1e-10


def _sanitize(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _layer_profile(W: torch.Tensor, top_k: int) -> Dict[str, np.ndarray | float]:
    Wf = W.detach().float()
    S = gpu_svdvals(Wf).detach().cpu().numpy().astype(np.float64)
    if S.size == 0:
        vec = np.zeros(top_k, dtype=np.float64)
        return {
            "top1_energy": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "frob_norm": 0.0,
            "sv_vec": vec,
        }

    S2 = S ** 2
    total = S2.sum() + EPS
    p = S / (S.sum() + EPS)
    vec = np.zeros(top_k, dtype=np.float64)
    kk = min(top_k, S.size)
    vec[:kk] = S[:kk]
    vec = vec / (np.linalg.norm(vec) + EPS)

    frob_sq = float((Wf ** 2).sum().item())
    return {
        "top1_energy": float(S2[0] / total),
        "effective_rank": float(np.exp(-(p * np.log(p + EPS)).sum())),
        "stable_rank": float(frob_sq / ((S[0] ** 2) + EPS)),
        "frob_norm": float(np.sqrt(frob_sq)),
        "sv_vec": vec,
    }


def _profiles(
    weights: Dict[int, torch.Tensor],
    top_k: int,
) -> Tuple[list[int], Dict[int, Dict[str, np.ndarray | float]]]:
    layers = sorted(weights.keys())
    return layers, {l: _layer_profile(weights[l], top_k=top_k) for l in layers}


def _mirror_break_series(
    layers: Sequence[int],
    profile: Dict[int, Dict[str, np.ndarray | float]],
) -> Dict[str, np.ndarray]:
    n = len(layers)
    idx_map = {l: i for i, l in enumerate(layers)}

    top1_gap = np.zeros(n, dtype=np.float64)
    erank_gap = np.zeros(n, dtype=np.float64)
    stable_gap = np.zeros(n, dtype=np.float64)
    norm_log_gap = np.zeros(n, dtype=np.float64)
    sv_cos_shift = np.zeros(n, dtype=np.float64)

    for i, l in enumerate(layers):
        m = layers[n - 1 - i]
        j = idx_map[m]

        pi = profile[l]
        pm = profile[m]

        t1 = abs(float(pi["top1_energy"]) - float(pm["top1_energy"]))
        er = abs(float(pi["effective_rank"]) - float(pm["effective_rank"]))
        st = abs(float(pi["stable_rank"]) - float(pm["stable_rank"]))
        nl = abs(np.log(float(pi["frob_norm"]) + EPS) - np.log(float(pm["frob_norm"]) + EPS))

        vi = np.asarray(pi["sv_vec"], dtype=np.float64)
        vm = np.asarray(pm["sv_vec"], dtype=np.float64)
        cos = float(np.clip(np.dot(vi, vm), -1.0, 1.0))
        cs = 1.0 - cos

        top1_gap[i] = top1_gap[j] = t1
        erank_gap[i] = erank_gap[j] = er
        stable_gap[i] = stable_gap[j] = st
        norm_log_gap[i] = norm_log_gap[j] = nl
        sv_cos_shift[i] = sv_cos_shift[j] = cs

    return {
        "mirror_top1_gap": top1_gap,
        "mirror_effective_rank_gap": erank_gap,
        "mirror_stable_rank_gap": stable_gap,
        "mirror_norm_log_gap": norm_log_gap,
        "mirror_sv_cos_shift": sv_cos_shift,
    }


class MirrorSymmetryDetector:
    """Mirror-layer symmetry analytics and optional blind anomaly localization."""

    def __init__(
        self,
        top_k: int = 20,
        boundary: int = 2,
        local_windows: Sequence[int] = (3, 5, 7),
    ):
        self.top_k = max(2, int(top_k))
        self.boundary = max(0, int(boundary))
        self.local_windows = tuple(int(w) for w in local_windows)

    @property
    def _config(self) -> dict:
        return {
            "top_k": self.top_k,
            "boundary": self.boundary,
            "local_windows": list(self.local_windows),
        }

    def _empty(self) -> Dict:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "has_fc_weights": False,
            "mirror_layer_map": {},
            "proj_symmetry_series": {},
            "fc_symmetry_series": {},
            "symmetry_contrast_series": {},
            "raw_rank_score": {},
            "local_window_scores": {},
            "combined_score": {},
            "config": self._config,
        }

    def detect(
        self,
        proj_weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        layers, proj_profile = _profiles(proj_weights, top_k=self.top_k)
        if not layers:
            return self._empty()

        n = len(layers)
        mirror_map = {int(l): int(layers[n - 1 - i]) for i, l in enumerate(layers)}

        proj_sym = {k: _sanitize(v) for k, v in _mirror_break_series(layers, proj_profile).items()}

        has_fc = fc_weights is not None and all(l in fc_weights for l in layers)
        fc_sym: Dict[str, np.ndarray] = {}
        contrast: Dict[str, np.ndarray] = {}
        if has_fc:
            _, fc_profile = _profiles({l: fc_weights[l] for l in layers}, top_k=self.top_k)
            fc_sym = {k: _sanitize(v) for k, v in _mirror_break_series(layers, fc_profile).items()}
            for key, vals in proj_sym.items():
                if key in fc_sym:
                    contrast[f"{key}_proj_fc_abs_gap"] = _sanitize(np.abs(vals - fc_sym[key]))

        components = list(proj_sym.values()) + list(contrast.values())
        raw_rank = np.mean(np.stack([rank01(v) for v in components]), axis=0)

        local_inputs = {"raw_rank": raw_rank, **proj_sym, **contrast}
        local_scores = {
            name: local_score_bank(vals, windows=self.local_windows)
            for name, vals in local_inputs.items()
        }

        local_agg = np.mean(
            np.stack([bank["max_local_rank"] for bank in local_scores.values()]),
            axis=0,
        )
        combined = 0.45 * raw_rank + 0.55 * local_agg

        lo = min(self.boundary, n // 2)
        hi = n - min(self.boundary, n // 2)
        cands = np.arange(lo, hi) if hi > lo else np.arange(n)
        best = int(cands[int(np.argmax(combined[cands]))])

        return {
            "anomalous_layer": int(layers[best]),
            "detection_score": float(combined[best]),
            "has_fc_weights": bool(has_fc),
            "mirror_layer_map": mirror_map,
            "proj_symmetry_series": {
                k: map_array_to_layers(layers, vals) for k, vals in proj_sym.items()
            },
            "fc_symmetry_series": {
                k: map_array_to_layers(layers, vals) for k, vals in fc_sym.items()
            },
            "symmetry_contrast_series": {
                k: map_array_to_layers(layers, vals) for k, vals in contrast.items()
            },
            "raw_rank_score": map_array_to_layers(layers, _sanitize(raw_rank)),
            "local_window_scores": {
                name: map_bank_to_layers(layers, bank)
                for name, bank in local_scores.items()
            },
            "combined_score": map_array_to_layers(layers, _sanitize(combined)),
            "config": self._config,
        }

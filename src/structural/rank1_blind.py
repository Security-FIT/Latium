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


def _layer_rank1_metrics(W: torch.Tensor) -> Dict[str, float]:
    """Compute rank-1-sensitive structural metrics for one matrix."""
    Wf = W.detach().float()
    S = gpu_svdvals(Wf).detach().cpu().numpy().astype(np.float64)
    if S.size == 0:
        return {
            "top1_energy": 0.0,
            "top5_energy": 0.0,
            "gap12": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "rank1_residual": 1.0,
            "spectral_entropy": 0.0,
            "frob_norm": 0.0,
        }

    S2 = S ** 2
    total = S2.sum() + EPS
    s1 = S[0]
    s2 = S[1] if S.size > 1 else 0.0

    p = S / (S.sum() + EPS)
    q = S2 / total

    effective_rank = float(np.exp(-(p * np.log(p + EPS)).sum()))
    spectral_entropy = float(-(q * np.log(q + EPS)).sum() / np.log(max(2, S.size)))
    frob_sq = float((Wf ** 2).sum().item())

    top1_energy = float((s1 ** 2) / total)
    top5_energy = float(S2[:5].sum() / total)

    return {
        "top1_energy": top1_energy,
        "top5_energy": top5_energy,
        "gap12": float(s1 / (s2 + EPS)) if S.size > 1 else 0.0,
        "effective_rank": effective_rank,
        "stable_rank": float(frob_sq / ((s1 ** 2) + EPS)),
        "rank1_residual": float(1.0 - top1_energy),
        "spectral_entropy": spectral_entropy,
        "frob_norm": float(np.sqrt(frob_sq)),
    }


def _profiles_from_weights(
    weights: Dict[int, torch.Tensor],
) -> Tuple[list[int], Dict[int, Dict[str, float]], Dict[str, np.ndarray]]:
    layers = sorted(weights.keys())
    if not layers:
        return [], {}, {}

    per_layer = {int(l): _layer_rank1_metrics(weights[l]) for l in layers}
    metric_names = tuple(per_layer[layers[0]].keys())
    series = {
        name: np.array([per_layer[l][name] for l in layers], dtype=np.float64)
        for name in metric_names
    }
    return layers, per_layer, series


def _contrast_series(
    proj_series: Dict[str, np.ndarray],
    fc_series: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for metric in ("top1_energy", "top5_energy", "gap12", "effective_rank", "stable_rank", "rank1_residual"):
        if metric not in proj_series or metric not in fc_series:
            continue
        signed = proj_series[metric] - fc_series[metric]
        out[f"{metric}_signed_gap"] = signed
        out[f"{metric}_abs_gap"] = np.abs(signed)
    return out


class BlindRank1Detector:
    """Blind rank-1 detector using intrinsic signatures and local window anomalies."""

    def __init__(
        self,
        boundary: int = 2,
        local_windows: Sequence[int] = (3, 5, 7),
    ):
        self.boundary = max(0, int(boundary))
        self.local_windows = tuple(int(w) for w in local_windows)

    @property
    def _config(self) -> dict:
        return {
            "boundary": self.boundary,
            "local_windows": list(self.local_windows),
        }

    def _empty(self) -> Dict:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "has_fc_weights": False,
            "proj_layer_metrics": {},
            "fc_layer_metrics": {},
            "proj_series": {},
            "fc_series": {},
            "contrast_series": {},
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
        layers, proj_metrics, proj_series = _profiles_from_weights(proj_weights)
        if not layers:
            return self._empty()

        has_fc = fc_weights is not None and all(l in fc_weights for l in layers)
        fc_metrics: Dict[int, Dict[str, float]] = {}
        fc_series: Dict[str, np.ndarray] = {}
        if has_fc:
            _, fc_metrics, fc_series = _profiles_from_weights({l: fc_weights[l] for l in layers})

        component_arrays = [
            proj_series["top1_energy"],
            proj_series["top5_energy"],
            proj_series["gap12"],
            -proj_series["effective_rank"],
            -proj_series["stable_rank"],
            -proj_series["rank1_residual"],
        ]

        contrast = _contrast_series(proj_series, fc_series) if has_fc else {}
        if has_fc:
            for key in (
                "top1_energy_abs_gap",
                "gap12_abs_gap",
                "effective_rank_abs_gap",
                "stable_rank_abs_gap",
            ):
                if key in contrast:
                    component_arrays.append(contrast[key])

        raw_rank = np.mean(np.stack([rank01(v) for v in component_arrays]), axis=0)

        local_inputs = {
            "raw_rank": raw_rank,
            "top1_energy": proj_series["top1_energy"],
            "gap12": proj_series["gap12"],
            "rank1_residual": proj_series["rank1_residual"],
        }
        if has_fc:
            for key in ("top1_energy_abs_gap", "gap12_abs_gap", "effective_rank_abs_gap"):
                if key in contrast:
                    local_inputs[key] = contrast[key]

        local_scores = {
            name: local_score_bank(vals, windows=self.local_windows)
            for name, vals in local_inputs.items()
        }

        local_agg = np.mean(
            np.stack([bank["max_local_rank"] for bank in local_scores.values()]),
            axis=0,
        )
        combined = 0.45 * raw_rank + 0.55 * local_agg

        n = len(layers)
        lo = min(self.boundary, n // 2)
        hi = n - min(self.boundary, n // 2)
        cands = np.arange(lo, hi) if hi > lo else np.arange(n)
        best = int(cands[int(np.argmax(combined[cands]))])

        return {
            "anomalous_layer": int(layers[best]),
            "detection_score": float(combined[best]),
            "has_fc_weights": bool(has_fc),
            "proj_layer_metrics": {str(k): v for k, v in proj_metrics.items()},
            "fc_layer_metrics": {str(k): v for k, v in fc_metrics.items()},
            "proj_series": {
                k: map_array_to_layers(layers, v) for k, v in proj_series.items()
            },
            "fc_series": {
                k: map_array_to_layers(layers, v) for k, v in fc_series.items()
            },
            "contrast_series": {
                k: map_array_to_layers(layers, v) for k, v in contrast.items()
            },
            "raw_rank_score": map_array_to_layers(layers, raw_rank),
            "local_window_scores": {
                name: map_bank_to_layers(layers, bank)
                for name, bank in local_scores.items()
            },
            "combined_score": map_array_to_layers(layers, combined),
            "config": self._config,
        }

"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from src.structural.detectors.local_scores import (
    local_score_bank,
    map_array_to_layers,
    map_bank_to_layers,
    rank01,
)
from src.structural.detectors.profiles import matrix_profile, series_from_profiles

EPS = 1e-10


def derive_attention_templates(proj_layer_template: str) -> Dict[str, str]:
    """Best-effort architecture-aware mapping from MLP proj template to attention templates."""
    t = proj_layer_template
    out: Dict[str, str] = {}

    if ".mlp.c_proj" in t:
        out["o_proj"] = t.replace(".mlp.c_proj", ".attn.c_proj")
        out["qkv_combined"] = t.replace(".mlp.c_proj", ".attn.c_attn")
        return out

    if ".mlp.down_proj" in t:
        base = t.replace(".mlp.down_proj", "")
        out["q_proj"] = f"{base}.self_attn.q_proj"
        out["k_proj"] = f"{base}.self_attn.k_proj"
        out["v_proj"] = f"{base}.self_attn.v_proj"
        out["o_proj"] = f"{base}.self_attn.o_proj"
        return out

    if ".mlp.fc_out" in t:
        base = t.replace(".mlp.fc_out", "")
        out["q_proj"] = f"{base}.attn.q_proj"
        out["k_proj"] = f"{base}.attn.k_proj"
        out["v_proj"] = f"{base}.attn.v_proj"
        out["o_proj"] = f"{base}.attn.out_proj"
        return out

    return out


def split_qkv_weight(weight: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
    """Split combined qkv projection into q/k/v matrices when possible."""
    W = weight.detach().clone()
    m, n = W.shape

    if n % 3 == 0:
        q, k, v = torch.chunk(W, 3, dim=1)
        return {"q_proj": q, "k_proj": k, "v_proj": v}

    if m % 3 == 0:
        q, k, v = torch.chunk(W, 3, dim=0)
        return {"q_proj": q, "k_proj": k, "v_proj": v}

    return None


def _sanitize(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _metrics_to_series(
    layers: Sequence[int],
    layer_metrics: Dict[int, Dict[str, float]],
    metric_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    return series_from_profiles(list(layers), layer_metrics, tuple(metric_names))


class AttentionContrastDetector:
    """Blind detector using attention-vs-MLP structural contrast and local scores."""

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
            "available_attention_modules": [],
            "proj_series": {},
            "fc_series": {},
            "attention_aggregate_series": {},
            "attention_module_series": {},
            "contrast_series": {},
            "raw_rank_score": {},
            "local_window_scores": {},
            "combined_score": {},
            "config": self._config,
        }

    def detect(
        self,
        proj_weights: Dict[int, torch.Tensor],
        attention_weights: Dict[str, Dict[int, torch.Tensor]],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        layers = sorted(proj_weights.keys())
        if not layers:
            return self._empty()

        metric_names = ("frob_norm", "top1_energy", "effective_rank", "stable_rank", "norm_cv")

        proj_metrics = {l: matrix_profile(proj_weights[l]) for l in layers}
        proj_series = _metrics_to_series(layers, proj_metrics, metric_names)

        has_fc = fc_weights is not None and all(l in fc_weights for l in layers)
        fc_series: Dict[str, np.ndarray] = {}
        if has_fc:
            fc_metrics = {l: matrix_profile(fc_weights[l]) for l in layers}
            fc_series = _metrics_to_series(layers, fc_metrics, metric_names)

        # Per-module attention series
        module_series: Dict[str, Dict[str, np.ndarray]] = {}
        for module_name, layer_map in attention_weights.items():
            if not layer_map:
                continue
            module_layer_metrics = {l: matrix_profile(layer_map[l]) for l in layers if l in layer_map}
            if not module_layer_metrics:
                continue
            module_series[module_name] = _metrics_to_series(
                layers,
                module_layer_metrics,
                metric_names,
            )

        if not module_series:
            return self._empty()

        available_modules = sorted(module_series.keys())

        # Aggregate attention metrics across modules per layer.
        attn_aggr: Dict[str, np.ndarray] = {}
        for metric in metric_names:
            stacked = np.stack([module_series[m][metric] for m in available_modules], axis=0)
            attn_aggr[f"attn_{metric}_mean"] = np.nanmean(stacked, axis=0)
            attn_aggr[f"attn_{metric}_std"] = np.nanstd(stacked, axis=0)

        # Cross-family contrasts (edited layer should stand out because MLP changes, attention does not).
        contrast = {
            "proj_attn_norm_log_ratio": np.abs(
                np.log(proj_series["frob_norm"] + EPS) - np.log(attn_aggr["attn_frob_norm_mean"] + EPS)
            ),
            "proj_attn_top1_gap": np.abs(proj_series["top1_energy"] - attn_aggr["attn_top1_energy_mean"]),
            "proj_attn_rank_gap": np.abs(proj_series["effective_rank"] - attn_aggr["attn_effective_rank_mean"]),
            "proj_attn_stable_gap": np.abs(proj_series["stable_rank"] - attn_aggr["attn_stable_rank_mean"]),
            "attn_module_top1_dispersion": attn_aggr["attn_top1_energy_std"],
            "attn_module_rank_dispersion": attn_aggr["attn_effective_rank_std"],
        }
        if has_fc:
            contrast["proj_fc_top1_abs_gap"] = np.abs(proj_series["top1_energy"] - fc_series["top1_energy"])
            contrast["fc_attn_top1_gap"] = np.abs(fc_series["top1_energy"] - attn_aggr["attn_top1_energy_mean"])

        contrast = {k: _sanitize(v) for k, v in contrast.items()}
        raw_rank = np.mean(np.stack([rank01(v) for v in contrast.values()]), axis=0)

        local_inputs = {"raw_rank": raw_rank, **contrast}
        local_scores = {name: local_score_bank(vals, windows=self.local_windows) for name, vals in local_inputs.items()}
        local_agg = np.mean(
            np.stack([bank["max_local_rank"] for bank in local_scores.values()]),
            axis=0,
        )
        combined = 0.4 * raw_rank + 0.6 * local_agg

        n = len(layers)
        lo = min(self.boundary, n // 2)
        hi = n - min(self.boundary, n // 2)
        cands = np.arange(lo, hi) if hi > lo else np.arange(n)
        best = int(cands[int(np.argmax(combined[cands]))])

        return {
            "anomalous_layer": int(layers[best]),
            "detection_score": float(combined[best]),
            "has_fc_weights": bool(has_fc),
            "available_attention_modules": available_modules,
            "proj_series": {k: map_array_to_layers(layers, _sanitize(v)) for k, v in proj_series.items()},
            "fc_series": {k: map_array_to_layers(layers, _sanitize(v)) for k, v in fc_series.items()},
            "attention_aggregate_series": {k: map_array_to_layers(layers, _sanitize(v)) for k, v in attn_aggr.items()},
            "attention_module_series": {
                module_name: {
                    metric: map_array_to_layers(layers, _sanitize(vals)) for metric, vals in module_data.items()
                }
                for module_name, module_data in module_series.items()
            },
            "contrast_series": {k: map_array_to_layers(layers, _sanitize(v)) for k, v in contrast.items()},
            "raw_rank_score": map_array_to_layers(layers, _sanitize(raw_rank)),
            "local_window_scores": {name: map_bank_to_layers(layers, bank) for name, bank in local_scores.items()},
            "combined_score": map_array_to_layers(layers, _sanitize(combined)),
            "config": self._config,
        }

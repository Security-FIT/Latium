"""
Artifact edit-presence detector over captured matrix profiles.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.structural.detectors.local_scores import local_score_bank, map_array_to_layers, map_bank_to_layers, rank01

EPS = 1e-10


def _robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return 0.6745 * (x - med) / mad


def _normalized_entropy(vals: np.ndarray) -> float:
    x = np.array(vals, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    if x.size <= 1:
        return 0.0
    total = x.sum()
    if total <= 0.0:
        return 1.0
    p = x / total
    ent = -np.sum(p * np.log(np.clip(p, EPS, 1.0)))
    return float(ent / np.log(x.size))


def edit_presence_config(
    detection_threshold: float,
    min_peak_robust_z: float,
    min_margin: float,
    local_windows: Sequence[int],
) -> Dict[str, object]:
    return {
        "detection_threshold": float(detection_threshold),
        "min_peak_robust_z": float(min_peak_robust_z),
        "min_margin": float(min_margin),
        "local_windows": [int(window) for window in local_windows],
    }


def _profile(
    profiles: Mapping[int | str, Mapping[str, float]],
    layer: int,
) -> Mapping[str, float]:
    return profiles.get(layer) or profiles.get(str(layer)) or {}


def detect_edit_presence_from_profiles(
    proj_metrics: Mapping[int | str, Mapping[str, float]],
    *,
    fc_metrics: Optional[Mapping[int | str, Mapping[str, float]]] = None,
    modified_spectral: Optional[Dict] = None,
    detection_threshold: float = 0.58,
    min_peak_robust_z: float = 2.0,
    min_margin: float = 0.08,
    local_windows: Sequence[int] = (3, 5, 7),
) -> Dict:
    layers = sorted(int(layer) for layer in proj_metrics.keys())
    config = edit_presence_config(
        detection_threshold,
        min_peak_robust_z,
        min_margin,
        local_windows,
    )
    if not layers:
        return {
            "model_detected": False,
            "is_edited": False,
            "confidence": 0.0,
            "score": 0.0,
            "reason": "no_layers",
            "config": config,
        }

    top1 = np.array(
        [float(_profile(proj_metrics, layer).get("top1_energy", 0.0)) for layer in layers],
        dtype=np.float64,
    )
    gap = np.array(
        [float(_profile(proj_metrics, layer).get("spectral_gap", 0.0)) for layer in layers],
        dtype=np.float64,
    )
    erank = np.array(
        [float(_profile(proj_metrics, layer).get("effective_rank", 0.0)) for layer in layers],
        dtype=np.float64,
    )
    norm_cv = np.array(
        [float(_profile(proj_metrics, layer).get("norm_cv", 0.0)) for layer in layers],
        dtype=np.float64,
    )
    log_frob = np.log(
        np.array(
            [float(_profile(proj_metrics, layer).get("frob_norm", 0.0)) for layer in layers],
            dtype=np.float64,
        )
        + EPS
    )

    top1_z = np.maximum(_robust_z(top1), 0.0)
    gap_z = np.maximum(_robust_z(gap), 0.0)
    cv_z = np.maximum(_robust_z(norm_cv), 0.0)
    erank_inv_z = np.maximum(-_robust_z(erank), 0.0)
    frob_z = np.maximum(_robust_z(log_frob), 0.0)

    component_arrays = {
        "top1_energy_rank": rank01(top1_z),
        "spectral_gap_rank": rank01(gap_z),
        "norm_cv_rank": rank01(cv_z),
        "effective_rank_inverse_rank": rank01(erank_inv_z),
        "log_frob_rank": rank01(frob_z),
    }

    has_fc = fc_metrics is not None and all(layer in fc_metrics or str(layer) in fc_metrics for layer in layers)
    resolved_fc_metrics: dict[int, Mapping[str, float]] = {}
    if has_fc and fc_metrics is not None:
        resolved_fc_metrics = {layer: _profile(fc_metrics, layer) for layer in layers}
        proj_fc_top1_gap = np.abs(
            np.array(
                [float(_profile(proj_metrics, layer).get("top1_energy", 0.0)) for layer in layers],
                dtype=np.float64,
            )
            - np.array(
                [float(resolved_fc_metrics[layer].get("top1_energy", 0.0)) for layer in layers],
                dtype=np.float64,
            )
        )
        proj_fc_erank_gap = np.abs(
            np.array(
                [float(_profile(proj_metrics, layer).get("effective_rank", 0.0)) for layer in layers],
                dtype=np.float64,
            )
            - np.array(
                [float(resolved_fc_metrics[layer].get("effective_rank", 0.0)) for layer in layers],
                dtype=np.float64,
            )
        )
        component_arrays["proj_fc_top1_abs_gap_rank"] = rank01(proj_fc_top1_gap)
        component_arrays["proj_fc_erank_abs_gap_rank"] = rank01(proj_fc_erank_gap)

    if modified_spectral is not None:
        hyb_map = modified_spectral.get("rome_hybrid_scores", {})
        hyb = np.array(
            [float(hyb_map.get(layer, hyb_map.get(str(layer), 0.0))) for layer in layers],
            dtype=np.float64,
        )
        component_arrays["spectral_hybrid_rank"] = rank01(hyb)

    raw_rank = np.mean(np.stack(list(component_arrays.values())), axis=0)
    local_bank = local_score_bank(raw_rank, windows=tuple(int(w) for w in local_windows))
    local_rank = local_bank["max_local_rank"]
    combined = 0.55 * raw_rank + 0.45 * local_rank

    best_idx = int(np.argmax(combined))
    peak_layer = int(layers[best_idx])
    peak_score = float(combined[best_idx])
    sorted_scores = np.sort(combined)
    second = float(sorted_scores[-2]) if sorted_scores.size > 1 else peak_score
    margin = float(max(0.0, peak_score - second))
    entropy = _normalized_entropy(combined)

    combined_z = _robust_z(combined)
    peak_robust_z = float(combined_z[best_idx]) if combined_z.size else 0.0

    score = float(0.55 * peak_score + 0.30 * margin + 0.15 * (1.0 - entropy))

    adaptive_threshold = float(np.quantile(combined, 0.85)) if combined.size else 0.0
    is_edited = bool(
        score >= float(detection_threshold)
        and peak_robust_z >= float(min_peak_robust_z)
        and margin >= float(min_margin)
        and peak_score >= adaptive_threshold
    )

    return {
        "model_detected": is_edited,
        "is_edited": is_edited,
        "confidence": float(np.clip(score, 0.0, 1.0)),
        "score": score,
        "reason": "ok" if is_edited else "below_presence_threshold",
        "peak_layer": peak_layer,
        "anomalous_layer": peak_layer,
        "detection_score": score,
        "peak_combined_score": peak_score,
        "peak_combined_robust_z": peak_robust_z,
        "peak_robust_z": peak_robust_z,
        "confidence_margin": margin,
        "entropy": entropy,
        "adaptive_threshold": adaptive_threshold,
        "peak_rank_one_score": float(_profile(proj_metrics, peak_layer).get("top1_energy", 0.0)),
        "peak_effective_rank": float(_profile(proj_metrics, peak_layer).get("effective_rank", 0.0)),
        "has_fc_weights": bool(has_fc),
        "proj_layer_metrics": {str(layer): dict(_profile(proj_metrics, layer)) for layer in layers},
        "fc_layer_metrics": {str(layer): dict(resolved_fc_metrics.get(layer, {})) for layer in layers},
        "component_series": {name: map_array_to_layers(layers, vals) for name, vals in component_arrays.items()},
        "raw_rank_score": map_array_to_layers(layers, raw_rank),
        "local_window_scores": map_bank_to_layers(layers, local_bank),
        "combined_score": map_array_to_layers(layers, combined),
        "config": config,
        "thresholds": {
            "detection_threshold": float(detection_threshold),
            "min_peak_robust_z": float(min_peak_robust_z),
            "min_margin": float(min_margin),
            "adaptive_quantile": 0.85,
        },
    }


__all__ = ["detect_edit_presence_from_profiles", "edit_presence_config"]

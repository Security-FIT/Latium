"""
Composite layer detector implementation.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

from src.common.arrays import curvature, local_zscore

EPS = 1e-10


def _feature_array(
    layer_features: dict,
    layers: list[str],
    name: str,
) -> np.ndarray:
    return np.array([layer_features[layer][name] for layer in layers], dtype=float)


def _spectral_signal_array(
    spectral: dict,
    layers: list[str],
    name: str,
) -> Optional[np.ndarray]:
    layer_map = spectral.get(name)
    if not isinstance(layer_map, dict):
        return None
    try:
        values = np.array([float(layer_map.get(layer, layer_map.get(str(layer), 0.0))) for layer in layers])
    except (TypeError, ValueError):
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.max(np.abs(finite)) <= EPS:
        return None
    return values


def _peak(values: np.ndarray, layers: list[int]) -> tuple[int, int, float]:
    index = int(np.argmax(values))
    score = float((values[index] - values.mean()) / (values.std() + EPS))
    return index, layers[index], score


def detect_layer(
    test: dict,
    *,
    trim_first: int,
    trim_last: int,
    small_window: int,
    large_window: int,
    te_window: int,
    nc_window: int,
    signal_a_confirm_z_min: float,
    signal_ab_boundary_width: int,
    signal_ab_cluster_span: int,
) -> Tuple[Optional[int], str, Dict]:
    features = test["blind_detection"]["layer_features"]
    layers = sorted(features, key=int)
    start = max(0, int(trim_first))
    end = len(layers) - max(0, int(trim_last))
    if end <= start:
        return None, "empty", {}

    evaluated = [int(layer) for layer in layers[start:end]]
    small_window = max(3, int(small_window))
    large_window = max(small_window + 2, int(large_window))
    te_window = max(3, int(te_window))
    nc_window = max(3, int(nc_window))

    spectral_gap_full = _feature_array(features, layers, "spectral_gap")
    top1_full = _feature_array(features, layers, "top1_energy")
    norm_cv_full = _feature_array(features, layers, "norm_cv")
    effective_rank_full = _feature_array(features, layers, "effective_rank")
    alignment_full = _feature_array(features, layers, "row_alignment")

    spectral_gap = spectral_gap_full[start:end]
    top1_local = np.abs(local_zscore(top1_full, te_window))[start:end]
    gap_local_small = np.abs(local_zscore(spectral_gap_full, small_window))[start:end]
    gap_local_large = np.abs(local_zscore(spectral_gap_full, large_window))[start:end]
    norm_local = np.abs(local_zscore(norm_cv_full, nc_window))[start:end]
    rank_curvature = curvature(effective_rank_full)[start:end]
    alignment = alignment_full[start:end]

    sg_i, sg_layer, sg_z = _peak(spectral_gap, evaluated)
    te_i, te_layer, te_z = _peak(top1_local, evaluated)
    small_i, small_layer, small_z = _peak(gap_local_small, evaluated)
    large_i, large_layer, large_z = _peak(gap_local_large, evaluated)
    nc_i, nc_layer, nc_z = _peak(norm_local, evaluated)
    rank_i, rank_layer, rank_z = _peak(rank_curvature, evaluated)
    align_i, align_layer, align_z = _peak(alignment, evaluated)

    info = {
        "windows": {
            "trim_first": start,
            "trim_last": max(0, int(trim_last)),
            "small_window": small_window,
            "large_window": large_window,
            "te_window": te_window,
            "nc_window": nc_window,
        },
        "sg_raw": {"layer": sg_layer, "z": round(sg_z, 2), "idx": sg_i},
        "te_local": {"layer": te_layer, "z": round(te_z, 2), "idx": te_i},
        "sg_local_small": {
            "layer": small_layer,
            "z": round(small_z, 2),
            "idx": small_i,
        },
        "sg_local_large": {
            "layer": large_layer,
            "z": round(large_z, 2),
            "idx": large_i,
        },
        "nc_local": {"layer": nc_layer, "z": round(nc_z, 2), "idx": nc_i},
        "er_curv": {"layer": rank_layer, "z": round(rank_z, 2), "idx": rank_i},
        "ra_raw": {"layer": align_layer, "z": round(align_z, 2), "idx": align_i},
        "eval_layers": evaluated,
    }

    spectral = test.get("spectral_detection", {})
    signal_a = _spectral_signal_array(spectral, layers, "sv_z_scores")
    signal_b = _spectral_signal_array(spectral, layers, "sv_ratio_scores")
    a_i = a_layer = b_i = b_layer = None
    a_z = b_z = 0.0
    if signal_a is not None:
        a_i, a_layer, a_z = _peak(signal_a[start:end], evaluated)
        info["signal_a"] = {"layer": a_layer, "z": round(a_z, 2), "idx": a_i}
    if signal_b is not None:
        b_i, b_layer, b_z = _peak(signal_b[start:end], evaluated)
        info["signal_b"] = {"layer": b_layer, "z": round(b_z, 2), "idx": b_i}

    detected: Optional[int] = None
    method = "none"
    detected_index: Optional[int] = None
    if sg_layer == te_layer:
        detected, method, detected_index = sg_layer, "agree", sg_i
    elif abs(small_i - sg_i) <= 1 and abs(small_i - te_i) > 1:
        detected, method, detected_index = sg_layer, f"sg(lz{small_window})", sg_i
    elif abs(small_i - te_i) <= 1 and abs(small_i - sg_i) > 1:
        detected, method, detected_index = te_layer, f"te(lz{te_window})", te_i
    elif abs(large_i - sg_i) <= 1 and abs(large_i - te_i) > 1:
        detected, method, detected_index = sg_layer, f"sg(lz{large_window})", sg_i
    elif abs(large_i - te_i) <= 1 and abs(large_i - sg_i) > 1:
        detected, method, detected_index = te_layer, f"te(lz{large_window})", te_i

    if detected is None:
        rho, _ = stats.spearmanr(np.arange(len(evaluated)), spectral_gap)
        info["rho"] = round(float(rho), 3)
        if abs(rho) > 0.3 and abs(small_i - large_i) <= 1:
            if small_z >= large_z:
                detected, method, detected_index = small_layer, f"lz_cons({small_window})", small_i
            else:
                detected, method, detected_index = large_layer, f"lz_cons({large_window})", large_i
        elif abs(rho) > 0.3 and abs(large_i - te_i) <= 1:
            detected, method, detected_index = te_layer, "te(trend)", te_i
        elif abs(rho) > 0.3:
            detected, method, detected_index = large_layer, "s7(trend)", large_i
        else:
            detected, method, detected_index = sg_layer, "sg(fb)", sg_i

    if a_layer is not None and a_z >= float(signal_a_confirm_z_min) and a_layer == te_layer:
        info["spectral_support"] = {"kind": "signal_a", "reason": "te_alignment"}
        return a_layer, "signal_a", info

    if a_i is not None and b_i is not None:
        peak_indices = [sg_i, a_i, b_i]
        cluster_span = max(peak_indices) - min(peak_indices)
        boundary_width = int(signal_ab_boundary_width)
        early_limit = min(boundary_width - 1, len(evaluated) - 1)
        late_limit = max(0, len(evaluated) - boundary_width)
        if cluster_span <= int(signal_ab_cluster_span) and (
            max(peak_indices) <= early_limit or min(peak_indices) >= late_limit
        ):
            info["spectral_support"] = {
                "kind": "signal_ab_boundary",
                "reason": "boundary_cluster",
                "cluster_span": cluster_span,
            }
            return sg_layer, "signal_ab_boundary", info

    if detected_index is not None:
        info["detected_index"] = detected_index
    return detected, method, info

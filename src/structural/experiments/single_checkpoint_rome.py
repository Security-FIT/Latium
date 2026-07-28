"""Transparent checkpoint-level statistics for suspect-only ROME research."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROBUST_PEAK_VERSION = "rome-single-checkpoint-robust-peak-v1"
PEAK_PROMINENCE_VERSION = "rome-single-checkpoint-peak-prominence-v1"
LOCAL_PROMINENCE_VERSION = "rome-single-checkpoint-local-prominence-v1"


def dtype_scale_bound(
    values: Sequence[float],
    *,
    dtype: str = "float32",
) -> float:
    """Return a dtype- and profile-length-derived zero-scale safeguard."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("A non-empty one-dimensional score vector is required")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Layer scores must be finite")
    try:
        info = np.finfo(np.dtype(dtype))
    except ValueError as exc:
        raise ValueError(f"Unsupported score dtype: {dtype}") from exc
    magnitude = max(1.0, float(np.max(np.abs(array))))
    return float(info.eps) * max(1, int(array.size)) * magnitude


def robust_peak_statistic(
    layer_scores: Mapping[int | str, float],
    *,
    eligible_layers: Sequence[int],
    score_dtype: str = "float32",
) -> dict[str, Any]:
    """Compute the one-statistic median/MAD peak candidate."""
    eligible = sorted({int(layer) for layer in eligible_layers})
    if not eligible:
        raise ValueError("At least one eligible layer is required")
    missing = [layer for layer in eligible if str(layer) not in layer_scores and layer not in layer_scores]
    if missing:
        raise ValueError(f"Missing scores for eligible layers: {missing[:8]}")
    values = np.asarray(
        [float(layer_scores[str(layer)] if str(layer) in layer_scores else layer_scores[layer]) for layer in eligible],
        dtype=np.float64,
    )
    if not bool(np.isfinite(values).all()):
        raise ValueError("Layer scores must be finite")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale_bound = dtype_scale_bound(values, dtype=score_dtype)
    scale = max(mad, scale_bound)
    peak = float(np.max(values))
    selected_layer = min(layer for layer, value in zip(eligible, values, strict=True) if float(value) == peak)
    return {
        "candidate_version": ROBUST_PEAK_VERSION,
        "selected_layer": int(selected_layer),
        "peak": peak,
        "center": center,
        "mad": mad,
        "scale_bound": scale_bound,
        "z_peak": (peak - center) / scale,
    }


def peak_prominence_statistic(
    layer_scores: Mapping[int | str, float],
    *,
    eligible_layers: Sequence[int],
    score_dtype: str = "float32",
) -> dict[str, Any]:
    """Measure the highest M3 score relative to the second-highest score."""
    eligible = sorted({int(layer) for layer in eligible_layers})
    if len(eligible) < 2:
        raise ValueError("At least two eligible layers are required")
    missing = [layer for layer in eligible if str(layer) not in layer_scores and layer not in layer_scores]
    if missing:
        raise ValueError(f"Missing scores for eligible layers: {missing[:8]}")
    pairs = sorted(
        (
            float(layer_scores[str(layer)] if str(layer) in layer_scores else layer_scores[layer]),
            layer,
        )
        for layer in eligible
    )
    if not all(math.isfinite(value) for value, _ in pairs):
        raise ValueError("Layer scores must be finite")
    peak, selected_layer = max(pairs, key=lambda item: (item[0], -item[1]))
    remaining = [item for item in pairs if item[1] != selected_layer]
    second, second_layer = max(remaining, key=lambda item: (item[0], -item[1]))
    scale_bound = dtype_scale_bound([value for value, _ in pairs], dtype=score_dtype)
    return {
        "candidate_version": PEAK_PROMINENCE_VERSION,
        "selected_layer": int(selected_layer),
        "peak": float(peak),
        "second_layer": int(second_layer),
        "second_peak": float(second),
        "scale_bound": scale_bound,
        "peak_prominence": (float(peak) - float(second)) / max(abs(float(second)), scale_bound),
    }


def local_prominence_statistic(
    layer_scores: Mapping[int | str, float],
    *,
    eligible_layers: Sequence[int],
    score_dtype: str = "float32",
) -> dict[str, Any]:
    """Measure the global peak relative to its immediate eligible neighbors."""
    eligible = sorted({int(layer) for layer in eligible_layers})
    if len(eligible) < 2:
        raise ValueError("At least two eligible layers are required")
    missing = [layer for layer in eligible if str(layer) not in layer_scores and layer not in layer_scores]
    if missing:
        raise ValueError(f"Missing scores for eligible layers: {missing[:8]}")
    values = [
        float(layer_scores[str(layer)] if str(layer) in layer_scores else layer_scores[layer]) for layer in eligible
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Layer scores must be finite")
    peak = max(values)
    selected_index = next(index for index, value in enumerate(values) if value == peak)
    neighbor_indices = [index for index in (selected_index - 1, selected_index + 1) if 0 <= index < len(eligible)]
    neighbor_index = max(
        neighbor_indices,
        key=lambda index: (values[index], -eligible[index]),
    )
    neighbor_peak = values[neighbor_index]
    scale_bound = dtype_scale_bound(values, dtype=score_dtype)
    return {
        "candidate_version": LOCAL_PROMINENCE_VERSION,
        "selected_layer": int(eligible[selected_index]),
        "peak": float(peak),
        "neighbor_layer": int(eligible[neighbor_index]),
        "neighbor_peak": float(neighbor_peak),
        "scale_bound": scale_bound,
        "local_prominence": (float(peak) - float(neighbor_peak)) / max(abs(float(neighbor_peak)), scale_bound),
    }


def threshold_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff: float,
    statistic: str = "z_peak",
) -> dict[str, Any]:
    """Evaluate a strict global cutoff without model-specific routing."""
    tp = fp = tn = fn = 0
    for record in records:
        truth = str(record["label"]) == "rome"
        predicted = float(record[statistic]) > float(cutoff)
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    return {
        "cutoff": float(cutoff),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0 if math.isfinite(sensitivity) and math.isfinite(specificity) else math.nan
        ),
    }


def calibrate_equal_family_cutoff(
    records: Sequence[Mapping[str, Any]],
    *,
    statistic: str = "z_peak",
) -> dict[str, Any]:
    """Choose one cutoff by equal-family development balanced accuracy."""
    if not records:
        raise ValueError("Calibration records are required")
    values = sorted({float(record[statistic]) for record in records})
    candidates = [np.nextafter(values[0], -math.inf)]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:], strict=False))
    candidates.append(np.nextafter(values[-1], math.inf))
    families = sorted({str(record["family"]) for record in records})
    evaluated: list[tuple[tuple[float, float, float], float, list[dict[str, Any]]]] = []
    for cutoff in candidates:
        per_family = [
            {
                "family": family,
                **threshold_metrics(
                    [record for record in records if str(record["family"]) == family],
                    cutoff=float(cutoff),
                    statistic=statistic,
                ),
            }
            for family in families
        ]
        valid = [item["balanced_accuracy"] for item in per_family if math.isfinite(float(item["balanced_accuracy"]))]
        macro = float(np.mean(valid)) if valid else -math.inf
        worst = float(np.min(valid)) if valid else -math.inf
        pooled = threshold_metrics(records, cutoff=float(cutoff), statistic=statistic)
        evaluated.append(
            (
                (macro, worst, float(pooled["specificity"])),
                float(cutoff),
                per_family,
            )
        )
    objective, cutoff, per_family = max(
        evaluated,
        key=lambda item: (item[0], item[1]),
    )
    return {
        "cutoff": cutoff,
        "equal_family_balanced_accuracy": objective[0],
        "worst_family_balanced_accuracy": objective[1],
        "pooled": threshold_metrics(records, cutoff=cutoff, statistic=statistic),
        "per_family": per_family,
    }


__all__ = [
    "LOCAL_PROMINENCE_VERSION",
    "PEAK_PROMINENCE_VERSION",
    "ROBUST_PEAK_VERSION",
    "calibrate_equal_family_cutoff",
    "dtype_scale_bound",
    "local_prominence_statistic",
    "peak_prominence_statistic",
    "robust_peak_statistic",
    "threshold_metrics",
]

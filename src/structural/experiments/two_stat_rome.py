"""Transparent two-statistic rules for the final suspect-only ablation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.structural.experiments.single_checkpoint_rome import (
    selected_signed_consistency,
)


TWO_STAT_VERSION = "rome-single-checkpoint-two-stat-rectangle-v1"
SECONDARY_STATISTICS = (
    "peak",
    "robust_z",
    "global_prominence",
    "local_prominence",
)


def checkpoint_statistics(capture: dict[str, Any]) -> dict[str, Any]:
    """Return the signed value and four transparent M3 profile summaries."""
    signed = selected_signed_consistency(capture)
    eligible = sorted(int(layer) for layer in capture["eligible_layers"])
    profiles = capture["profiles"]
    values = np.asarray(
        [float(profiles[str(layer)]["relative_subspace_frobenius"]) for layer in eligible],
        dtype=np.float64,
    )
    if not bool(np.isfinite(values).all()):
        raise ValueError("M3 scores must be finite")
    order = sorted(
        range(len(eligible)),
        key=lambda index: (-float(values[index]), eligible[index]),
    )
    peak_index = order[0]
    peak = float(values[peak_index])
    second = float(values[order[1]])
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    bound = float(np.finfo(np.float32).eps) * max(1, len(values)) * max(1.0, float(np.max(np.abs(values))))
    neighbors = [float(values[index]) for index in (peak_index - 1, peak_index + 1) if 0 <= index < len(values)]
    return {
        **signed,
        "peak": peak,
        "robust_z": (peak - center) / max(mad, bound),
        "global_prominence": (peak - second) / max(abs(second), bound),
        "local_prominence": (peak - max(neighbors)) / max(abs(max(neighbors)), bound),
    }


def _cutoffs(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    middle = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(
        (
            [np.nextafter(unique[0], -math.inf)],
            middle,
            [np.nextafter(unique[-1], math.inf)],
        )
    )


def _conditions(
    values: np.ndarray,
    cutoffs: np.ndarray,
    direction: str,
) -> np.ndarray:
    if direction == "below":
        return values[None, :] < cutoffs[:, None]
    if direction == "above":
        return values[None, :] > cutoffs[:, None]
    raise ValueError(f"Unsupported direction: {direction}")


def calibrate_two_stat_rule(
    records: list[dict[str, Any]],
    *,
    secondary: str,
) -> dict[str, Any]:
    """Fit one global two-threshold conjunction with equal-family weighting."""
    if secondary not in SECONDARY_STATISTICS:
        raise ValueError(f"Unsupported secondary statistic: {secondary}")
    signed = np.asarray(
        [float(record["signed_residual_consistency"]) for record in records],
        dtype=np.float64,
    )
    other = np.asarray(
        [float(record[secondary]) for record in records],
        dtype=np.float64,
    )
    positive = np.asarray(
        [record["label"] == "rome" for record in records],
        dtype=bool,
    )
    families = np.asarray([str(record["family"]) for record in records])
    signed_cutoffs = _cutoffs(signed)
    other_cutoffs = _cutoffs(other)
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for signed_direction in ("below", "above"):
        signed_conditions = _conditions(
            signed,
            signed_cutoffs,
            signed_direction,
        ).astype(np.float32)
        for other_direction in ("below", "above"):
            other_conditions = _conditions(
                other,
                other_cutoffs,
                other_direction,
            ).astype(np.float32)
            family_balanced: list[np.ndarray] = []
            for family in sorted(set(families)):
                family_mask = families == family
                positive_mask = family_mask & positive
                negative_mask = family_mask & ~positive
                if not bool(positive_mask.any()) or not bool(negative_mask.any()):
                    continue
                true_positive = signed_conditions[:, positive_mask] @ other_conditions[:, positive_mask].T
                false_positive = signed_conditions[:, negative_mask] @ other_conditions[:, negative_mask].T
                sensitivity = true_positive / int(positive_mask.sum())
                specificity = 1.0 - false_positive / int(negative_mask.sum())
                family_balanced.append((sensitivity + specificity) / 2.0)
            macro = np.mean(family_balanced, axis=0)
            worst = np.min(family_balanced, axis=0)
            negative_mask = ~positive
            pooled_false_positive = signed_conditions[:, negative_mask] @ other_conditions[:, negative_mask].T
            pooled_specificity = 1.0 - pooled_false_positive / int(negative_mask.sum())
            flat_index = int(
                np.lexsort(
                    (
                        pooled_specificity.ravel(),
                        worst.ravel(),
                        macro.ravel(),
                    )
                )[-1]
            )
            signed_index, other_index = np.unravel_index(
                flat_index,
                macro.shape,
            )
            objective = (
                float(macro[signed_index, other_index]),
                float(worst[signed_index, other_index]),
                float(pooled_specificity[signed_index, other_index]),
            )
            candidate = {
                "candidate_version": TWO_STAT_VERSION,
                "secondary": secondary,
                "signed_direction": signed_direction,
                "signed_cutoff": float(signed_cutoffs[signed_index]),
                "secondary_direction": other_direction,
                "secondary_cutoff": float(other_cutoffs[other_index]),
                "signed_scale": max(
                    float(np.ptp(signed)),
                    float(np.finfo(np.float64).eps),
                ),
                "secondary_scale": max(
                    float(np.ptp(other)),
                    float(np.finfo(np.float64).eps),
                ),
                "equal_family_balanced_accuracy": objective[0],
                "worst_family_balanced_accuracy": objective[1],
            }
            if best is None or objective > best[0]:
                best = (objective, candidate)
    if best is None:
        raise ValueError("Calibration requires positives and negatives per family")
    return best[1]


def predict_two_stat(record: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Apply the calibrated two-threshold conjunction."""
    signed = float(record["signed_residual_consistency"])
    secondary = float(record[str(rule["secondary"])])
    signed_passes = (
        signed < float(rule["signed_cutoff"])
        if rule["signed_direction"] == "below"
        else signed > float(rule["signed_cutoff"])
    )
    secondary_passes = (
        secondary < float(rule["secondary_cutoff"])
        if rule["secondary_direction"] == "below"
        else secondary > float(rule["secondary_cutoff"])
    )
    return signed_passes and secondary_passes


def two_stat_margin(record: dict[str, Any], rule: dict[str, Any]) -> float:
    """Return a continuous margin whose sign matches the conjunction."""
    signed = float(record["signed_residual_consistency"])
    secondary = float(record[str(rule["secondary"])])
    signed_margin = (
        float(rule["signed_cutoff"]) - signed
        if rule["signed_direction"] == "below"
        else signed - float(rule["signed_cutoff"])
    ) / float(rule["signed_scale"])
    secondary_margin = (
        float(rule["secondary_cutoff"]) - secondary
        if rule["secondary_direction"] == "below"
        else secondary - float(rule["secondary_cutoff"])
    ) / float(rule["secondary_scale"])
    return min(signed_margin, secondary_margin)


__all__ = [
    "SECONDARY_STATISTICS",
    "TWO_STAT_VERSION",
    "calibrate_two_stat_rule",
    "checkpoint_statistics",
    "predict_two_stat",
    "two_stat_margin",
]

"""Parameter-free weighted-spectrum layer detection for localized rank-one edits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PROFILE_FIELDS = (
    "operator_norm",
    "frobenius_norm",
    "rank1_energy",
    "rank2_energy",
    "neighbor_cka_distance",
    "directional_background",
    "relative_operator_norm",
    "signed_relative_shift",
    "relative_subspace_operator_norm",
    "relative_subspace_frobenius",
    "relative_subspace_rank1_energy",
    "bilateral_coherence",
    "bilateral_alignment",
    "bilateral_frobenius",
    "bilateral_balance",
)
SCORE_FIELD = "relative_subspace_frobenius"


def detect_from_profiles(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    trim_first: int,
    trim_last: int,
) -> dict[str, Any]:
    """Select the largest affine-relative perturbation in the ROME subspace."""
    layers = sorted(int(layer) for layer in profiles)
    if not layers:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "layer_scores": {},
            "profiles": {},
        }
    missing = [
        str(layer)
        for layer in layers
        if str(layer) not in profiles
        or any(field not in profiles[str(layer)] for field in PROFILE_FIELDS)
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"Weighted-spectrum profiles are incomplete on layers {preview}{suffix}")

    non_finite = [
        f"{layer}:{field}"
        for layer in layers
        for field in PROFILE_FIELDS
        if not np.isfinite(float(profiles[str(layer)][field]))
    ]
    if non_finite:
        preview = ", ".join(non_finite[:8])
        suffix = "..." if len(non_finite) > 8 else ""
        raise ValueError(f"Weighted-spectrum profiles contain non-finite values at {preview}{suffix}")

    start = min(max(0, int(trim_first)), len(layers))
    end = len(layers) - min(max(0, int(trim_last)), len(layers) - start)
    if end <= start:
        start, end = 0, len(layers)
    candidates = np.arange(start, end)
    score = np.asarray([float(profiles[str(layer)][SCORE_FIELD]) for layer in layers], dtype=np.float64)
    best_index = int(candidates[int(np.argmax(score[candidates]))])
    ordered = np.sort(score[candidates])
    margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])

    return {
        "anomalous_layer": int(layers[best_index]),
        "detection_score": float(score[best_index]),
        "margin": margin,
        "score_field": SCORE_FIELD,
        "layer_scores": {str(layer): float(score[index]) for index, layer in enumerate(layers)},
        "profiles": {
            str(layer): {field: float(profiles[str(layer)][field]) for field in PROFILE_FIELDS}
            for layer in layers
        },
        "config": {
            "trim_first": int(start),
            "trim_last": int(len(layers) - end),
        },
        "evaluated_layers": [int(layer) for layer in layers[start:end]],
        "excluded_layers": [int(layer) for layer in layers[:start] + layers[end:]],
    }


__all__ = ["PROFILE_FIELDS", "SCORE_FIELD", "detect_from_profiles"]

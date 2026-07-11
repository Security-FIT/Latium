"""Architecture-neutral, training-free decisions for ROME-like edit presence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from src.structural.detectors.weighted_spectrum import (
    FOOTPRINT_PROFILE_FIELDS,
    LOCALIZER_PROFILE_FIELDS,
    detect_from_profiles,
)


BlindStrategy = Literal["peak", "footprint"]
_MAD_NORMAL_SCALE = 1.482602218505602
_EPS = np.finfo(np.float64).eps


def _universal_outlier(values: np.ndarray) -> dict[str, float | bool]:
    """Test the upper extreme against the universal Gaussian-noise bound."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Universal outlier test requires a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("Universal outlier test received non-finite values")
    center = float(np.median(values))
    scale = float(_MAD_NORMAL_SCALE * np.median(np.abs(values - center)))
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    effective_scale = max(scale, _EPS * max(1.0, abs(center), abs(peak)))
    robust_z = max(0.0, (peak - center) / effective_scale)
    threshold = math.sqrt(2.0 * math.log(max(2, int(values.size))))
    tail_probability = 1.0 - NormalDist().cdf(float(robust_z))
    return {
        "is_outlier": bool(robust_z > threshold),
        "peak_index": peak_index,
        "peak": peak,
        "median": center,
        "mad_scale": scale,
        "effective_mad_scale": effective_scale,
        "robust_z": robust_z,
        "universal_threshold": threshold,
        "gaussian_tail_probability": tail_probability,
        "evidence_ratio": robust_z / threshold,
    }


def _presence_series(
    profiles: Mapping[str, Mapping[str, float]],
    layers: list[int],
    strategy: BlindStrategy,
) -> np.ndarray:
    required_fields = (
        LOCALIZER_PROFILE_FIELDS if strategy == "peak" else FOOTPRINT_PROFILE_FIELDS
    )
    missing = [
        f"{layer}:{field}"
        for layer in layers
        for field in required_fields
        if field not in profiles.get(str(layer), {})
    ]
    if missing:
        raise ValueError(
            "ROME-presence profiles are incomplete at " + ", ".join(missing[:8])
        )
    spectral = np.asarray(
        [float(profiles[str(layer)]["relative_subspace_frobenius"]) for layer in layers],
        dtype=np.float64,
    )
    if strategy == "peak":
        return np.log1p(np.maximum(spectral, 0.0))
    if strategy != "footprint":
        raise ValueError(f"Unknown blind ROME-presence strategy: {strategy}")
    # An isolated ROME update produces a balanced same-sign curvature peak and
    # an at-most-rank-two Gram residual.  Multiplication is a conjunction of
    # those dimensionless pieces, with no fitted feature weights.
    coherence = np.asarray(
        [float(profiles[str(layer)]["bilateral_coherence"]) for layer in layers],
        dtype=np.float64,
    )
    balance = np.asarray(
        [float(profiles[str(layer)]["bilateral_balance"]) for layer in layers],
        dtype=np.float64,
    )
    rank2 = np.asarray(
        [float(profiles[str(layer)]["rank2_energy"]) for layer in layers],
        dtype=np.float64,
    )
    morphology = (
        np.clip(coherence, 0.0, 1.0)
        * np.clip(balance, 0.0, 1.0)
        * np.clip(rank2, 0.0, 1.0)
    )
    return np.log1p(np.maximum(spectral, 0.0) * morphology)


def detect_rome_presence_blind(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    trim_first: int,
    trim_last: int,
    strategy: BlindStrategy,
) -> dict[str, Any]:
    """Make a suspect-only ROME-like presence decision from depth profiles."""
    localized = detect_from_profiles(
        profiles,
        trim_first=trim_first,
        trim_last=trim_last,
    )
    layers = list(localized.get("evaluated_layers", ()))
    if not layers:
        return {
            "is_rome_like": False,
            "is_edited": False,
            "verdict": "insufficient_layers",
            "strategy": strategy,
            "anomalous_layer": None,
        }
    series = _presence_series(profiles, layers, strategy)
    outlier = _universal_outlier(series)
    peak_index = int(outlier["peak_index"])
    peak_layer = int(layers[peak_index])
    is_rome_like = bool(outlier["is_outlier"])
    return {
        "is_rome_like": is_rome_like,
        "is_edited": is_rome_like,
        "verdict": "rome_like" if is_rome_like else "no_universal_outlier",
        "strategy": strategy,
        "threat_model": "suspect_only",
        "calibration": "universal_bound_not_empirically_calibrated",
        "anomalous_layer": peak_layer,
        "detection_score": float(outlier["evidence_ratio"]),
        "evidence": outlier,
        "layer_evidence": {str(layer): float(series[index]) for index, layer in enumerate(layers)},
        "localizer": localized,
        "required_profile_fields": list(
            LOCALIZER_PROFILE_FIELDS if strategy == "peak" else FOOTPRINT_PROFILE_FIELDS
        ),
    }


def detect_rome_presence_delta(
    families: Mapping[str, Mapping[str, Mapping[str, float | bool]]],
) -> dict[str, Any]:
    """Accept one numerically rank-one MLP-output checkpoint delta."""
    projection = dict(families.get("proj", {}))
    fc = dict(families.get("fc", {}))
    all_changes = [("proj", layer, profile) for layer, profile in projection.items()]
    all_changes.extend(("fc", layer, profile) for layer, profile in fc.items())
    if not all_changes:
        return {
            "is_rome_like": False,
            "is_edited": False,
            "verdict": "no_detectable_change",
            "threat_model": "clean_baseline",
            "anomalous_layer": None,
            "detection_score": 0.0,
            "families": {"proj": projection, "fc": fc},
        }
    if len(all_changes) != 1 or len(projection) != 1:
        return {
            "is_rome_like": False,
            "is_edited": True,
            "verdict": "change_not_confined_to_one_mlp_output",
            "threat_model": "clean_baseline",
            "anomalous_layer": None,
            "detection_score": 0.0,
            "changed_matrices": [f"{family}:{layer}" for family, layer, _ in all_changes],
            "families": {"proj": projection, "fc": fc},
        }
    layer, profile = next(iter(projection.items()))
    rank_one = bool(profile.get("rank_one_within_roundoff", False))
    residual = float(profile.get("rank1_residual", math.inf))
    roundoff = float(profile.get("roundoff_bound", 0.0))
    evidence_ratio = float(roundoff / max(residual, float(_EPS))) if roundoff > 0.0 else 0.0
    return {
        "is_rome_like": rank_one,
        "is_edited": True,
        "verdict": "rome_like" if rank_one else "localized_update_not_rank_one",
        "threat_model": "clean_baseline",
        "attribution_scope": "rome_family_single_rank_edit",
        "anomalous_layer": int(layer),
        "detection_score": evidence_ratio,
        "evidence": {
            "rank_one_within_roundoff": rank_one,
            "rank1_residual": residual,
            "roundoff_bound": roundoff,
            "roundoff_to_residual_ratio": evidence_ratio,
        },
        "families": {"proj": projection, "fc": fc},
    }


__all__ = [
    "BlindStrategy",
    "detect_rome_presence_blind",
    "detect_rome_presence_delta",
]

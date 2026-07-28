"""Minimal architecture-neutral layer localization for ROME-compatible edits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "rome-detector-minimal-v1"
SCORE_FIELD = "relative_subspace_frobenius"
PROFILE_FIELDS = (SCORE_FIELD,)
DEFAULT_TRIM_FRACTION = 0.10


def numerical_tolerance(dtype: torch.dtype, dimension: int, scale: float) -> float:
    """Return a dimension- and scale-aware floating-point roundoff bound."""
    info = torch.finfo(dtype if dtype.is_floating_point else torch.float32)
    safe_scale = max(abs(float(scale)), float(info.tiny))
    return float(info.eps) * max(1, int(dimension)) * safe_scale


def hidden_gram(weight: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    """Orient a rectangular editable matrix into its smaller hidden space."""
    if weight.ndim != 2:
        raise ValueError(f"Editable projection must be a matrix, got shape {tuple(weight.shape)}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Editable projection contains non-finite values")
    device = weight.device if weight.is_cuda else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    matrix = weight.detach().to(device=device, dtype=compute_dtype)
    raw = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    if not normalize:
        return raw
    scale = float(matrix.square().sum().item())
    tolerance = numerical_tolerance(compute_dtype, max(matrix.shape), scale)
    if not math.isfinite(scale) or scale <= tolerance:
        raise ValueError("Editable projection must contain finite, non-zero values")
    return raw / scale


def eligible_layers(
    layers: list[int],
    *,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> list[int]:
    """Return deterministic interior eligibility using one fractional trim."""
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if len(layers) < 3:
        return []
    trim = int(math.floor(len(layers) * float(trim_fraction)))
    start = max(1, trim)
    stop = min(len(layers) - 1, len(layers) - trim)
    return [int(layer) for layer in layers[start:stop]]


def localize_scores(
    layer_scores: Mapping[str, float],
    *,
    layers: list[int],
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    clean_reference_presence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the lowest layer on exact score ties."""
    eligible = eligible_layers(layers, trim_fraction=trim_fraction)
    missing = [str(layer) for layer in eligible if str(layer) not in layer_scores]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"ROME localizer scores are incomplete on layers {preview}{suffix}")
    non_finite = [
        str(layer)
        for layer in eligible
        if not np.isfinite(float(layer_scores[str(layer)]))
    ]
    if non_finite:
        raise ValueError(
            "ROME localizer scores contain non-finite values at "
            + ", ".join(non_finite[:8])
        )
    ordered = sorted(
        eligible,
        key=lambda layer: (-float(layer_scores[str(layer)]), int(layer)),
    )
    selected = ordered[0] if ordered else None
    selected_score = float(layer_scores[str(selected)]) if selected is not None else 0.0
    second_score = float(layer_scores[str(ordered[1])]) if len(ordered) > 1 else 0.0
    excluded = [int(layer) for layer in layers if layer not in set(eligible)]
    return {
        "schema_version": SCHEMA_VERSION,
        "localization": {
            "eligible_layers": eligible,
            "excluded_layers": excluded,
            "layer_scores": {
                str(layer): float(layer_scores[str(layer)])
                for layer in sorted(int(value) for value in layer_scores)
            },
            "selected_layer": (int(selected) if selected is not None else None),
            "margin": selected_score - second_score,
        },
        "clean_reference_presence": dict(
            clean_reference_presence
            or {
                "available": False,
                "is_rome_compatible": None,
                "verdict": "clean_reference_unavailable",
                "selected_layer": None,
            }
        ),
    }


def detect_from_profiles(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    layers: list[int] | None = None,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    clean_reference_presence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Localize from the one-field capture retained by the minimal detector."""
    resolved_layers = (
        [int(layer) for layer in layers]
        if layers is not None
        else sorted(int(layer) for layer in profiles)
    )
    layer_scores: dict[str, float] = {}
    for raw_layer, profile in profiles.items():
        if SCORE_FIELD not in profile:
            raise ValueError(f"ROME localizer profile for layer {raw_layer} has no {SCORE_FIELD}")
        layer_scores[str(int(raw_layer))] = float(profile[SCORE_FIELD])
    return localize_scores(
        layer_scores,
        layers=resolved_layers,
        trim_fraction=trim_fraction,
        clean_reference_presence=clean_reference_presence,
    )


__all__ = [
    "DEFAULT_TRIM_FRACTION",
    "PROFILE_FIELDS",
    "SCHEMA_VERSION",
    "SCORE_FIELD",
    "detect_from_profiles",
    "eligible_layers",
    "hidden_gram",
    "localize_scores",
    "numerical_tolerance",
]

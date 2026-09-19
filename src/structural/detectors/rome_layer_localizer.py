"""Architecture-neutral layer localization for ROME-compatible edits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from src.common.linalg import gpu_svd_topk


SCORE_FIELD = "diagonal_relative"
PROFILE_FIELDS = (SCORE_FIELD,)
DEFAULT_TRIM_FRACTION = 0.10
_SVD_SEED = 433494437


def numerical_tolerance(dtype: torch.dtype, dimension: int, scale: float) -> float:
    """Return a dimension- and scale-aware floating-point roundoff bound."""
    info = torch.finfo(dtype if dtype.is_floating_point else torch.float32)
    safe_scale = max(abs(float(scale)), float(info.tiny))
    return float(info.eps) * max(1, int(dimension)) * safe_scale


def hidden_gram(weight: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Orient an editable matrix into its smaller hidden space."""
    if weight.ndim != 2:
        raise ValueError(f"Editable projection must be a matrix, got shape {tuple(weight.shape)}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Editable projection contains non-finite values")

    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    matrix = weight.detach().to(dtype=compute_dtype)
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
    if len(set(layers)) != len(layers):
        raise ValueError("layers must be unique")

    ordered = sorted(int(layer) for layer in layers)
    trim = int(math.floor(len(ordered) * float(trim_fraction)))
    start = max(1, trim)
    stop = min(len(ordered) - 1, len(ordered) - trim)
    return ordered[start:stop]


def score_layer(current: torch.Tensor, reference: torch.Tensor, *, layer: int) -> float:
    """Score one layer against the mean normalized Gram of its neighbors."""
    if current.shape != reference.shape:
        raise ValueError(
            f"Layer {layer} Gram shape {tuple(current.shape)} does not match "
            f"neighbor shape {tuple(reference.shape)}"
        )

    residual = current - reference
    cuda_devices = [current.device.index or 0] if current.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(_SVD_SEED + int(layer))
        if current.is_cuda:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(_SVD_SEED + int(layer))
        left, singular_values, _right = gpu_svd_topk(
            residual,
            k=min(2, min(residual.shape)),
            niter=4,
            device=str(residual.device),
        )

    basis = left.to(device=reference.device, dtype=reference.dtype)
    singular_values = singular_values.to(device=reference.device, dtype=reference.dtype)
    reference_subspace = basis.T @ reference @ basis
    support_scale = torch.linalg.matrix_norm(reference_subspace, ord="fro")
    residual_scale = torch.linalg.matrix_norm(residual, ord="fro")
    tolerance = numerical_tolerance(
        residual.dtype,
        int(residual.shape[0]),
        max(float(support_scale.item()), float(residual_scale.item())),
    )
    directional_support = torch.diagonal(reference_subspace).clamp_min(tolerance)
    return float(torch.linalg.vector_norm(singular_values / directional_support).item())


def profile_weights(
    weights: Mapping[int, torch.Tensor],
    *,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> dict[str, Any]:
    """Build the minimal one-field profile while retaining three Grams at most."""
    layers = sorted(int(layer) for layer in weights)
    eligible = eligible_layers(layers, trim_fraction=trim_fraction)
    positions = {layer: index for index, layer in enumerate(layers)}
    densities: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, float]] = {}

    for layer in eligible:
        index = positions[layer]
        neighborhood = layers[index - 1 : index + 2]
        for other in neighborhood:
            if other not in densities:
                densities[other] = hidden_gram(weights[other])
        reference = (densities[neighborhood[0]] + densities[neighborhood[2]]) * 0.5
        profiles[str(layer)] = {SCORE_FIELD: score_layer(densities[layer], reference, layer=layer)}
        densities = {
            cached_layer: density
            for cached_layer, density in densities.items()
            if positions[cached_layer] >= index
        }

    eligible_set = set(eligible)
    return {
        "mode": "single_checkpoint",
        "layers": layers,
        "trim_fraction": float(trim_fraction),
        "eligible_layers": eligible,
        "excluded_layers": [layer for layer in layers if layer not in eligible_set],
        "profile_fields": list(PROFILE_FIELDS),
        "profiles": profiles,
    }


def localize_scores(
    layer_scores: Mapping[str, float],
    *,
    layers: list[int],
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> dict[str, Any]:
    """Select the highest score, preferring the lower layer on exact ties."""
    ordered_layers = sorted(int(layer) for layer in layers)
    eligible = eligible_layers(ordered_layers, trim_fraction=trim_fraction)
    if not eligible:
        raise ValueError("ROME layer localization requires at least three layers")

    missing = [str(layer) for layer in eligible if str(layer) not in layer_scores]
    if missing:
        raise ValueError(f"ROME localizer scores are incomplete on layers {', '.join(missing[:8])}")
    non_finite = [str(layer) for layer in eligible if not math.isfinite(float(layer_scores[str(layer)]))]
    if non_finite:
        raise ValueError(f"ROME localizer scores are non-finite on layers {', '.join(non_finite[:8])}")

    ranked = sorted(eligible, key=lambda layer: (-float(layer_scores[str(layer)]), layer))
    selected = ranked[0]
    selected_score = float(layer_scores[str(selected)])
    second_score = float(layer_scores[str(ranked[1])]) if len(ranked) > 1 else 0.0
    eligible_set = set(eligible)
    return {
        "localization": {
            "eligible_layers": eligible,
            "excluded_layers": [layer for layer in ordered_layers if layer not in eligible_set],
            "layer_scores": {
                str(layer): float(layer_scores[str(layer)])
                for layer in sorted(int(raw_layer) for raw_layer in layer_scores)
            },
            "selected_layer": selected,
            "margin": selected_score - second_score,
        },
    }


def detect_from_profiles(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    layers: list[int] | None = None,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> dict[str, Any]:
    """Localize from the one-field capture retained by the minimal detector."""
    if layers is None:
        raise ValueError("ROME localizer requires the full capture layer list; profile keys are already trimmed")
    resolved_layers = sorted(int(layer) for layer in layers)
    layer_scores: dict[str, float] = {}
    for raw_layer, profile in profiles.items():
        if SCORE_FIELD not in profile:
            raise ValueError(f"ROME localizer profile for layer {raw_layer} has no {SCORE_FIELD}")
        layer_scores[str(int(raw_layer))] = float(profile[SCORE_FIELD])
    return localize_scores(layer_scores, layers=resolved_layers, trim_fraction=trim_fraction)


class RomeLayerLocalizer:
    """Small direct API for localizing a ROME-style edit from projection weights."""

    def __init__(self, *, trim_fraction: float = DEFAULT_TRIM_FRACTION) -> None:
        self.trim_fraction = float(trim_fraction)

    def profile(self, weights: Mapping[int, torch.Tensor]) -> dict[str, Any]:
        return profile_weights(weights, trim_fraction=self.trim_fraction)

    def localize(self, weights: Mapping[int, torch.Tensor]) -> dict[str, Any]:
        profile = self.profile(weights)
        return detect_from_profiles(
            profile["profiles"],
            layers=profile["layers"],
            trim_fraction=profile["trim_fraction"],
        )


__all__ = [
    "DEFAULT_TRIM_FRACTION",
    "PROFILE_FIELDS",
    "RomeLayerLocalizer",
    "SCORE_FIELD",
    "detect_from_profiles",
    "eligible_layers",
    "hidden_gram",
    "localize_scores",
    "numerical_tolerance",
    "profile_weights",
    "score_layer",
]

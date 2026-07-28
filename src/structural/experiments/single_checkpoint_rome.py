"""Opt-in signed residual capture for suspect-only ROME research."""

from __future__ import annotations

import math
from typing import Any

import torch

from src.common.io import to_serializable
from src.structural.capture.producers import (
    CaptureContext,
    _hidden_spectral_density,
    _weighted_spectrum_relative_subspace,
)
from src.structural.detectors.weighted_spectrum import (
    DEFAULT_TRIM_FRACTION,
    eligible_layers,
    numerical_tolerance,
)
SIGNED_CAPTURE_SCHEMA = "rome-single-checkpoint-signed-capture-v1"
SIGNED_CONSISTENCY_VERSION = "rome-single-checkpoint-signed-consistency-v1"


def signed_residual_profile(relative_subspace: torch.Tensor) -> dict[str, float]:
    """Summarize the signed 2x2 residual without adding another decomposition."""
    if relative_subspace.shape != (2, 2):
        raise ValueError("The support-whitened residual must be 2x2")
    if not bool(torch.isfinite(relative_subspace).all()):
        raise ValueError("The support-whitened residual must be finite")
    score = float(torch.linalg.matrix_norm(relative_subspace, ord="fro").item())
    bound = numerical_tolerance(
        relative_subspace.dtype,
        relative_subspace.numel(),
        score,
    )
    return {
        "relative_subspace_frobenius": score,
        "signed_residual_consistency": float(torch.trace(relative_subspace).item()) / max(score, bound),
        "consistency_bound": bound,
    }


def capture_single_checkpoint_signed(context: CaptureContext) -> dict[str, Any]:
    """Capture M3 plus one signed statistic behind an experimental opt-in."""
    layers = sorted(context.proj_weights)
    included = layers[1:-1]
    positions = {layer: index for index, layer in enumerate(layers)}
    densities: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, float]] = {}
    for layer in included:
        index = positions[layer]
        neighborhood = layers[max(0, index - 1) : min(len(layers), index + 2)]
        for other in neighborhood:
            if other not in densities:
                densities[other] = _hidden_spectral_density(context.proj_weights[other])
        neighbors = [densities[other] for other in neighborhood if other != layer]
        if not neighbors:
            continue
        reference = torch.stack(neighbors).mean(dim=0)
        relative = _weighted_spectrum_relative_subspace(
            densities[layer],
            reference,
            layer=layer,
        )
        profiles[str(layer)] = signed_residual_profile(relative)
        densities = {
            cached_layer: density for cached_layer, density in densities.items() if positions[cached_layer] >= index
        }

    eligible = eligible_layers(layers, trim_fraction=DEFAULT_TRIM_FRACTION)
    return to_serializable(
        {
            "schema_version": SIGNED_CAPTURE_SCHEMA,
            "candidate_version": SIGNED_CONSISTENCY_VERSION,
            "mode": "single_checkpoint",
            "layers": layers,
            "trim_fraction": DEFAULT_TRIM_FRACTION,
            "eligible_layers": eligible,
            "excluded_layers": [layer for layer in layers if layer not in set(eligible)],
            "profile_fields": [
                "relative_subspace_frobenius",
                "signed_residual_consistency",
                "consistency_bound",
            ],
            "profiles": profiles,
        }
    )


def selected_signed_consistency(capture: dict[str, Any]) -> dict[str, Any]:
    """Return the signed statistic at the deterministic M3 maximum."""
    eligible = sorted(int(layer) for layer in capture["eligible_layers"])
    profiles = capture["profiles"]
    selected = min(
        eligible,
        key=lambda layer: (
            -float(profiles[str(layer)]["relative_subspace_frobenius"]),
            layer,
        ),
    )
    value = float(profiles[str(selected)]["signed_residual_consistency"])
    if not math.isfinite(value):
        raise ValueError("Signed residual consistency must be finite")
    return {
        "candidate_version": SIGNED_CONSISTENCY_VERSION,
        "selected_layer": selected,
        "signed_residual_consistency": value,
    }


__all__ = [
    "SIGNED_CAPTURE_SCHEMA",
    "SIGNED_CONSISTENCY_VERSION",
    "capture_single_checkpoint_signed",
    "selected_signed_consistency",
    "signed_residual_profile",
]

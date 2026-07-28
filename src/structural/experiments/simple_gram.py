"""Minimal no-reference Gram scores for ROME layer-localization experiments."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from src.common.io import to_serializable
from src.common.linalg import gpu_svd_topk
from src.structural.capture.producers import CaptureContext
from src.structural.detectors.weighted_spectrum import (
    DEFAULT_TRIM_FRACTION,
    eligible_layers,
    hidden_gram,
    numerical_tolerance,
)


SCHEMA_VERSION = "rome-simple-gram-experiment-v1"
GRAM_FROBENIUS = "gram_frobenius"
GRAM_RELATIVE = "gram_relative"
TOP2_FROBENIUS = "top2_frobenius"
SCALAR_RELATIVE = "scalar_relative"
DIAGONAL_RELATIVE = "diagonal_relative"
M3_CONTROL = "m3_control"
PROFILE_FIELDS = (
    GRAM_FROBENIUS,
    GRAM_RELATIVE,
    TOP2_FROBENIUS,
    SCALAR_RELATIVE,
    DIAGONAL_RELATIVE,
    M3_CONTROL,
)


def _deterministic_top_two(
    residual: torch.Tensor,
    *,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a reproducible rank-two residual basis and its singular values."""
    rank = min(int(residual.shape[0]), int(residual.shape[1]))
    if rank < 2:
        left, singular_values, _right = torch.linalg.svd(
            residual,
            full_matrices=False,
        )
        return left[:, :rank], singular_values[:rank]

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed = 433494437 + int(layer)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        left, singular_values, _right = gpu_svd_topk(
            residual,
            k=2,
            niter=4,
        )
    return left[:, :2], singular_values[:2]


def simple_gram_profile(
    current: torch.Tensor,
    neighbor: torch.Tensor,
    *,
    layer: int,
) -> dict[str, float]:
    """Calculate the G0-G3 complexity ladder from two normalized Grams.

    G0 uses only the Gram residual Frobenius norm. G0R divides that norm by
    the neighboring Gram's Frobenius norm and still uses no SVD. G1 adds the
    leading two residual singular values. G2 divides the projected residual
    by one scalar support norm. G3 independently rescales the two projected
    directions; unlike M3, it does not rotate the support or calculate a
    matrix inverse square root. M3 is retained only as a paired accuracy
    control and shares the same Gram matrices and rank-two basis.
    """
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("Current hidden Gram must be square")
    if current.shape != neighbor.shape:
        raise ValueError("Current and neighbor hidden Grams must have equal shapes")
    if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(neighbor).all()):
        raise ValueError("Hidden Grams must contain only finite values")

    residual = current - neighbor
    gram_frobenius = torch.linalg.matrix_norm(residual, ord="fro")
    neighbor_frobenius = torch.linalg.matrix_norm(neighbor, ord="fro")
    full_tolerance = numerical_tolerance(
        residual.dtype,
        int(residual.shape[0]),
        max(
            float(neighbor_frobenius.item()),
            float(gram_frobenius.item()),
        ),
    )
    gram_relative = gram_frobenius / max(
        float(neighbor_frobenius.item()),
        full_tolerance,
    )
    basis, singular_values = _deterministic_top_two(residual, layer=layer)
    basis = basis.to(device=residual.device, dtype=residual.dtype)
    singular_values = singular_values.to(device=residual.device, dtype=residual.dtype)
    top2_frobenius = torch.linalg.vector_norm(singular_values)

    if basis.shape[1] == 0:
        return {
            GRAM_FROBENIUS: float(gram_frobenius.item()),
            GRAM_RELATIVE: float(gram_relative.item()),
            TOP2_FROBENIUS: 0.0,
            SCALAR_RELATIVE: 0.0,
            DIAGONAL_RELATIVE: 0.0,
            M3_CONTROL: 0.0,
        }

    projected_residual = basis.T @ residual @ basis
    projected_support = basis.T @ neighbor @ basis
    residual_scale = float(gram_frobenius.item())
    support_frobenius = torch.linalg.matrix_norm(projected_support, ord="fro")
    tolerance = numerical_tolerance(
        residual.dtype,
        int(residual.shape[0]),
        max(float(support_frobenius.item()), residual_scale),
    )
    scalar_relative = (
        top2_frobenius
        / max(float(support_frobenius.item()), tolerance)
    )

    diagonal = torch.diagonal(projected_support).clamp_min(tolerance)
    diagonal_relative = torch.linalg.vector_norm(singular_values / diagonal)
    support_eigenvalues, support_eigenvectors = torch.linalg.eigh(
        projected_support
    )
    inverse_sqrt = (
        support_eigenvectors
        @ torch.diag(support_eigenvalues.clamp_min(1e-10).rsqrt())
        @ support_eigenvectors.T
    )
    m3_control = torch.linalg.matrix_norm(
        inverse_sqrt @ projected_residual @ inverse_sqrt,
        ord="fro",
    )
    values = {
        GRAM_FROBENIUS: float(gram_frobenius.item()),
        GRAM_RELATIVE: float(gram_relative.item()),
        TOP2_FROBENIUS: float(top2_frobenius.item()),
        SCALAR_RELATIVE: float(scalar_relative.item()),
        DIAGONAL_RELATIVE: float(diagonal_relative.item()),
        M3_CONTROL: float(m3_control.item()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Simple Gram profile produced a non-finite score")
    return values


def select_layer(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    eligible: list[int],
    field: str,
) -> dict[str, Any]:
    """Select the lowest layer on an exact score tie."""
    if field not in PROFILE_FIELDS:
        raise ValueError(f"Unknown simple Gram field: {field}")
    missing = [layer for layer in eligible if field not in profiles.get(str(layer), {})]
    if missing:
        raise ValueError(f"Simple Gram profiles are incomplete at layers {missing[:8]}")
    ordered = sorted(
        eligible,
        key=lambda layer: (-float(profiles[str(layer)][field]), int(layer)),
    )
    selected = ordered[0] if ordered else None
    score = float(profiles[str(selected)][field]) if selected is not None else 0.0
    second = float(profiles[str(ordered[1])][field]) if len(ordered) > 1 else 0.0
    return {
        "field": field,
        "selected_layer": selected,
        "score": score,
        "margin": score - second,
    }


def spike_statistics(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    eligible: list[int],
    field: str,
) -> dict[str, float | int | str | None]:
    """Summarize one candidate's peak without making a binary claim."""
    selected = select_layer(profiles, eligible=eligible, field=field)
    selected_layer = selected["selected_layer"]
    if selected_layer is None:
        return {
            "field": field,
            "selected_layer": None,
            "robust_peak": 0.0,
            "global_prominence": 0.0,
            "local_prominence": 0.0,
        }

    values = np.asarray(
        [float(profiles[str(layer)][field]) for layer in eligible],
        dtype=np.float64,
    )
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    peak = float(profiles[str(selected_layer)][field])
    ordered = np.sort(values)
    second = float(ordered[-2]) if len(ordered) > 1 else 0.0
    tolerance = numerical_tolerance(
        torch.float64,
        len(eligible),
        max(abs(center), abs(peak)),
    )
    selected_index = eligible.index(int(selected_layer))
    local_layers = eligible[
        max(0, selected_index - 1) : selected_index
    ] + eligible[selected_index + 1 : selected_index + 2]
    local_values = [
        float(profiles[str(layer)][field]) for layer in local_layers
    ]
    local_center = (
        float(np.median(np.asarray(local_values, dtype=np.float64)))
        if local_values
        else center
    )
    return {
        "field": field,
        "selected_layer": int(selected_layer),
        "robust_peak": (peak - center) / max(mad, tolerance),
        "global_prominence": peak / max(abs(second), tolerance),
        "local_prominence": peak / max(abs(local_center), tolerance),
    }


def capture_simple_gram(context: CaptureContext) -> dict[str, Any]:
    """Capture G0-G3 from one checkpoint without any reference model."""
    started = time.perf_counter()
    layers = sorted(context.proj_weights)
    included = layers[1:-1]
    positions = {layer: index for index, layer in enumerate(layers)}
    cache: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, float]] = {}

    for layer in included:
        index = positions[layer]
        neighborhood = layers[index - 1 : index + 2]
        for other in neighborhood:
            if other not in cache:
                cache[other] = hidden_gram(
                    context.proj_weights[other],
                    normalize=True,
                )
        neighbor = (cache[neighborhood[0]] + cache[neighborhood[2]]) / 2.0
        profiles[str(layer)] = simple_gram_profile(
            cache[layer],
            neighbor,
            layer=layer,
        )
        cache = {
            cached_layer: gram
            for cached_layer, gram in cache.items()
            if positions[cached_layer] >= index
        }

    eligible = eligible_layers(layers, trim_fraction=DEFAULT_TRIM_FRACTION)
    return to_serializable(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "single_checkpoint",
            "layers": layers,
            "trim_fraction": DEFAULT_TRIM_FRACTION,
            "eligible_layers": eligible,
            "excluded_layers": [
                layer for layer in layers if layer not in set(eligible)
            ],
            "profile_fields": list(PROFILE_FIELDS),
            "profiles": profiles,
            "localization": {
                field: select_layer(profiles, eligible=eligible, field=field)
                for field in PROFILE_FIELDS
            },
            "spike_statistics": {
                field: spike_statistics(
                    profiles,
                    eligible=eligible,
                    field=field,
                )
                for field in PROFILE_FIELDS
            },
            "runtime_seconds": time.perf_counter() - started,
        }
    )


__all__ = [
    "DIAGONAL_RELATIVE",
    "GRAM_FROBENIUS",
    "GRAM_RELATIVE",
    "M3_CONTROL",
    "PROFILE_FIELDS",
    "SCALAR_RELATIVE",
    "SCHEMA_VERSION",
    "TOP2_FROBENIUS",
    "capture_simple_gram",
    "select_layer",
    "simple_gram_profile",
    "spike_statistics",
]

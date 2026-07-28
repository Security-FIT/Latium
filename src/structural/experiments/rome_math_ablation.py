"""Architecture-independent ROME detector mathematics for recapture ablations.

This module is intentionally experimental.  It computes M0--M3 from the same
editable projection weights while leaving the production weighted-spectrum
detector unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from src.common.io import to_serializable
from src.structural.capture.producers import (
    CaptureContext,
    _deterministic_topk_svd,
    _weighted_spectrum_profile,
)
from src.structural.detectors.rome_presence import _presence_series, _universal_outlier
from src.structural.detectors.weighted_spectrum import FOOTPRINT_PROFILE_FIELDS


CAPTURE_SCHEMA_VERSION = "rome-math-ablation-capture-v1"
EVALUATION_SCHEMA_VERSION = "rome-math-ablation-evaluation-v1"
DEFAULT_TRIM_FRACTION = 0.10
DEFAULT_SVD_MODE = "randomized"
CANDIDATE_FIELDS = {
    "M0": "m0",
    "M1": "m1",
    "M2": "m2",
    "M3": "m3",
    "M0_raw": "m0_raw",
    "M1_raw": "m1_raw",
}


@dataclass(frozen=True)
class GramRecord:
    raw: torch.Tensor
    normalized: torch.Tensor
    trace: float
    trace_tolerance: float
    source_dtype: str
    compute_dtype: str
    orientation: str
    weight_shape: tuple[int, int]


def numerical_tolerance(
    dtype: torch.dtype,
    dimension: int,
    scale: float,
) -> float:
    """Return a dimension- and scale-aware floating-point roundoff bound."""
    info = torch.finfo(dtype if dtype.is_floating_point else torch.float32)
    safe_scale = max(abs(float(scale)), float(info.tiny))
    return float(info.eps) * max(1, int(dimension)) * safe_scale


def hidden_gram(weight: torch.Tensor) -> GramRecord:
    """Orient a rectangular editable matrix into its smaller hidden space."""
    if weight.ndim != 2:
        raise ValueError(f"Editable projection must be a matrix, got shape {tuple(weight.shape)}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Editable projection contains non-finite values")
    device = weight.device if weight.is_cuda else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    matrix = weight.detach().to(device=device, dtype=compute_dtype)
    rows, columns = (int(value) for value in matrix.shape)
    if rows <= columns:
        raw = matrix @ matrix.T
        orientation = "rows"
        reduction_dimension = columns
    else:
        raw = matrix.T @ matrix
        orientation = "columns"
        reduction_dimension = rows
    trace = float(torch.trace(raw).item())
    tolerance = numerical_tolerance(compute_dtype, reduction_dimension, trace)
    denominator = max(trace, tolerance)
    normalized = raw / denominator
    return GramRecord(
        raw=raw,
        normalized=normalized,
        trace=trace,
        trace_tolerance=tolerance,
        source_dtype=str(weight.dtype).removeprefix("torch."),
        compute_dtype=str(compute_dtype).removeprefix("torch."),
        orientation=orientation,
        weight_shape=(rows, columns),
    )


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


def _top_two(
    matrix: torch.Tensor,
    *,
    layer: int,
    svd_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
    if rank == 0:
        return (
            torch.empty((matrix.shape[0], 0), device=matrix.device, dtype=matrix.dtype),
            torch.empty((0,), device=matrix.device, dtype=matrix.dtype),
        )
    if svd_mode == "exact" or rank <= 2:
        left, singular_values, _right = torch.linalg.svd(matrix, full_matrices=False)
    elif svd_mode == "randomized":
        left, singular_values, _right = _deterministic_topk_svd(
            matrix,
            # Oversampling makes the leading two values accurate enough for a
            # tail-energy diagnostic while retaining bounded storage.
            top_k=min(8, rank),
            seed=701408733 + int(layer),
        )
    else:
        raise ValueError(f"Unknown SVD mode: {svd_mode}")
    take = min(2, int(singular_values.numel()))
    return (
        left[:, :take].to(device=matrix.device, dtype=matrix.dtype),
        singular_values[:take].to(device=matrix.device, dtype=matrix.dtype),
    )


def _rank_two_residual(
    matrix: torch.Tensor,
    *,
    layer: int,
    svd_mode: str,
) -> tuple[torch.Tensor, float, torch.dtype]:
    """Return leading values and a stable blockwise rank-two residual."""
    rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
    if svd_mode == "exact" or rank <= 2:
        left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
        solver_dtype = matrix.dtype
    elif svd_mode == "randomized":
        left, singular_values, right = _deterministic_topk_svd(
            matrix,
            top_k=min(8, rank),
            seed=701408733 + int(layer),
        )
        solver_dtype = torch.float32
    else:
        raise ValueError(f"Unknown SVD mode: {svd_mode}")
    take = min(2, int(singular_values.numel()))
    leading = singular_values[:take].to(device=matrix.device)
    left = left[:, :take].to(device=matrix.device, dtype=torch.float64)
    right = right[:take, :].to(device=matrix.device, dtype=torch.float64)
    scaled_left = left * leading[:take].double().unsqueeze(0)
    residual_squared = 0.0
    for start in range(0, int(matrix.shape[0]), 256):
        stop = min(int(matrix.shape[0]), start + 256)
        block = matrix[start:stop].double() - scaled_left[start:stop] @ right
        residual_squared += float(block.square().sum().item())
    return leading.to(device=matrix.device, dtype=matrix.dtype), math.sqrt(residual_squared), solver_dtype


def _residual_profile(
    current: GramRecord,
    previous: GramRecord,
    following: GramRecord,
    *,
    layer: int,
    eligible: bool,
    svd_mode: str,
) -> dict[str, Any]:
    neighbor = (previous.normalized + following.normalized) / 2.0
    residual = current.normalized - neighbor
    raw_neighbor = (previous.raw + following.raw) / 2.0
    raw_residual = current.raw - raw_neighbor

    full_frobenius = float(torch.linalg.vector_norm(residual).item())
    basis, singular_values = _top_two(residual, layer=layer, svd_mode=svd_mode)
    top_two = float(torch.linalg.vector_norm(singular_values).item())
    support = basis.T @ neighbor @ basis
    projected_residual = basis.T @ residual @ basis
    support_frobenius = float(torch.linalg.vector_norm(support).item())
    support_tolerance = numerical_tolerance(
        residual.dtype,
        int(residual.shape[0]),
        max(support_frobenius, full_frobenius),
    )
    relative_top_two = float(torch.linalg.vector_norm(projected_residual).item()) / max(
        support_frobenius,
        support_tolerance,
    )

    raw_full_frobenius = float(torch.linalg.vector_norm(raw_residual).item())
    _raw_basis, raw_singular_values = _top_two(
        raw_residual,
        layer=layer + 104729,
        svd_mode=svd_mode,
    )
    raw_top_two = float(torch.linalg.vector_norm(raw_singular_values).item())
    singular_1 = float(singular_values[0].item()) if singular_values.numel() else 0.0
    singular_2 = float(singular_values[1].item()) if singular_values.numel() > 1 else 0.0
    raw_singular_1 = float(raw_singular_values[0].item()) if raw_singular_values.numel() else 0.0
    raw_singular_2 = float(raw_singular_values[1].item()) if raw_singular_values.numel() > 1 else 0.0
    concentration = min(1.0, (top_two * top_two) / (full_frobenius * full_frobenius)) if full_frobenius > 0.0 else 0.0

    control = _weighted_spectrum_profile(
        current.normalized,
        neighbor,
        layer=layer,
        neighbors=(previous.normalized, following.normalized),
        fields=FOOTPRINT_PROFILE_FIELDS,
    )
    return {
        "eligible": bool(eligible),
        "m0": full_frobenius,
        "m1": top_two,
        "m2": relative_top_two,
        "m3": float(control["relative_subspace_frobenius"]),
        "m0_raw": raw_full_frobenius,
        "m1_raw": raw_top_two,
        "sigma_1": singular_1,
        "sigma_2": singular_2,
        "raw_sigma_1": raw_singular_1,
        "raw_sigma_2": raw_singular_2,
        "full_frobenius": full_frobenius,
        "raw_full_frobenius": raw_full_frobenius,
        "top2_concentration": concentration,
        "neighbor_support_frobenius": support_frobenius,
        "neighbor_support_tolerance": support_tolerance,
        "m3_rank2_energy": float(control["rank2_energy"]),
        "m3_bilateral_coherence": float(control["bilateral_coherence"]),
        "m3_bilateral_balance": float(control["bilateral_balance"]),
        "trace": current.trace,
        "trace_tolerance": current.trace_tolerance,
        "source_dtype": current.source_dtype,
        "compute_dtype": current.compute_dtype,
        "orientation": current.orientation,
        "weight_shape": list(current.weight_shape),
    }


def score_weight_sequence(
    weights: Mapping[int, torch.Tensor],
    *,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    svd_mode: str = "exact",
    included_layers: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Compute M0--M3 and raw-Gram diagnostics with a three-Gram cache."""
    layers = sorted(int(layer) for layer in weights)
    eligible = eligible_layers(layers, trim_fraction=trim_fraction)
    eligible_set = set(eligible)
    scoreable = set(layers[1:-1])
    requested = scoreable if included_layers is None else scoreable.intersection(included_layers)
    positions = {layer: index for index, layer in enumerate(layers)}
    cache: dict[int, GramRecord] = {}
    profiles: dict[str, dict[str, Any]] = {}

    for layer in sorted(requested):
        index = positions[layer]
        neighborhood = layers[index - 1 : index + 2]
        for neighbor_layer in neighborhood:
            if neighbor_layer not in cache:
                cache[neighbor_layer] = hidden_gram(weights[neighbor_layer])
        profiles[str(layer)] = _residual_profile(
            cache[neighborhood[1]],
            cache[neighborhood[0]],
            cache[neighborhood[2]],
            layer=layer,
            eligible=layer in eligible_set,
            svd_mode=svd_mode,
        )
        cache = {cached_layer: record for cached_layer, record in cache.items() if positions[cached_layer] >= index}

    return {
        "layers": layers,
        "eligible_layers": eligible,
        "excluded_layers": [layer for layer in layers if layer not in eligible_set],
        "profiles": profiles,
    }


def select_candidate(
    profiles: Mapping[str, Mapping[str, Any]],
    eligible: list[int],
    candidate: str,
) -> dict[str, Any]:
    """Select the lowest layer on exact ties and report the top-two margin."""
    field = CANDIDATE_FIELDS[candidate]
    available = [
        int(layer) for layer in eligible if str(layer) in profiles and profiles[str(layer)].get(field) is not None
    ]
    if not available:
        return {"selected_layer": None, "score": 0.0, "margin": 0.0}
    ordered = sorted(
        available,
        key=lambda layer: (-float(profiles[str(layer)][field]), int(layer)),
    )
    best = ordered[0]
    best_score = float(profiles[str(best)][field])
    second_score = float(profiles[str(ordered[1])][field]) if len(ordered) > 1 else 0.0
    return {
        "selected_layer": int(best),
        "score": best_score,
        "margin": best_score - second_score,
    }


def blind_robust_peak(
    profiles: Mapping[str, Mapping[str, Any]],
    eligible: list[int],
    candidate: str,
    *,
    cutoff: float,
) -> dict[str, Any]:
    """B1: one candidate score, one global cutoff, and one boolean."""
    if not math.isfinite(float(cutoff)) or float(cutoff) < 0.0:
        raise ValueError("B1 cutoff must be a finite non-negative global value")
    field = CANDIDATE_FIELDS[candidate]
    layers = [layer for layer in eligible if str(layer) in profiles]
    if not layers:
        return {
            "is_rome_like": False,
            "verdict": "insufficient_layers",
            "selected_layer": None,
            "robust_peak": 0.0,
            "cutoff": float(cutoff),
        }
    values = np.asarray([float(profiles[str(layer)][field]) for layer in layers])
    if not np.all(np.isfinite(values)):
        raise ValueError("B1 candidate scores contain non-finite values")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    tolerance = numerical_tolerance(torch.float64, len(layers), max(abs(center), abs(peak)))
    robust_peak = max(0.0, (peak - center) / max(mad, tolerance))
    detected = bool(robust_peak > float(cutoff))
    return {
        "is_rome_like": detected,
        "verdict": "rome_suspicion" if detected else "not_rome_suspicion",
        "threat_model": "blind_suspect_only",
        "calibration": "global_cutoff_supplied_by_evaluator",
        "candidate": candidate,
        "selected_layer": int(layers[peak_index]),
        "robust_peak": robust_peak,
        "median": center,
        "mad": mad,
        "effective_mad": max(mad, tolerance),
        "cutoff": float(cutoff),
    }


def footprint_control(
    profiles: Mapping[str, Mapping[str, Any]],
    eligible: list[int],
) -> dict[str, Any]:
    """B2: current morphology/universal-bound rule on the shared eligible set."""
    layers = [layer for layer in eligible if str(layer) in profiles]
    if not layers:
        return {
            "is_rome_like": False,
            "verdict": "insufficient_layers",
            "localized_layer": None,
            "presence_peak_layer": None,
        }
    control_profiles = {
        str(layer): {
            "relative_subspace_frobenius": float(profiles[str(layer)]["m3"]),
            "rank2_energy": float(profiles[str(layer)]["m3_rank2_energy"]),
            "bilateral_coherence": float(profiles[str(layer)]["m3_bilateral_coherence"]),
            "bilateral_balance": float(profiles[str(layer)]["m3_bilateral_balance"]),
        }
        for layer in layers
    }
    raw = np.asarray([control_profiles[str(layer)]["relative_subspace_frobenius"] for layer in layers])
    localized_layer = int(layers[int(np.argmax(raw))])
    footprint = _universal_outlier(_presence_series(control_profiles, layers, "footprint"))
    peak_guard = _universal_outlier(_presence_series(control_profiles, layers, "peak"))
    peak_layer = int(layers[int(footprint["peak_index"])])
    detected = bool(footprint["is_outlier"]) and bool(peak_guard["is_outlier"])
    return {
        "is_rome_like": detected,
        "verdict": "rome_like" if detected else "no_universal_outlier",
        "threat_model": "blind_suspect_only",
        "calibration": "current_universal_bound_not_empirically_calibrated",
        "localized_layer": localized_layer,
        "presence_peak_layer": peak_layer,
        "evidence": footprint,
        "peak_guard_evidence": peak_guard,
    }


def _gram_delta_profile(
    suspect: torch.Tensor,
    clean: torch.Tensor,
    *,
    layer: int,
    svd_mode: str,
) -> dict[str, Any]:
    suspect_record = hidden_gram(suspect)
    clean_record = hidden_gram(clean)
    if suspect_record.raw.shape != clean_record.raw.shape:
        raise ValueError(f"Layer {layer} hidden Gram shape changed")
    delta = suspect_record.raw - clean_record.raw
    full = float(torch.linalg.vector_norm(delta).item())
    singular_values, tail, solver_dtype = _rank_two_residual(
        delta,
        layer=layer + 209759,
        svd_mode=svd_mode,
    )
    clean_frobenius = float(torch.linalg.vector_norm(clean_record.raw).item())
    eps = torch.finfo(solver_dtype).eps
    reduction_dimension = max(int(value) for value in suspect_record.weight_shape)
    product = min(0.5, float(eps) * max(1, reduction_dimension))
    gamma = product / max(1.0 - product, float(eps))
    noise_bound = gamma * (float(torch.linalg.vector_norm(suspect_record.raw).item()) + clean_frobenius)
    magnitude = full / max(
        clean_frobenius,
        numerical_tolerance(delta.dtype, int(delta.shape[0]), clean_frobenius),
    )
    magnitude_bound = noise_bound / max(
        clean_frobenius,
        numerical_tolerance(delta.dtype, int(delta.shape[0]), clean_frobenius),
    )
    tail_ratio = tail / max(
        full,
        numerical_tolerance(delta.dtype, int(delta.shape[0]), full),
    )
    tail_ratio_bound = noise_bound / max(
        full,
        numerical_tolerance(delta.dtype, int(delta.shape[0]), full),
    )
    return {
        "magnitude": magnitude,
        "magnitude_bound": magnitude_bound,
        "tail_ratio": tail_ratio,
        "tail_ratio_bound": tail_ratio_bound,
        "delta_frobenius": full,
        "tail_frobenius": tail,
        "noise_bound": noise_bound,
        "solver_dtype": str(solver_dtype).removeprefix("torch."),
        "sigma_1": float(singular_values[0].item()) if singular_values.numel() else 0.0,
        "sigma_2": (float(singular_values[1].item()) if singular_values.numel() > 1 else 0.0),
        "passes_dtype_bound": bool(magnitude > magnitude_bound and tail_ratio <= tail_ratio_bound),
    }


def clean_reference_decision(
    delta_profiles: Mapping[str, Mapping[str, Any]],
    eligible: list[int],
) -> dict[str, Any]:
    """B0: dtype-bounded, clean-reference low-rank Gram delta decision."""
    layers = [layer for layer in eligible if str(layer) in delta_profiles]
    if not layers:
        return {
            "is_rome_like": False,
            "verdict": "no_detectable_change",
            "selected_layer": None,
            "threat_model": "clean_reference",
        }
    selected = sorted(
        layers,
        key=lambda layer: (
            -float(delta_profiles[str(layer)]["magnitude"]),
            int(layer),
        ),
    )[0]
    detected = bool(delta_profiles[str(selected)]["passes_dtype_bound"])
    return {
        "is_rome_like": detected,
        "verdict": ("rome_compatible_low_rank_edit" if detected else "not_rome_compatible_low_rank_edit"),
        "selected_layer": int(selected),
        "threat_model": "clean_reference",
        "attribution_scope": "generic_rank_at_most_two_gram_change",
        "evidence": dict(delta_profiles[str(selected)]),
    }


def evaluate_capture_data(
    baseline: Mapping[str, Any],
    suspect_patch: Mapping[str, Any],
    *,
    blind_candidate: str,
    blind_cutoff: float | None,
) -> dict[str, Any]:
    """Merge a baseline/patch capture pair and evaluate M0--M3 and B0--B2."""
    if baseline.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("Baseline capture has an unsupported schema version")
    if suspect_patch.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("Suspect capture has an unsupported schema version")
    profiles = {str(layer): dict(profile) for layer, profile in dict(baseline.get("profiles", {})).items()}
    profiles.update({str(layer): dict(profile) for layer, profile in dict(suspect_patch.get("profiles", {})).items()})
    eligible = [int(layer) for layer in suspect_patch.get("eligible_layers", ())]
    candidates = {candidate: select_candidate(profiles, eligible, candidate) for candidate in CANDIDATE_FIELDS}
    b2 = footprint_control(profiles, eligible)
    if blind_cutoff is None:
        b1 = {
            "status": "not_evaluated_uncalibrated",
            "is_rome_like": None,
            "verdict": "not_evaluated_uncalibrated",
            "selected_layer": None,
            "candidate": blind_candidate,
            "cutoff": None,
            "threat_model": "blind",
        }
    else:
        b1 = blind_robust_peak(
            profiles,
            eligible,
            blind_candidate,
            cutoff=blind_cutoff,
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "profiles": profiles,
        "eligible_layers": eligible,
        "excluded_layers": [int(layer) for layer in suspect_patch.get("excluded_layers", ())],
        "candidates": candidates,
        "localized_layer": b2["localized_layer"],
        "presence_peak_layer": b2["presence_peak_layer"],
        "binary": {
            "B0": clean_reference_decision(
                dict(suspect_patch.get("delta_profiles", {})),
                eligible,
            ),
            "B1": b1,
            "B2": b2,
        },
    }


def _actual_changed_layers(context: CaptureContext, layers: list[int]) -> list[int]:
    if context.is_baseline or context.baseline_proj_weights is None:
        return []
    declared = context.changed_layers("proj", layers)
    changed: list[int] = []
    for layer in declared:
        current = context.proj_weights[layer]
        baseline = context.baseline_proj_weights[layer]
        if current is baseline:
            continue
        if current.shape == baseline.shape and torch.equal(current, baseline):
            continue
        changed.append(int(layer))
    return changed


def capture_rome_math_ablation(context: CaptureContext) -> dict[str, Any]:
    """Structural capture producer for an isolated M0--M3 recapture."""
    started = time.perf_counter()
    layers = sorted(int(layer) for layer in context.proj_weights)
    eligible = eligible_layers(layers)
    changed = _actual_changed_layers(context, layers)
    if context.is_baseline:
        included: Optional[set[int]] = None
    else:
        positions = {layer: index for index, layer in enumerate(layers)}
        included = set()
        for layer in changed:
            index = positions[layer]
            included.update(layers[max(0, index - 1) : min(len(layers), index + 2)])
    scored = score_weight_sequence(
        context.proj_weights,
        trim_fraction=DEFAULT_TRIM_FRACTION,
        svd_mode=DEFAULT_SVD_MODE,
        included_layers=included,
    )
    delta_profiles: dict[str, dict[str, Any]] = {}
    if not context.is_baseline and context.baseline_proj_weights is not None:
        for layer in changed:
            delta_profiles[str(layer)] = _gram_delta_profile(
                context.proj_weights[layer],
                context.baseline_proj_weights[layer],
                layer=layer,
                svd_mode=DEFAULT_SVD_MODE,
            )
    hidden_dimensions = [min(int(value) for value in context.proj_weights[layer].shape) for layer in layers]
    compute_bytes = torch.tensor([], dtype=torch.float32).element_size()
    peak_dimension = max(hidden_dimensions, default=0)
    estimated_peak_bytes = int(7 * peak_dimension * peak_dimension * compute_bytes)
    return to_serializable(
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "scientific_baseline": False,
            "mode": "baseline" if context.is_baseline else "patch",
            "candidate_versions": list(CANDIDATE_FIELDS),
            "trim_fraction": DEFAULT_TRIM_FRACTION,
            "svd_mode": DEFAULT_SVD_MODE,
            "layers": layers,
            "eligible_layers": eligible,
            "excluded_layers": [layer for layer in layers if layer not in set(eligible)],
            "profiles": scored["profiles"],
            "delta_profiles": delta_profiles,
            "changed_layers": {"proj": changed},
            "scored_layers": [int(layer) for layer in scored["profiles"]],
            "runtime_seconds": time.perf_counter() - started,
            "estimated_peak_bytes": estimated_peak_bytes,
        }
    )


__all__ = [
    "CANDIDATE_FIELDS",
    "CAPTURE_SCHEMA_VERSION",
    "DEFAULT_TRIM_FRACTION",
    "EVALUATION_SCHEMA_VERSION",
    "blind_robust_peak",
    "capture_rome_math_ablation",
    "clean_reference_decision",
    "eligible_layers",
    "evaluate_capture_data",
    "footprint_control",
    "hidden_gram",
    "numerical_tolerance",
    "score_weight_sequence",
    "select_candidate",
]

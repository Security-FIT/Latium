"""Architecture-neutral layer localization for ROME-compatible edits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from src.common.linalg import gpu_svd_topk


SCORE_FIELD = "diagonal_relative"
PROFILE_FIELDS = (SCORE_FIELD,)
EXPERIMENT_CAPTURE_VERSION = "gram-experiments-v1"
DIRECTIONAL_CAPTURE_VERSION = "gram-directional-error-v1"
CROSS_LAYER_CAPTURE_VERSION = "gram-cross-layer-v1"
EXPERIMENT_GROUPS = ("neighbors", "quadratic", "footprint")
DEFAULT_EXPERIMENT_GROUPS = ("neighbors",)
DEFAULT_TRIM_FRACTION = 0.10
RELATIVE_DECISION_VERSION = "relative-profile-b0-v1"
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


def _top_residual_subspace(
    residual: torch.Tensor,
    *,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cuda_devices = [residual.device.index or 0] if residual.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(_SVD_SEED + int(layer))
        if residual.is_cuda:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(_SVD_SEED + int(layer))
        left, singular_values, _right = gpu_svd_topk(
            residual,
            k=min(2, min(residual.shape)),
            niter=4,
            device=str(residual.device),
        )
    return (
        left.to(device=residual.device, dtype=residual.dtype),
        singular_values.to(device=residual.device, dtype=residual.dtype),
    )


def _residual_measurements(
    current: torch.Tensor,
    predicted: torch.Tensor,
    support: torch.Tensor,
    *,
    layer: int,
) -> dict[str, Any]:
    if current.shape != predicted.shape or current.shape != support.shape:
        raise ValueError(f"Layer {layer} experimental Gram shapes do not match")
    residual = current - predicted
    basis, singular_values = _top_residual_subspace(residual, layer=layer)
    support_subspace = basis.T @ support @ basis
    support_scale = torch.linalg.matrix_norm(support_subspace, ord="fro")
    residual_scale = torch.linalg.matrix_norm(residual, ord="fro")
    tolerance = numerical_tolerance(
        residual.dtype,
        int(residual.shape[0]),
        max(float(support_scale.item()), float(residual_scale.item())),
    )
    raw_support = torch.diagonal(support_subspace)
    clamped_support = raw_support.clamp_min(tolerance)
    original_score = torch.linalg.vector_norm(singular_values / clamped_support)
    top2 = torch.linalg.vector_norm(singular_values)
    concentration = None
    if float(residual_scale.item()) > tolerance:
        concentration = float((top2.square() / residual_scale.square()).item())
    return {
        "basis": basis,
        "residual": residual,
        "singular_values": singular_values,
        "raw_support": raw_support,
        "tolerance": tolerance,
        "original_score": float(original_score.item()),
        "residual_frobenius": float(residual_scale.item()),
        "top2_frobenius": float(top2.item()),
        "top2_concentration": concentration,
        "support_clamped": [bool(value <= tolerance) for value in raw_support.tolist()],
    }


def neighbor_experiment_measurements(
    left: torch.Tensor,
    current: torch.Tensor,
    right: torch.Tensor,
    *,
    layer: int,
) -> dict[str, Any]:
    """Capture Part 2 scores while sharing the original residual subspace."""
    reference = (left + right) * 0.5
    measured = _residual_measurements(current, reference, reference, layer=layer)
    basis = measured.pop("basis")
    residual = measured.pop("residual")
    singular_values = measured["singular_values"]
    raw_support = measured["raw_support"]
    tolerance = float(measured["tolerance"])

    projected_left = basis.T @ (current - left) @ basis
    projected_right = basis.T @ (current - right) @ basis
    numerator = 2.0 * max(
        0.0,
        float(torch.sum(projected_left * projected_right).item()),
    )
    agreement_denominator = float(
        (projected_left.square().sum() + projected_right.square().sum()).item()
    )
    agreement = numerator / agreement_denominator if agreement_denominator > tolerance * tolerance else 0.0
    agreement = min(1.0, max(0.0, agreement))

    signed_small = (basis.T @ residual @ basis)
    signed_small = (signed_small + signed_small.T) * 0.5
    signed_values, rotation = torch.linalg.eigh(signed_small)
    order = torch.argsort(torch.abs(signed_values), descending=True)
    signed_values = signed_values[order]
    refined_basis = basis @ rotation[:, order]
    current_support = torch.diagonal(refined_basis.T @ current @ refined_basis)
    neighbor_support = torch.diagonal(refined_basis.T @ reference @ refined_basis)
    if bool((current_support < -tolerance).any()) or bool((neighbor_support < -tolerance).any()):
        raise ValueError(f"Layer {layer} has materially negative projected Gram support")
    current_support = current_support.clamp_min(0.0)
    neighbor_support = neighbor_support.clamp_min(0.0)
    refined_denominator = neighbor_support.clamp_min(tolerance)
    refined_ratio = torch.linalg.vector_norm(torch.abs(signed_values) / refined_denominator)
    support_sum = current_support + neighbor_support
    resolved = support_sum > tolerance
    contrasts = torch.zeros_like(support_sum)
    contrasts[resolved] = 2.0 * torch.abs(current_support[resolved] - neighbor_support[resolved]) / support_sum[resolved]

    return {
        **{key: value for key, value in measured.items() if key not in {"singular_values", "raw_support"}},
        "singular_values": [float(value) for value in singular_values.tolist()],
        "neighbor_support": [float(value) for value in raw_support.tolist()],
        "projected_left_difference": projected_left.tolist(),
        "projected_right_difference": projected_right.tolist(),
        "neighbor_agreement": float(agreement),
        "agreement_score": float(agreement * measured["original_score"]),
        "refined_signed_eigenvalues": [float(value) for value in signed_values.tolist()],
        "refined_current_support": [float(value) for value in current_support.tolist()],
        "refined_neighbor_support": [float(value) for value in neighbor_support.tolist()],
        "refined_ratio_score": float(refined_ratio.item()),
        "bounded_contrast_score": float(torch.linalg.vector_norm(contrasts).item()),
        "numerical_status": "ok",
    }


def _directional_error_measurement(
    grams: Mapping[int, torch.Tensor],
    window: list[int],
    *,
    candidate_index: int,
) -> dict[str, Any]:
    """Measure V0/V0R/V1/V2 together in one candidate's signed directions."""
    layer = window[candidate_index]
    current = grams[layer]
    neighbor_support = (grams[window[candidate_index - 1]] + grams[window[candidate_index + 1]]) * 0.5
    measured = _residual_measurements(current, neighbor_support, neighbor_support, layer=layer)
    basis = measured["basis"]
    residual = measured["residual"]
    small = basis.T @ residual @ basis
    small = (small + small.T) * 0.5
    signed_values, rotation = torch.linalg.eigh(small)
    order = torch.argsort(torch.abs(signed_values), descending=True)
    signed_values = signed_values[order]
    refined_basis = basis @ rotation[:, order]
    candidate_support = torch.diagonal(refined_basis.T @ neighbor_support @ refined_basis)
    tolerance = float(measured["tolerance"])
    if bool((candidate_support < -tolerance).any()):
        raise ValueError(f"Layer {layer} has materially negative directional support")
    candidate_support = candidate_support.clamp_min(tolerance)
    refined_score = float(torch.linalg.vector_norm(torch.abs(signed_values) / candidate_support).item())

    reference_positions = sorted(
        (
            position for position in range(1, len(window) - 1)
            if abs(position - candidate_index) > 1
        ),
        key=lambda position: (abs(position - candidate_index), window[position]),
    )[:6]
    if len(reference_positions) != 6:
        raise ValueError("Directional error capture requires six reference residual centers")
    reference_layers = [window[position] for position in reference_positions]
    projected_residuals: list[torch.Tensor] = []
    projected_supports: list[torch.Tensor] = []
    normalized_errors: list[torch.Tensor] = []
    for position in reference_positions:
        reference_layer = window[position]
        reference_center = (grams[window[position - 1]] + grams[window[position + 1]]) * 0.5
        reference_residual = grams[reference_layer] - reference_center
        projected = refined_basis.T @ reference_residual @ refined_basis
        projected = (projected + projected.T) * 0.5
        support = torch.diagonal(refined_basis.T @ reference_center @ refined_basis)
        if bool((support < -tolerance).any()):
            raise ValueError(f"Reference layer {reference_layer} has materially negative directional support")
        support = support.clamp_min(tolerance)
        projected_residuals.append(projected)
        projected_supports.append(support)
        normalized_errors.append(torch.diagonal(projected) / support)

    reference_errors = torch.stack(normalized_errors)
    center = torch.median(reference_errors, dim=0).values
    candidate_error = signed_values / candidate_support
    centered = candidate_error - center
    centered_score = float(torch.linalg.vector_norm(centered).item())
    absolute_deviations = torch.abs(reference_errors - center)
    mad = 1.482602218505602 * torch.median(absolute_deviations, dim=0).values
    scale_tolerance = numerical_tolerance(
        reference_errors.dtype,
        len(reference_layers),
        max(1.0, float(torch.linalg.vector_norm(reference_errors).item())),
    )
    resolved = mad > scale_tolerance
    standardized_score: float | None = None
    v2_status = "unavailable"
    if bool(resolved.all()):
        standardized_score = float(torch.linalg.vector_norm(centered / mad).item())
        v2_status = "ok"

    return {
        "original_score": float(measured["original_score"]),
        "refined_ratio_score": refined_score,
        "centered_directional_score": centered_score,
        "standardized_directional_score": standardized_score,
        "candidate_projection": (refined_basis.T @ residual @ refined_basis).tolist(),
        "candidate_directional_error": candidate_error.tolist(),
        "candidate_support": candidate_support.tolist(),
        "reference_layers": reference_layers,
        "reference_projections": [projection.tolist() for projection in projected_residuals],
        "reference_supports": [support.tolist() for support in projected_supports],
        "reference_directional_errors": reference_errors.tolist(),
        "reference_median": center.tolist(),
        "reference_mad": mad.tolist(),
        "refined_signed_eigenvalues": signed_values.tolist(),
        "precision": str(current.dtype),
        "tolerance": tolerance,
        "mad_tolerance": scale_tolerance,
        "v0_status": "ok",
        "v0r_status": "ok",
        "v1_status": "ok",
        "v2_status": v2_status,
        "numerical_status": "ok" if v2_status == "ok" else "partial",
    }


def capture_directional_error_weights(weights: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    """Capture directional-error profiles from contiguous eleven-layer windows."""
    layers = sorted(int(layer) for layer in weights)
    positions = {layer: index for index, layer in enumerate(layers)}
    candidates = [
        layers[index]
        for index in range(5, len(layers) - 5)
        if all(layers[offset + 1] == layers[offset] + 1 for offset in range(index - 5, index + 5))
    ]
    grams: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for layer in candidates:
        index = positions[layer]
        window = layers[index - 5 : index + 6]
        try:
            for other in window:
                if other not in grams:
                    grams[other] = hidden_gram(weights[other])
            shapes = {tuple(grams[other].shape) for other in window}
            if len(shapes) != 1:
                raise ValueError("eleven-layer Gram window has incompatible shapes")
            profiles[str(layer)] = _directional_error_measurement(grams, window, candidate_index=5)
        except (RuntimeError, ValueError) as exc:
            profiles[str(layer)] = {
                "original_score": None,
                "refined_ratio_score": None,
                "centered_directional_score": None,
                "standardized_directional_score": None,
                "numerical_status": "unavailable",
                "reason": str(exc),
            }
        minimum_position = index - 4
        grams = {
            cached_layer: gram
            for cached_layer, gram in grams.items()
            if positions[cached_layer] >= minimum_position
        }
    return {
        "mode": "single_checkpoint",
        "capture_version": DIRECTIONAL_CAPTURE_VERSION,
        "layers": layers,
        "eligible_layers": candidates,
        "excluded_layers": [layer for layer in layers if layer not in set(candidates)],
        "reference_count": 6,
        "required_contiguous_layers": 11,
        "profiles": profiles,
    }


def lof_scores_from_kernel(
    kernel: torch.Tensor,
    *,
    neighbors: int = 5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute deterministic LOF scores with explicit distance ties."""
    matrix = torch.as_tensor(kernel, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cross-layer kernel must be square")
    count = matrix.shape[0]
    if count <= neighbors:
        raise ValueError(f"LOF requires more than {neighbors} layers")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Cross-layer kernel contains non-finite values")
    diagonal = torch.diagonal(matrix)
    squared = diagonal[:, None] + diagonal[None, :] - 2.0 * matrix
    tolerance = numerical_tolerance(matrix.dtype, count, max(1.0, float(matrix.abs().max().item())))
    if float(squared.min().item()) < -tolerance:
        raise ValueError("Cross-layer kernel produces materially negative squared distances")
    distances = torch.sqrt(squared.clamp_min(0.0))
    off_diagonal = distances[~torch.eye(count, dtype=torch.bool)]
    if bool((off_diagonal <= tolerance).all()):
        return torch.ones(count, dtype=torch.float64), {
            "neighbors": neighbors,
            "tie_expanded_neighbor_counts": [count - 1] * count,
            "identical_graph": True,
            "distance_tolerance": tolerance,
        }

    neighborhoods: list[torch.Tensor] = []
    k_distances = torch.empty(count, dtype=torch.float64)
    indices = torch.arange(count)
    for row in range(count):
        mask = indices != row
        available = distances[row, mask]
        kth = torch.sort(available).values[neighbors - 1]
        k_distances[row] = kth
        neighborhoods.append(indices[mask & (distances[row] <= kth + tolerance)])
    local_reachability = torch.empty(count, dtype=torch.float64)
    for row, neighborhood in enumerate(neighborhoods):
        reachability = torch.maximum(k_distances[neighborhood], distances[row, neighborhood])
        mean_reachability = float(reachability.mean().item())
        if mean_reachability <= tolerance:
            raise ValueError("LOF is unresolved for a duplicate subcluster")
        local_reachability[row] = 1.0 / mean_reachability
    scores = torch.stack([
        torch.mean(local_reachability[neighborhood] / local_reachability[row])
        for row, neighborhood in enumerate(neighborhoods)
    ])
    return scores, {
        "neighbors": neighbors,
        "tie_expanded_neighbor_counts": [int(neighborhood.numel()) for neighborhood in neighborhoods],
        "identical_graph": False,
        "distance_tolerance": tolerance,
    }


def robust_decomposition_scores(
    observations: torch.Tensor,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decompose layer rows into low-rank structure and row-sparse anomalies."""
    data = torch.as_tensor(observations, dtype=torch.float64, device="cpu")
    if data.ndim != 2 or min(data.shape) < 2 or not bool(torch.isfinite(data).all()):
        raise ValueError("Decomposition requires a finite layer-by-feature matrix")
    layer_count = data.shape[0]
    regularization = 2.0 / math.sqrt(layer_count)
    norm = float(torch.linalg.matrix_norm(data, ord="fro").item())
    if norm == 0.0:
        raise ValueError("Decomposition input is trivial")
    spectral = float(torch.linalg.matrix_norm(data, ord=2).item())
    dual_norm = max(spectral, float(torch.linalg.vector_norm(data, dim=1).max().item()) / regularization)
    dual = data / max(dual_norm, torch.finfo(torch.float64).tiny)
    low_rank = torch.zeros_like(data)
    sparse = torch.zeros_like(data)
    mu = 1.25 / max(spectral, torch.finfo(torch.float64).tiny)
    mu_max = mu * 1e7
    converged = False
    residual_ratio = math.inf
    rank = 0
    for iteration in range(1, max_iterations + 1):
        candidate = data - sparse + dual / mu
        left, singular_values, right = torch.linalg.svd(candidate, full_matrices=False)
        shrunk = torch.clamp(singular_values - 1.0 / mu, min=0.0)
        rank = int((shrunk > 0.0).sum().item())
        low_rank = (left[:, :rank] * shrunk[:rank]) @ right[:rank] if rank else torch.zeros_like(data)

        candidate_sparse = data - low_rank + dual / mu
        row_norms = torch.linalg.vector_norm(candidate_sparse, dim=1)
        shrink = torch.clamp(1.0 - (regularization / mu) / row_norms.clamp_min(torch.finfo(torch.float64).tiny), min=0.0)
        sparse = candidate_sparse * shrink[:, None]
        residual = data - low_rank - sparse
        residual_ratio = float(torch.linalg.matrix_norm(residual, ord="fro").item()) / norm
        dual = dual + mu * residual
        if residual_ratio <= tolerance:
            converged = True
            break
        mu = min(mu * 1.5, mu_max)
    if not converged:
        raise ValueError(f"Decomposition did not converge in {max_iterations} iterations")
    sparse_norm = float(torch.linalg.matrix_norm(sparse, ord="fro").item())
    low_rank_norm = float(torch.linalg.matrix_norm(low_rank, ord="fro").item())
    numerical = numerical_tolerance(data.dtype, max(data.shape), norm)
    if sparse_norm <= numerical or low_rank_norm <= numerical:
        raise ValueError("Decomposition is trivial")
    scores = torch.linalg.vector_norm(sparse, dim=1)
    return scores, {
        "solver": "deterministic-float64-row-sparse-admm-v1",
        "lambda": regularization,
        "iterations": iteration,
        "converged": converged,
        "residual_ratio": residual_ratio,
        "rank": rank,
        "low_rank_frobenius": low_rank_norm,
        "sparse_frobenius": sparse_norm,
    }


def factor_cross_layer_kernel(kernel: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a deterministic row embedding whose Gram matrix is ``kernel``."""
    matrix = torch.as_tensor(kernel, dtype=torch.float64, device="cpu")
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    tolerance = numerical_tolerance(matrix.dtype, matrix.shape[0], max(1.0, float(matrix.abs().max().item())))
    if float(eigenvalues.min().item()) < -tolerance:
        raise ValueError("Cross-layer kernel is not positive semidefinite")
    positive = eigenvalues > tolerance
    if int(positive.sum().item()) < 2:
        raise ValueError("Cross-layer kernel has insufficient nontrivial rank")
    factor = eigenvectors[:, positive] * torch.sqrt(eigenvalues[positive])
    return factor, {
        "kernel_rank": int(positive.sum().item()),
        "minimum_eigenvalue": float(eigenvalues.min().item()),
        "psd_tolerance": tolerance,
    }


def evaluate_cross_layer_experiments(
    capture: Mapping[str, Any],
    *,
    experiments: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Derive LOF/decomposition profiles from a shared cross-layer kernel."""
    layers = [int(layer) for layer in capture.get("layers", [])]
    kernel = torch.tensor(capture.get("kernel", []), dtype=torch.float64)
    if kernel.shape != (len(layers), len(layers)):
        raise ValueError("Cross-layer capture kernel does not match its layer metadata")
    output: dict[str, Any] = {}
    for identifier in experiments:
        try:
            if identifier == "lof-relative-b0-v1":
                scores, method_diagnostics = lof_scores_from_kernel(kernel, neighbors=5)
            elif identifier == "decomposition-relative-b0-v1":
                factor, factor_diagnostics = factor_cross_layer_kernel(kernel)
                scores, method_diagnostics = robust_decomposition_scores(factor)
                method_diagnostics = {**factor_diagnostics, **method_diagnostics}
            else:
                raise ValueError(f"Unknown cross-layer experiment: {identifier}")
            decision = relative_profile_decision(
                {str(layer): float(scores[index].item()) for index, layer in enumerate(layers)},
                eligible_layers=layers,
                experiment_id=identifier,
            )
            decision["diagnostics"]["evidence_method"] = method_diagnostics
            output[identifier] = decision
        except (RuntimeError, ValueError) as exc:
            decision = relative_profile_decision({}, eligible_layers=layers, experiment_id=identifier)
            decision["diagnostics"]["reason"] = str(exc)
            output[identifier] = decision
    return output


def quadratic_experiment_measurements(
    left2: torch.Tensor,
    left1: torch.Tensor,
    current: torch.Tensor,
    right1: torch.Tensor,
    right2: torch.Tensor,
    *,
    layer: int,
) -> dict[str, Any]:
    """Score a fixed four-neighbor quadratic prediction with positive support."""
    predicted = (left1 + right1) * (2.0 / 3.0) - (left2 + right2) * (1.0 / 6.0)
    support = (left1 + right1) * 0.5
    measured = _residual_measurements(current, predicted, support, layer=layer)
    return {
        "quadratic_score": measured["original_score"],
        "quadratic_singular_values": [float(value) for value in measured["singular_values"].tolist()],
        "quadratic_directional_support": [float(value) for value in measured["raw_support"].tolist()],
        "quadratic_support_clamped": measured["support_clamped"],
        "quadratic_tolerance": measured["tolerance"],
        "quadratic_residual_frobenius": measured["residual_frobenius"],
        "quadratic_top2_frobenius": measured["top2_frobenius"],
        "quadratic_top2_concentration": measured["top2_concentration"],
        "quadratic_numerical_status": "ok",
    }


def _least_squares_rss(values: torch.Tensor, design: torch.Tensor) -> tuple[float, torch.Tensor]:
    if values.ndim != 1 or design.ndim != 2 or design.shape[0] != values.shape[0]:
        raise ValueError("Profile regression dimensions do not match")
    if design.shape[0] <= design.shape[1] or int(torch.linalg.matrix_rank(design).item()) < design.shape[1]:
        raise ValueError("Profile regression design is not identifiable")
    coefficients = torch.linalg.lstsq(design, values).solution
    residual = values - design @ coefficients
    return float(torch.dot(residual, residual).item()), coefficients


def relative_profile_decision(
    layer_scores: Mapping[str, float],
    *,
    eligible_layers: list[int] | tuple[int, ...] | None = None,
    experiment_id: str = "original-v3-relative-b0-v1",
    original_localizer_layer: int | None = None,
) -> dict[str, Any]:
    """Compare ordinary profiles with a jointly fitted positive excursion."""
    diagnostics: dict[str, Any] = {"transform": "log1p"}

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "decision_version": RELATIVE_DECISION_VERSION,
            "status": "unavailable",
            "rome_compatible_detected": None,
            "is_rome_like": None,
            "candidate_layer": None,
            "original_localizer_layer": original_localizer_layer,
            "eligible_layers": [] if eligible_layers is None else [int(value) for value in eligible_layers],
            "layer_scores": {},
            "background_cost": None,
            "anomaly_cost": None,
            "gain": None,
            "diagnostics": {**diagnostics, "reason": reason},
        }

    try:
        score_by_layer = {int(layer): float(score) for layer, score in layer_scores.items()}
    except (TypeError, ValueError) as exc:
        return unavailable(f"invalid profile: {exc}")
    declared = sorted(score_by_layer) if eligible_layers is None else [int(layer) for layer in eligible_layers]
    if len(set(declared)) != len(declared):
        return unavailable("eligible layers must be unique")
    declared = sorted(declared)
    declared_set = set(declared)
    missing = [layer for layer in declared if layer not in score_by_layer]
    diagnostics["extra_profile_layers"] = [layer for layer in score_by_layer if layer not in declared_set]
    if missing:
        return unavailable(f"incomplete eligible profile; missing layers {missing[:8]}")
    if len(declared) < 6:
        return unavailable("relative profile comparison requires at least six eligible observations")
    scores = [score_by_layer[layer] for layer in declared]
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        return unavailable("profile scores must be finite and non-negative")

    values = torch.log1p(torch.tensor(scores, dtype=torch.float64))
    if bool(torch.all(values == values[0])):
        return unavailable("profile is numerically constant")
    coordinates = torch.tensor(declared, dtype=torch.float64)
    coordinate_range = float((coordinates[-1] - coordinates[0]).item())
    if coordinate_range <= 0.0:
        return unavailable("layer coordinates have no range")
    depth = (coordinates - coordinates[0]) / coordinate_range
    n = len(declared)
    family_penalty = 2.0 * math.log(3.0)
    resolution = numerical_tolerance(
        values.dtype,
        n,
        max(1.0, float(torch.linalg.vector_norm(values).item())),
    ) ** 2
    diagnostics.update({
        "actual_layer_coordinates": declared,
        "scaled_layer_coordinates": depth.tolist(),
        "rss_resolution": resolution,
        "family_search_penalty": family_penalty,
    })

    affine = torch.stack((torch.ones_like(depth), depth), dim=1)
    background_designs: list[dict[str, Any]] = [
        {"family": "affine", "design": affine, "split": None, "search_cost": family_penalty},
        {
            "family": "quadratic",
            "design": torch.column_stack((affine, depth.square())),
            "split": None,
            "search_cost": family_penalty,
        },
    ]
    permitted_splits = list(range(2, n - 1))
    split_penalty = 2.0 * math.log(len(permitted_splits))
    for split in permitted_splits:
        step = (torch.arange(n, dtype=torch.int64) >= split).to(torch.float64)
        background_designs.append({
            "family": "affine-plus-step",
            "design": torch.column_stack((affine, step)),
            "split": split,
            "search_cost": family_penalty + split_penalty,
        })
    diagnostics["step_split_count"] = len(permitted_splits)
    diagnostics["split_search_penalty"] = split_penalty

    background_fits: list[dict[str, Any]] = []
    anomaly_fits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for spec in background_designs:
        design = spec["design"]
        try:
            rss, coefficients = _least_squares_rss(values, design)
        except ValueError as exc:
            rejected.append({"family": spec["family"], "split": spec["split"], "reason": str(exc)})
            continue
        if n - design.shape[1] <= 0:
            continue
        rss = max(rss, resolution)
        background_fits.append({
            **spec,
            "rss": rss,
            "coefficients": coefficients.tolist(),
            "cost": n * math.log(rss / n) + design.shape[1] * math.log(n) + spec["search_cost"],
        })
        candidates: list[dict[str, Any]] = []
        for index, layer in enumerate(declared):
            excursion = torch.zeros(n, dtype=torch.float64)
            excursion[index] = 1.0
            try:
                anomaly_rss, anomaly_coefficients = _constrained_profile_fit(values, design, [excursion])
            except ValueError:
                continue
            if n - (design.shape[1] + 1) <= 0:
                continue
            candidates.append({
                "candidate_layer": layer,
                "rss": max(anomaly_rss, resolution),
                "coefficients": anomaly_coefficients.tolist(),
                "amplitude": float(anomaly_coefficients[-1].item()),
            })
        location_count = len(candidates)
        if not location_count:
            continue
        location_penalty = 2.0 * math.log(location_count)
        for candidate in candidates:
            candidate.update({
                "family": spec["family"],
                "split": spec["split"],
                "location_count": location_count,
                "location_search_penalty": location_penalty,
                "cost": (
                    n * math.log(candidate["rss"] / n)
                    + (design.shape[1] + 1) * math.log(n)
                    + spec["search_cost"]
                    + location_penalty
                ),
            })
            anomaly_fits.append(candidate)

    diagnostics["rejected_fits"] = rejected
    if not background_fits or not anomaly_fits:
        return unavailable("no identifiable background/anomaly comparison")
    best_background = min(
        background_fits,
        key=lambda fit: (fit["cost"], fit["family"], -1 if fit["split"] is None else fit["split"]),
    )
    best_anomaly = min(
        anomaly_fits,
        key=lambda fit: (fit["cost"], fit["candidate_layer"], fit["family"]),
    )
    gain = float(best_background["cost"] - best_anomaly["cost"])
    raw_maximum = min(declared, key=lambda layer: (-score_by_layer[layer], layer))
    original = raw_maximum if original_localizer_layer is None else int(original_localizer_layer)
    diagnostics.update({
        "raw_profile_maximum": raw_maximum,
        "selected_background": {key: value for key, value in best_background.items() if key != "design"},
        "selected_anomaly": best_anomaly,
    })
    detected = bool(best_anomaly["cost"] < best_background["cost"])
    return {
        "experiment_id": experiment_id,
        "decision_version": RELATIVE_DECISION_VERSION,
        "status": "complete",
        "rome_compatible_detected": detected,
        "is_rome_like": detected,
        "candidate_layer": int(best_anomaly["candidate_layer"]),
        "original_localizer_layer": original,
        "eligible_layers": declared,
        "layer_scores": {str(layer): score_by_layer[layer] for layer in declared},
        "background_cost": float(best_background["cost"]),
        "anomaly_cost": float(best_anomaly["cost"]),
        "gain": gain,
        "diagnostics": diagnostics,
    }


def profile_mdl(
    layer_scores: Mapping[str, float],
    *,
    degree: int,
    log_transform: bool = True,
    include_step_competitor: bool = False,
) -> dict[str, Any]:
    """Compare a smooth score profile with a searched positive excursion."""
    if degree not in (1, 2):
        raise ValueError("profile MDL degree must be 1 or 2")
    ordered = sorted((int(layer), float(score)) for layer, score in layer_scores.items())
    if any(not math.isfinite(score) or score < 0.0 for _, score in ordered):
        raise ValueError("Profile scores must be finite and non-negative")
    n = len(ordered)
    if n <= degree + 2:
        raise ValueError("Profile has too few layers for the requested model")
    values = torch.tensor([score for _, score in ordered], dtype=torch.float64)
    if log_transform:
        values = torch.log1p(values)
    depth = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    columns = [torch.ones_like(depth), depth]
    if degree == 2:
        columns.append(depth.square())
    background = torch.stack(columns, dim=1)
    rss0, _ = _least_squares_rss(values, background)
    resolution = numerical_tolerance(values.dtype, n, max(1.0, float(torch.linalg.vector_norm(values).item()))) ** 2
    if rss0 <= resolution:
        raise ValueError("Profile background fit is numerically degenerate")
    cost0 = n * math.log(rss0 / n) + background.shape[1] * math.log(n)
    candidates: list[tuple[float, int, float]] = []
    for index, (layer, _score) in enumerate(ordered):
        indicator = torch.zeros(n, dtype=torch.float64)
        indicator[index] = 1.0
        design = torch.column_stack((background, indicator))
        rss1, coefficients = _least_squares_rss(values, design)
        if float(coefficients[-1].item()) < 0.0:
            rss1 = rss0
        rss1 = max(rss1, resolution)
        cost1 = n * math.log(rss1 / n) + design.shape[1] * math.log(n) + 2.0 * math.log(n)
        candidates.append((cost1, layer, rss1))
    best_cost, best_layer, best_rss = min(candidates, key=lambda item: (item[0], item[1]))

    non_edit_cost = cost0
    selected_non_edit = "smooth"
    if include_step_competitor:
        step_candidates: list[tuple[float, int]] = []
        for split in range(1, n):
            step = (torch.arange(n) >= split).to(torch.float64)
            design = torch.column_stack((background, step))
            rss_step, _ = _least_squares_rss(values, design)
            rss_step = max(rss_step, resolution)
            cost = n * math.log(rss_step / n) + design.shape[1] * math.log(n) + 2.0 * math.log(n - 1)
            step_candidates.append((cost, split))
        step_cost, split = min(step_candidates, key=lambda item: (item[0], item[1]))
        if step_cost < non_edit_cost:
            non_edit_cost = step_cost
            selected_non_edit = f"step-before-{ordered[split][0]}"

    gain = non_edit_cost - best_cost
    return {
        "is_rome_like": bool(gain > 0.0),
        "gain": float(gain),
        "candidate_layer": int(best_layer),
        "background_degree": degree,
        "transform": "log1p" if log_transform else "raw",
        "non_edit_model": selected_non_edit,
        "rss_background": float(rss0),
        "rss_excursion": float(best_rss),
    }


PROFILE_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "original-v3-relative-b0-v1": {"relative": True},
    "affine-mdl-v1": {"degree": 1, "log_transform": True, "include_step_competitor": False},
    "affine-raw-mdl-v1": {"degree": 1, "log_transform": False, "include_step_competitor": False},
    "quadratic-mdl-v1": {"degree": 2, "log_transform": True, "include_step_competitor": False},
    "quadratic-raw-mdl-v1": {"degree": 2, "log_transform": False, "include_step_competitor": False},
    "affine-step-mdl-v1": {"degree": 1, "log_transform": True, "include_step_competitor": True},
}

LOCAL_PROFILE_EXPERIMENTS: dict[str, dict[str, int]] = {
    "local-linear-diagnostics-v1": {"degree": 1, "exclusion_radius": 0},
    "local-linear-neighborhood-excluded-v1": {"degree": 1, "exclusion_radius": 1},
    "local-quadratic-diagnostics-v1": {"degree": 2, "exclusion_radius": 0},
    "local-quadratic-neighborhood-excluded-v1": {"degree": 2, "exclusion_radius": 1},
}


MATRIX_SCORE_FIELDS: dict[str, str] = {
    "baseline-affine-mdl-v1": "original_score",
    "agreement-affine-mdl-v1": "agreement_score",
    "refined-ratio-affine-mdl-v1": "refined_ratio_score",
    "bounded-affine-mdl-v1": "bounded_contrast_score",
    "quadratic-neighbor-affine-mdl-v1": "quadratic_score",
}

RELATIVE_MATRIX_SCORE_FIELDS: dict[str, str] = {
    "v0-relative-b0-v1": "original_score",
    "v0r-relative-b0-v1": "refined_ratio_score",
    "v1-relative-b0-v1": "centered_directional_score",
    "v2-relative-b0-v1": "standardized_directional_score",
    "lof-relative-b0-v1": "lof_score",
    "decomposition-relative-b0-v1": "decomposition_score",
    "token-alignment-relative-b0-v1": "alignment_score",
}


def local_profile_diagnostics(
    layer_scores: Mapping[str, float],
    *,
    degree: int,
    exclusion_radius: int,
    radius: int = 4,
) -> dict[str, Any]:
    ordered = sorted((int(layer), float(value)) for layer, value in layer_scores.items())
    transformed = {layer: math.log1p(value) for layer, value in ordered}
    candidates: dict[str, Any] = {}
    for position, (layer, _value) in enumerate(ordered):
        selected = [
            other
            for offset, (other, _score) in enumerate(ordered)
            if abs(offset - position) <= radius and abs(offset - position) > exclusion_radius
        ]
        if len(selected) <= degree + 1:
            candidates[str(layer)] = {"status": "unavailable", "reason": "insufficient local background"}
            continue
        offsets = torch.tensor([float(other - layer) for other in selected], dtype=torch.float64)
        columns = [torch.ones_like(offsets), offsets]
        if degree == 2:
            columns.append(offsets.square())
        design = torch.stack(columns, dim=1)
        values = torch.tensor([transformed[other] for other in selected], dtype=torch.float64)
        try:
            rss, coefficients = _least_squares_rss(values, design)
        except ValueError as exc:
            candidates[str(layer)] = {"status": "unavailable", "reason": str(exc)}
            continue
        residual_dof = len(selected) - design.shape[1]
        prediction_error = math.sqrt(rss / residual_dof) if residual_dof > 0 else None
        prediction = float(coefficients[0].item())
        candidates[str(layer)] = {
            "status": "complete",
            "prediction": prediction,
            "signed_excess": transformed[layer] - prediction,
            "background_rmse": prediction_error,
            "background_layers": selected,
        }
    return {
        "status": "diagnostic",
        "degree": degree,
        "exclusion_radius": exclusion_radius,
        "radius": radius,
        "candidates": candidates,
    }


def evaluate_profile_experiments(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    experiments: list[str] | tuple[str, ...],
    eligible_only: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    allowed = None if eligible_only is None else {int(layer) for layer in eligible_only}
    scores = {
        str(int(layer)): float(profile[SCORE_FIELD])
        for layer, profile in profiles.items()
        if allowed is None or int(layer) in allowed
    }
    output: dict[str, Any] = {}
    for identifier in experiments:
        if identifier in PROFILE_EXPERIMENTS:
            settings = PROFILE_EXPERIMENTS[identifier]
            if settings.get("relative"):
                output[identifier] = relative_profile_decision(
                    scores,
                    eligible_layers=sorted(int(layer) for layer in scores),
                    experiment_id=identifier,
                )
            else:
                output[identifier] = profile_mdl(scores, **settings)
        elif identifier in LOCAL_PROFILE_EXPERIMENTS:
            output[identifier] = local_profile_diagnostics(scores, **LOCAL_PROFILE_EXPERIMENTS[identifier])
        else:
            raise ValueError(f"Unknown ROME profile experiment: {identifier}")
    return output


def evaluate_matrix_experiments(
    capture: Mapping[str, Any],
    *,
    experiments: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    profiles = capture.get("profiles", {})
    output: dict[str, Any] = {}
    for identifier in experiments:
        if identifier in RELATIVE_MATRIX_SCORE_FIELDS:
            field = RELATIVE_MATRIX_SCORE_FIELDS[identifier]
            scores = {
                str(int(layer)): float(profile[field])
                for layer, profile in profiles.items()
                if field in profile and profile[field] is not None and math.isfinite(float(profile[field]))
            }
            declared = [int(layer) for layer in capture.get("eligible_layers", sorted(int(value) for value in profiles))]
            raw_scores = {
                int(layer): float(profile.get("original_score", profile.get(field, 0.0)))
                for layer, profile in profiles.items()
                if profile.get("original_score", profile.get(field)) is not None
            }
            original = min(raw_scores, key=lambda layer: (-raw_scores[layer], layer)) if raw_scores else None
            output[identifier] = relative_profile_decision(
                scores,
                eligible_layers=declared,
                experiment_id=identifier,
                original_localizer_layer=original,
            )
            continue
        if identifier == "signed-footprint-mdl-v1":
            gains = {
                str(int(layer)): float(profile["footprint_gain"])
                for layer, profile in profiles.items()
                if "footprint_gain" in profile
            }
            if not gains:
                raise ValueError("Capture does not contain signed footprint measurements")
            selected = min(gains, key=lambda layer: (-gains[layer], int(layer)))
            output[identifier] = {
                "is_rome_like": bool(gains[selected] > 0.0),
                "gain": gains[selected],
                "candidate_layer": int(selected),
                "layer_gains": gains,
            }
            continue
        if identifier not in MATRIX_SCORE_FIELDS:
            raise ValueError(f"Unknown ROME matrix experiment: {identifier}")
        field = MATRIX_SCORE_FIELDS[identifier]
        scores = {
            str(int(layer)): float(profile[field])
            for layer, profile in profiles.items()
            if field in profile and math.isfinite(float(profile[field]))
        }
        decision = profile_mdl(scores, degree=1, log_transform=True)
        ranked = sorted((int(layer) for layer in scores), key=lambda layer: (-scores[str(layer)], layer))
        selected = ranked[0]
        second = scores[str(ranked[1])] if len(ranked) > 1 else 0.0
        localization = {
            "eligible_layers": sorted(ranked),
            "excluded_layers": [],
            "layer_scores": {str(layer): scores[str(layer)] for layer in sorted(ranked)},
            "selected_layer": selected,
            "margin": scores[str(selected)] - second,
        }
        output[identifier] = {**decision, "localization": localization, "score_field": field}
    return output


def _constrained_profile_fit(
    values: torch.Tensor,
    base: torch.Tensor,
    optional: list[torch.Tensor],
) -> tuple[float, torch.Tensor]:
    """Fit nonnegative optional columns by enumerating their small active set."""
    best: tuple[float, torch.Tensor] | None = None
    for mask in range(1 << len(optional)):
        active = [index for index in range(len(optional)) if mask & (1 << index)]
        columns = [base]
        if active:
            columns.append(torch.stack([optional[index] for index in active], dim=1))
        design = torch.column_stack(columns)
        try:
            rss, coefficients = _least_squares_rss(values, design)
        except ValueError:
            continue
        if active and any(float(coefficients[base.shape[1] + offset].item()) < 0.0 for offset in range(len(active))):
            continue
        full = torch.zeros(base.shape[1] + len(optional), dtype=values.dtype)
        full[: base.shape[1]] = coefficients[: base.shape[1]]
        for offset, index in enumerate(active):
            full[base.shape[1] + index] = coefficients[base.shape[1] + offset]
        if best is None or rss < best[0]:
            best = (rss, full)
    if best is None:
        raise ValueError("Constrained profile fit is unavailable")
    return best


def control_profile_mdl(
    target_scores: Mapping[str, float],
    control_scores: Mapping[str, float],
) -> dict[str, Any]:
    shared = sorted(set(int(layer) for layer in target_scores) & set(int(layer) for layer in control_scores))
    n = len(shared)
    if n <= 4:
        raise ValueError("Control profile has too few shared layers")
    target = torch.log1p(torch.tensor([float(target_scores[str(layer)]) for layer in shared], dtype=torch.float64))
    control = torch.log1p(torch.tensor([float(control_scores[str(layer)]) for layer in shared], dtype=torch.float64))
    depth = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    base = torch.stack((torch.ones_like(depth), depth), dim=1)
    rss0, coefficients0 = _constrained_profile_fit(target, base, [control])
    resolution = numerical_tolerance(target.dtype, n, max(1.0, float(torch.linalg.vector_norm(target).item()))) ** 2
    if rss0 <= resolution:
        raise ValueError("Control profile background fit is numerically degenerate")
    cost0 = n * math.log(rss0 / n) + 3.0 * math.log(n)
    candidates: list[tuple[float, int, float, torch.Tensor]] = []
    for index, layer in enumerate(shared):
        excursion = torch.zeros(n, dtype=torch.float64)
        excursion[index] = 1.0
        rss1, coefficients = _constrained_profile_fit(target, base, [control, excursion])
        rss1 = max(rss1, resolution)
        cost1 = n * math.log(rss1 / n) + 4.0 * math.log(n) + 2.0 * math.log(n)
        candidates.append((cost1, layer, rss1, coefficients))
    best_cost, layer, rss1, coefficients = min(candidates, key=lambda item: (item[0], item[1]))
    gain = cost0 - best_cost
    return {
        "is_rome_like": bool(gain > 0.0),
        "gain": float(gain),
        "candidate_layer": int(layer),
        "control_coefficient": float(coefficients[2].item()),
        "excursion_amplitude": float(coefficients[3].item()),
        "background_control_coefficient": float(coefficients0[2].item()),
        "shared_layers": shared,
        "rss_background": float(rss0),
        "rss_excursion": float(rss1),
    }


def _fit_signed_template(
    matrices: torch.Tensor,
    template: torch.Tensor,
    *,
    rank: int,
) -> float:
    """Return Frobenius RSS for affine background plus a signed low-rank template."""
    count = matrices.shape[0]
    depth = torch.linspace(-1.0, 1.0, count, dtype=matrices.dtype, device=matrices.device)
    background = torch.stack((torch.ones_like(depth), depth), dim=1)
    projection = background @ torch.linalg.pinv(background)
    residualizer = torch.eye(count, dtype=matrices.dtype, device=matrices.device) - projection
    residualized_template = residualizer @ template
    denominator = torch.dot(residualized_template, residualized_template)
    tolerance = numerical_tolerance(matrices.dtype, count, float(torch.linalg.vector_norm(template).item()))
    if float(denominator.item()) <= tolerance:
        raise ValueError("Signed footprint template is not identifiable")
    residualized = torch.einsum("st,tij->sij", residualizer, matrices)
    innovation = torch.einsum("t,tij->ij", residualized_template, residualized) / denominator
    innovation = (innovation + innovation.T) * 0.5
    if innovation.shape[0] <= 64:
        eigenvalues, eigenvectors = torch.linalg.eigh(innovation)
    else:
        subspace, _singular_values = _top_residual_subspace(innovation, layer=7919 + rank)
        small = subspace.T @ innovation @ subspace
        small = (small + small.T) * 0.5
        eigenvalues, rotation = torch.linalg.eigh(small)
        eigenvectors = subspace @ rotation
    order = torch.argsort(torch.abs(eigenvalues), descending=True)[:rank]
    approximation = (eigenvectors[:, order] * eigenvalues[order]) @ eigenvectors[:, order].T
    design = torch.column_stack((background, template))
    nuisance = matrices - template[:, None, None] * approximation
    coefficients = torch.einsum("pt,tij->pij", torch.linalg.pinv(background), nuisance)
    fitted = torch.einsum("tp,pij->tij", background, coefficients)
    error = matrices - fitted - template[:, None, None] * approximation
    return float(error.square().sum().item())


def _footprint_measurement(residuals: list[torch.Tensor], *, candidate_count: int) -> dict[str, Any]:
    if len(residuals) != 5:
        raise ValueError("Signed footprint requires five residual matrices")
    matrices = torch.stack(residuals)
    count, dimension, _ = matrices.shape
    depth = torch.linspace(-1.0, 1.0, count, dtype=matrices.dtype, device=matrices.device)
    background = torch.stack((torch.ones_like(depth), depth), dim=1)
    coefficients = torch.einsum("pt,tij->pij", torch.linalg.pinv(background), matrices)
    base_error = matrices - torch.einsum("tp,pij->tij", background, coefficients)
    rss0 = float(base_error.square().sum().item())
    q = dimension * (dimension + 1) // 2
    observations = count * q
    resolution = numerical_tolerance(matrices.dtype, dimension, max(1.0, math.sqrt(rss0))) ** 2
    edit_template = torch.tensor([0.0, -0.5, 1.0, -0.5, 0.0], dtype=matrices.dtype, device=matrices.device)

    edit_models: list[tuple[float, int, float]] = []
    step_models: list[tuple[float, int, int, float]] = []
    for rank in (1, 2):
        rss = max(_fit_signed_template(matrices, edit_template, rank=rank), resolution)
        parameters = rank * dimension - rank * (rank - 1) // 2
        edit_models.append((observations * math.log(rss / observations) + parameters * math.log(observations), rank, rss))

        for transition in range(1, 7):
            gram_step = torch.tensor(
                [float(index >= transition) for index in range(7)],
                dtype=matrices.dtype,
                device=matrices.device,
            )
            template = gram_step[1:6] - 0.5 * (gram_step[0:5] + gram_step[2:7])
            if float(torch.linalg.vector_norm(template).item()) == 0.0:
                continue
            try:
                step_rss = max(_fit_signed_template(matrices, template, rank=rank), resolution)
            except ValueError:
                continue
            parameters = rank * dimension - rank * (rank - 1) // 2
            cost = observations * math.log(step_rss / observations) + parameters * math.log(observations)
            step_models.append((cost, rank, transition, step_rss))

    search_edit = max(1, int(candidate_count) * len(edit_models))
    search_step = max(1, int(candidate_count) * len(step_models))
    edit_cost, edit_rank, edit_rss = min(
        ((cost + 2.0 * math.log(search_edit), rank, rss) for cost, rank, rss in edit_models),
        key=lambda item: (item[0], item[1]),
    )
    base_cost = observations * math.log(max(rss0, resolution) / observations)
    non_edit = (base_cost, "smooth", rss0)
    if step_models:
        step_cost, step_rank, transition, step_rss = min(
            ((cost + 2.0 * math.log(search_step), rank, transition, rss) for cost, rank, transition, rss in step_models),
            key=lambda item: (item[0], item[1], item[2]),
        )
        if step_cost < non_edit[0]:
            non_edit = (step_cost, f"step-r{step_rank}-at-{transition}", step_rss)
    return {
        "footprint_gain": float(non_edit[0] - edit_cost),
        "footprint_rank": int(edit_rank),
        "footprint_non_edit_model": non_edit[1],
        "footprint_rss_edit": float(edit_rss),
        "footprint_rss_non_edit": float(non_edit[2]),
        "footprint_observations": int(observations),
        "footprint_solver": "exact" if dimension <= 64 else "randomized-subspace-refined",
    }


def capture_experiment_weights(
    weights: Mapping[int, torch.Tensor],
    *,
    groups: tuple[str, ...] = DEFAULT_EXPERIMENT_GROUPS,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> dict[str, Any]:
    """Capture opt-in ROME research measurements from one checkpoint."""
    resolved_groups = tuple(dict.fromkeys(str(group) for group in groups))
    unknown = sorted(set(resolved_groups) - set(EXPERIMENT_GROUPS))
    if unknown:
        raise ValueError(f"Unknown ROME experiment groups: {', '.join(unknown)}")
    layers = sorted(int(layer) for layer in weights)
    eligible = eligible_layers(layers, trim_fraction=trim_fraction)
    eligible_set = set(eligible)
    positions = {layer: index for index, layer in enumerate(layers)}
    footprint_indices = [
        index for index in range(3, len(layers) - 3) if layers[index] in eligible_set
    ]
    radius = 3 if "footprint" in resolved_groups else 2 if "quadratic" in resolved_groups else 1
    grams: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, Any]] = {}

    for layer in eligible:
        index = positions[layer]
        start = max(0, index - radius)
        stop = min(len(layers), index + radius + 1)
        for other in layers[start:stop]:
            if other not in grams:
                grams[other] = hidden_gram(weights[other])
        profile: dict[str, Any] = {}
        if "neighbors" in resolved_groups:
            profile.update(neighbor_experiment_measurements(
                grams[layers[index - 1]], grams[layer], grams[layers[index + 1]], layer=layer
            ))
        if "quadratic" in resolved_groups and 1 < index < len(layers) - 2:
            profile.update(quadratic_experiment_measurements(
                grams[layers[index - 2]],
                grams[layers[index - 1]],
                grams[layer],
                grams[layers[index + 1]],
                grams[layers[index + 2]],
                layer=layer,
            ))
        if "footprint" in resolved_groups and index in footprint_indices:
            residuals = [
                grams[layers[other]] - 0.5 * (grams[layers[other - 1]] + grams[layers[other + 1]])
                for other in range(index - 2, index + 3)
            ]
            profile.update(_footprint_measurement(
                residuals,
                candidate_count=len(footprint_indices),
            ))
        profiles[str(layer)] = profile
        minimum_needed_position = index - radius + 1
        grams = {
            cached_layer: gram
            for cached_layer, gram in grams.items()
            if positions[cached_layer] >= minimum_needed_position
        }

    if "quadratic" in resolved_groups:
        quadratic_layers = [
            int(layer) for layer, profile in profiles.items()
            if "quadratic_residual_frobenius" in profile
        ]
        for layer in quadratic_layers:
            index = positions[layer]
            background = [
                float(profiles[str(other)]["quadratic_residual_frobenius"])
                for other in quadratic_layers
                if abs(positions[other] - index) > 2
            ]
            profile = profiles[str(layer)]
            profile["quadratic_background_error_count"] = len(background)
            if background:
                rms = math.sqrt(sum(value * value for value in background) / len(background))
                profile["quadratic_background_error_rms"] = rms
                tolerance = float(profile["quadratic_tolerance"])
                profile["quadratic_residual_to_background"] = (
                    float(profile["quadratic_residual_frobenius"]) / rms
                    if rms > tolerance
                    else None
                )
            else:
                profile["quadratic_background_error_rms"] = None
                profile["quadratic_residual_to_background"] = None

    return {
        "mode": "single_checkpoint",
        "capture_version": EXPERIMENT_CAPTURE_VERSION,
        "groups": list(resolved_groups),
        "layers": layers,
        "trim_fraction": float(trim_fraction),
        "eligible_layers": eligible,
        "excluded_layers": [layer for layer in layers if layer not in eligible_set],
        "profiles": profiles,
    }


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
    "DEFAULT_EXPERIMENT_GROUPS",
    "DEFAULT_TRIM_FRACTION",
    "DIRECTIONAL_CAPTURE_VERSION",
    "CROSS_LAYER_CAPTURE_VERSION",
    "EXPERIMENT_CAPTURE_VERSION",
    "EXPERIMENT_GROUPS",
    "MATRIX_SCORE_FIELDS",
    "PROFILE_FIELDS",
    "PROFILE_EXPERIMENTS",
    "RELATIVE_DECISION_VERSION",
    "RELATIVE_MATRIX_SCORE_FIELDS",
    "RomeLayerLocalizer",
    "SCORE_FIELD",
    "capture_experiment_weights",
    "capture_directional_error_weights",
    "control_profile_mdl",
    "detect_from_profiles",
    "eligible_layers",
    "evaluate_matrix_experiments",
    "evaluate_cross_layer_experiments",
    "factor_cross_layer_kernel",
    "evaluate_profile_experiments",
    "hidden_gram",
    "local_profile_diagnostics",
    "localize_scores",
    "lof_scores_from_kernel",
    "neighbor_experiment_measurements",
    "numerical_tolerance",
    "profile_mdl",
    "relative_profile_decision",
    "robust_decomposition_scores",
    "profile_weights",
    "quadratic_experiment_measurements",
    "score_layer",
]

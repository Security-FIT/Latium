"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from src.common.io import to_serializable
from src.structural.capture.matrix_features import resolve_matrix_features
from src.structural.detectors.bottom_rank_svd import BottomRankSVDDetector
from src.structural.detectors.matrix_anomaly import (
    condition_number,
    gram_coherence,
    row_entropy,
    row_norm_gini,
    schatten_ratio,
    stable_effective_ratio,
)
from src.structural.detectors.profiles import matrix_profile
from src.common.linalg import gpu_svd_topk
from src.structural.detectors.spectral_primitives import (
    canonical_orient,
    pcs_pairwise_rank_cumsums,
    spectral_decomposition,
)


EPS = 1e-10


@dataclass
class CaptureContext:
    proj_weights: dict[int, torch.Tensor]
    fc_weights: Optional[dict[int, torch.Tensor]]
    attention_weights: dict[str, dict[int, torch.Tensor]]
    probe_vector: Optional[torch.Tensor]
    token_predictor: Optional[Callable[[torch.Tensor], tuple[int, str]]]
    changed_weights: dict[str, tuple[int, ...] | None]
    options: dict[str, Any]
    baseline_proj_weights: Optional[dict[int, torch.Tensor]] = None
    baseline_fc_weights: Optional[dict[int, torch.Tensor]] = None

    @property
    def is_baseline(self) -> bool:
        return not self.changed_weights

    def changed_layers(self, family: str, available: list[int]) -> list[int]:
        if self.is_baseline:
            return list(available)
        if family not in self.changed_weights:
            return []
        changed = self.changed_weights[family]
        return list(available) if changed is None else [int(layer) for layer in changed if int(layer) in available]


def _decomposition(
    weights: dict[int, torch.Tensor],
    top_k: int,
) -> dict[str, Any]:
    layers, singular_values, right_vectors, left_vectors = spectral_decomposition(weights, max_k=top_k)
    return {
        "layers": layers,
        "singular_values": singular_values,
        "right_vectors": right_vectors,
        "left_vectors": left_vectors,
    }


def _pairwise_rows(
    right_vectors: np.ndarray,
    singular_values: np.ndarray,
    changed_indices: list[int],
) -> dict[str, Any]:
    dot, flip, weight = pcs_pairwise_rank_cumsums(
        right_vectors,
        singular_values,
        singular_values.shape[1] if singular_values.ndim == 2 else 0,
    )
    return {
        "dot_weight_cumsum": {str(index): dot[:, index, :].tolist() for index in changed_indices},
        "flip_weight_cumsum": {str(index): flip[:, index, :].tolist() for index in changed_indices},
        "weight_cumsum": {str(index): weight[:, index, :].tolist() for index in changed_indices},
    }


def _cross_rank_cumsums(
    proj_left_vectors: np.ndarray,
    proj_singular_values: np.ndarray,
    fc_right_vectors: np.ndarray,
    fc_singular_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        proj_left_vectors.ndim != 3
        or fc_right_vectors.ndim != 3
        or proj_left_vectors.shape[0] != fc_right_vectors.shape[0]
        or proj_left_vectors.shape[2] != fc_right_vectors.shape[2]
    ):
        return np.empty((0, 0)), np.empty((0, 0))
    rank = min(
        proj_left_vectors.shape[1],
        fc_right_vectors.shape[1],
        proj_singular_values.shape[1],
        fc_singular_values.shape[1],
    )
    if rank <= 0:
        return np.empty((0, 0)), np.empty((0, 0))
    dots: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for index in range(proj_left_vectors.shape[0]):
        proj = canonical_orient(proj_left_vectors[index, :rank])
        fc = canonical_orient(fc_right_vectors[index, :rank])
        weight = 0.5 * (proj_singular_values[index, :rank] + fc_singular_values[index, :rank])
        dots.append(np.cumsum(weight * np.sum(proj * fc, axis=1)))
        weights.append(np.cumsum(weight))
    return np.stack(dots), np.stack(weights)


def capture_spectral(context: CaptureContext) -> dict[str, Any]:
    top_k = int(context.options.get("spectral_top_k", 50))
    proj = _decomposition(context.proj_weights, top_k)
    layers = list(proj["layers"])
    changed_layers = context.changed_layers("proj", layers)
    changed_indices = [layers.index(layer) for layer in changed_layers]

    if context.is_baseline:
        dot, flip, weight = pcs_pairwise_rank_cumsums(
            proj["right_vectors"],
            proj["singular_values"],
            top_k,
        )
        payload: dict[str, Any] = {
            "mode": "baseline",
            "layers": layers,
            "stored_top_k": int(proj["singular_values"].shape[1]),
            "sv_proj_topk": {str(layer): proj["singular_values"][index].tolist() for index, layer in enumerate(layers)},
            "pcs_pairwise_dot_weight_cumsum": dot.tolist(),
            "pcs_flip_pairwise_weight_cumsum": flip.tolist(),
            "pcs_pairwise_weight_cumsum": weight.tolist(),
        }
    else:
        payload = {
            "mode": "patch",
            "layers": layers,
            "changed_layers": {"proj": changed_layers},
            "stored_top_k": int(proj["singular_values"].shape[1]),
            "sv_proj_topk": {
                str(layer): proj["singular_values"][layers.index(layer)].tolist() for layer in changed_layers
            },
            "pcs_pairwise_rows": _pairwise_rows(
                proj["right_vectors"],
                proj["singular_values"],
                changed_indices,
            ),
        }

    if context.fc_weights:
        fc = _decomposition(context.fc_weights, top_k)
        fc_layers = list(fc["layers"])
        changed_fc = context.changed_layers("fc", fc_layers)
        included_fc = fc_layers if context.is_baseline else changed_fc
        payload["changed_layers"] = {
            **dict(payload.get("changed_layers") or {}),
            "fc": changed_fc,
        }
        payload["sv_fc_topk"] = {
            str(layer): fc["singular_values"][fc_layers.index(layer)].tolist() for layer in included_fc
        }
        if fc_layers == layers:
            cross_dot, cross_weight = _cross_rank_cumsums(
                proj["left_vectors"],
                proj["singular_values"],
                fc["right_vectors"],
                fc["singular_values"],
            )
            if cross_dot.size and cross_weight.size:
                included_cross = layers if context.is_baseline else sorted(set(changed_layers).union(changed_fc))
                payload["pcs_cross_dot_weight_cumsum"] = {
                    str(layer): cross_dot[layers.index(layer)].tolist() for layer in included_cross
                }
                payload["pcs_cross_weight_cumsum"] = {
                    str(layer): cross_weight[layers.index(layer)].tolist() for layer in included_cross
                }
    return to_serializable(payload)


def capture_matrix_features(context: CaptureContext) -> dict[str, Any]:
    from src.structural.detectors.blind_resident import BlindMSDDetector

    feature_set = str(context.options.get("matrix_feature_set", "paper"))
    features = resolve_matrix_features(feature_set, context.options.get("matrix_features", ()))
    top_k = int(context.options.get("matrix_svd_top_k", 50))
    families: dict[str, dict[int, torch.Tensor]] = {"proj": context.proj_weights}
    if context.fc_weights:
        families["fc"] = context.fc_weights
    output: dict[str, Any] = {
        "mode": "baseline" if context.is_baseline else "patch",
        "feature_set": feature_set,
        "features": list(features),
        "stored_top_k": int(top_k),
        "families": {},
        "changed_layers": {},
    }
    for family, weights in families.items():
        layers = sorted(weights)
        included = context.changed_layers(family, layers)
        output["changed_layers"][family] = included
        blind_features = BlindMSDDetector().compute_layer_features(
            weights,
            top_k=top_k,
            features=features,
            niter=2,
        )
        output["families"][family] = {str(layer): blind_features.get(layer, {}) for layer in layers}
    return to_serializable(output)


def _deterministic_topk_svd(
    weight: torch.Tensor,
    *,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a reproducible randomized SVD for numeric feature extraction."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        return gpu_svd_topk(
            weight,
            k=max(2, int(top_k)),
            niter=4,
        )


def _hidden_spectral_density(weight: torch.Tensor) -> torch.Tensor:
    """Return a trace-one Gram matrix in the projection's shared hidden space."""
    device = weight.device if weight.is_cuda else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matrix = weight.detach().to(device=device, dtype=torch.float32)
    frobenius_sq = matrix.square().sum()
    if not bool(torch.isfinite(frobenius_sq).item()) or float(frobenius_sq.item()) <= 0.0:
        raise ValueError("Projection weight must contain finite, non-zero values")
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    return gram / frobenius_sq


def _weighted_spectrum_profile(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    layer: int,
    neighbors: tuple[torch.Tensor, ...] = (),
) -> dict[str, float]:
    """Measure scale-free, low-rank curvature of a hidden-space weighted spectrum."""
    residual = current - reference
    left, singular_tensor, _right = _deterministic_topk_svd(
        residual,
        top_k=2,
        seed=433494437 + int(layer),
    )
    singular = singular_tensor.detach().double().cpu().numpy()
    basis = left[:, :2].to(device=reference.device, dtype=reference.dtype)
    leading = basis[:, 0]
    signed_shift = torch.linalg.multi_dot((leading, residual, leading))
    background_energy = torch.linalg.multi_dot((leading, reference, leading)).clamp_min(EPS)
    relative_shift = signed_shift / background_energy
    residual_subspace = basis.T @ residual @ basis
    reference_subspace = basis.T @ reference @ basis
    reference_eigenvalues, reference_eigenvectors = torch.linalg.eigh(reference_subspace)
    inverse_sqrt = (
        reference_eigenvectors
        @ torch.diag(reference_eigenvalues.clamp_min(EPS).rsqrt())
        @ reference_eigenvectors.T
    )
    relative_subspace = inverse_sqrt @ residual_subspace @ inverse_sqrt
    relative_eigenvalues = torch.linalg.eigvalsh(relative_subspace)
    relative_squared = relative_eigenvalues.square()
    frobenius = float(torch.linalg.vector_norm(residual).item())
    current_norm = torch.linalg.vector_norm(current)
    reference_norm = torch.linalg.vector_norm(reference)
    alignment = float(
        (current * reference).sum().div(current_norm * reference_norm + EPS).item()
    )
    if len(neighbors) == 2:
        left_jump = current - neighbors[0]
        right_jump = current - neighbors[1]
        left_energy = left_jump.square().sum()
        right_energy = right_jump.square().sum()
        jump_energy = left_energy + right_energy
        curvature_energy = (left_jump + right_jump).square().sum()
        bilateral_coherence = curvature_energy / (2.0 * jump_energy + EPS)
        bilateral_alignment = (left_jump * right_jump).sum() / (
            torch.sqrt(left_energy * right_energy) + EPS
        )
        bilateral_frobenius = torch.sqrt(torch.sqrt(left_energy * right_energy))
        bilateral_balance = 2.0 * torch.sqrt(left_energy * right_energy) / (jump_energy + EPS)
    else:
        bilateral_coherence = torch.zeros((), device=current.device)
        bilateral_alignment = torch.zeros((), device=current.device)
        bilateral_frobenius = torch.linalg.vector_norm(residual)
        bilateral_balance = torch.zeros((), device=current.device)
    squared = singular**2
    return {
        "operator_norm": float(singular[0]),
        "frobenius_norm": frobenius,
        "rank1_energy": float(squared[0] / (frobenius**2 + EPS)),
        "rank2_energy": float(squared[:2].sum() / (frobenius**2 + EPS)),
        "neighbor_cka_distance": float(1.0 - alignment),
        "directional_background": float(current.shape[0] * background_energy.item()),
        "relative_operator_norm": float(abs(relative_shift.item())),
        "signed_relative_shift": float(relative_shift.item()),
        "relative_subspace_operator_norm": float(relative_eigenvalues.abs().max().item()),
        "relative_subspace_frobenius": float(torch.sqrt(relative_squared.sum()).item()),
        "relative_subspace_rank1_energy": float(
            relative_squared.max().div(relative_squared.sum() + EPS).item()
        ),
        "bilateral_coherence": float(bilateral_coherence.item()),
        "bilateral_alignment": float(bilateral_alignment.item()),
        "bilateral_frobenius": float(bilateral_frobenius.item()),
        "bilateral_balance": float(bilateral_balance.item()),
    }


def capture_weighted_spectrum(context: CaptureContext) -> dict[str, Any]:
    """Capture normalized hidden-space spectral curvature around every changed layer."""
    layers = sorted(context.proj_weights)
    direct = context.changed_layers("proj", layers)
    if context.is_baseline:
        included = layers
    else:
        positions = {layer: index for index, layer in enumerate(layers)}
        affected: set[int] = set()
        for layer in direct:
            index = positions[layer]
            affected.update(layers[max(0, index - 1) : min(len(layers), index + 2)])
        included = sorted(affected)

    positions = {layer: index for index, layer in enumerate(layers)}
    # A hidden Gram is quadratic in hidden width.  Keep only the rolling
    # three-layer neighborhood instead of retaining one Gram per model layer.
    densities: dict[int, torch.Tensor] = {}
    profiles: dict[str, dict[str, float]] = {}
    for layer in included:
        index = positions[layer]
        neighborhood = layers[max(0, index - 1) : min(len(layers), index + 2)]
        for other in neighborhood:
            if other not in densities:
                densities[other] = _hidden_spectral_density(context.proj_weights[other])
        neighbors = [
            densities[other]
            for other in neighborhood
            if other != layer
        ]
        if not neighbors:
            continue
        reference = torch.stack(neighbors).mean(dim=0)
        profiles[str(layer)] = _weighted_spectrum_profile(
            densities[layer],
            reference,
            layer=layer,
            neighbors=tuple(neighbors),
        )
        densities = {
            cached_layer: density
            for cached_layer, density in densities.items()
            if positions[cached_layer] >= index
        }

    first_weight = context.proj_weights[layers[0]] if layers else None
    return to_serializable(
        {
            "mode": "baseline" if context.is_baseline else "patch",
            "layers": layers,
            "weight_shape": list(first_weight.shape) if first_weight is not None else [],
            "profiles": profiles,
            "changed_layers": {"proj": included},
        }
    )


def _update_profile(
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    layer: int,
) -> dict[str, float | bool]:
    """Describe a checkpoint delta relative to floating-point roundoff."""
    if current.shape != baseline.shape:
        raise ValueError(
            f"Layer {layer} shape changed from {tuple(baseline.shape)} to {tuple(current.shape)}"
        )
    original = baseline.detach().to(dtype=torch.float32, device="cpu")
    modified = current.detach().to(dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(modified).all()):
        raise ValueError(f"Layer {layer} contains non-finite weights")
    delta = modified - original
    delta_frobenius = float(torch.linalg.vector_norm(delta).item())
    dtype = baseline.dtype if baseline.dtype.is_floating_point else torch.float32
    epsilon = max(float(torch.finfo(dtype).eps), float(torch.finfo(torch.float32).eps))
    # Higham's gamma bound for four rounded operations: outer-product entry,
    # weight addition, checkpoint subtraction, and reconstruction.  This is
    # numerical error accounting derived from dtype precision, not a fitted
    # edit threshold.
    operation_count = 4
    gamma = operation_count * epsilon / max(1.0 - operation_count * epsilon, epsilon)
    roundoff_bound = float(
        gamma
        * (
            torch.linalg.vector_norm(original).item()
            + torch.linalg.vector_norm(modified).item()
            + delta_frobenius
        )
    )
    baseline_frobenius = float(torch.linalg.vector_norm(original).item())
    if delta_frobenius <= roundoff_bound:
        return {
            "delta_frobenius": delta_frobenius,
            "relative_delta_frobenius": delta_frobenius / max(baseline_frobenius, EPS),
            "top1_singular": 0.0,
            "top1_energy": 0.0,
            "rank1_residual": 0.0,
            "rank1_residual_ratio": 0.0,
            "roundoff_bound": roundoff_bound,
            "rank_one_within_roundoff": False,
            "detectable_change": False,
        }

    left, singular, right = _deterministic_topk_svd(
        delta,
        top_k=2,
        seed=701408733 + int(layer),
    )
    top1 = float(singular[0].item())
    # Accumulate the best-rank-one residual in row blocks so wide production
    # matrices do not require another full float64 matrix.
    rank1_residual_squared = 0.0
    right_vector = right[0].double().unsqueeze(0)
    for start in range(0, int(delta.shape[0]), 256):
        stop = min(int(delta.shape[0]), start + 256)
        reconstruction = top1 * left[start:stop, 0].double().unsqueeze(1) * right_vector
        block = delta[start:stop].double() - reconstruction
        rank1_residual_squared += float(block.square().sum().item())
    rank1_residual = rank1_residual_squared**0.5
    return {
        "delta_frobenius": delta_frobenius,
        "relative_delta_frobenius": delta_frobenius / max(baseline_frobenius, EPS),
        "top1_singular": top1,
        "top1_energy": min(1.0, top1**2 / max(delta_frobenius**2, EPS)),
        "rank1_residual": rank1_residual,
        "rank1_residual_ratio": rank1_residual / max(delta_frobenius, EPS),
        "roundoff_bound": roundoff_bound,
        "rank_one_within_roundoff": bool(rank1_residual <= roundoff_bound),
        "detectable_change": True,
    }


def _update_family_profiles(
    current: Optional[dict[int, torch.Tensor]],
    baseline: Optional[dict[int, torch.Tensor]],
) -> dict[str, dict[str, float | bool]]:
    if current is None and baseline is None:
        return {}
    if current is None or baseline is None:
        raise ValueError("Current and baseline matrix families must both be available")
    if set(current) != set(baseline):
        raise ValueError("Current and baseline matrix families contain different layers")
    profiles: dict[str, dict[str, float | bool]] = {}
    for layer in sorted(current):
        current_weight = current[layer]
        baseline_weight = baseline[layer]
        if current_weight is baseline_weight:
            continue
        directly_equal = (
            current_weight.shape == baseline_weight.shape
            and current_weight.dtype == baseline_weight.dtype
            and current_weight.device == baseline_weight.device
            and torch.equal(current_weight, baseline_weight)
        )
        if directly_equal:
            continue
        profile = _update_profile(current_weight, baseline_weight, layer=layer)
        if profile["detectable_change"]:
            profiles[str(layer)] = profile
    family_energy = sum(float(profile["delta_frobenius"]) ** 2 for profile in profiles.values())
    for profile in profiles.values():
        profile["family_energy_fraction"] = (
            float(profile["delta_frobenius"]) ** 2 / family_energy if family_energy > 0.0 else 0.0
        )
    return profiles


def capture_rome_update(context: CaptureContext) -> dict[str, Any]:
    """Capture clean-to-suspect update fingerprints for ROME-like attribution."""
    if context.is_baseline:
        return {
            "mode": "baseline",
            "families": {"proj": {}, "fc": {}},
        }
    if context.baseline_proj_weights is None:
        raise RuntimeError("ROME update capture requires clean baseline projection weights")
    return to_serializable(
        {
            "mode": "patch",
            "families": {
                "proj": _update_family_profiles(context.proj_weights, context.baseline_proj_weights),
                "fc": _update_family_profiles(context.fc_weights, context.baseline_fc_weights),
            },
        }
    )


def capture_attention_features(context: CaptureContext) -> dict[str, Any]:
    output: dict[str, Any] = {
        "mode": "baseline" if context.is_baseline else "patch",
        "families": {},
    }
    for family, weights in context.attention_weights.items():
        layers = sorted(weights)
        if context.is_baseline:
            included = layers
        else:
            included = context.changed_layers("attention", layers)
        output["families"][family] = {str(layer): matrix_profile(weights[layer]) for layer in included}
    return to_serializable(output)


def _matrix_anomaly_profile(weight: torch.Tensor) -> dict[str, float]:
    return {
        "row_entropy": float(row_entropy(weight)),
        "schatten_ratio": float(schatten_ratio(weight)),
        "condition_number": float(condition_number(weight)),
        "stable_effective_ratio": float(stable_effective_ratio(weight)),
        "gram_coherence": float(gram_coherence(weight)),
        "row_norm_gini": float(row_norm_gini(weight)),
    }


def capture_matrix_anomaly_features(context: CaptureContext) -> dict[str, Any]:
    families: dict[str, dict[int, torch.Tensor]] = {"proj": context.proj_weights}
    if context.fc_weights:
        families["fc"] = context.fc_weights
    output: dict[str, Any] = {
        "mode": "baseline" if context.is_baseline else "patch",
        "families": {},
    }
    for family, weights in families.items():
        layers = sorted(weights)
        included = context.changed_layers(family, layers)
        output["families"][family] = {str(layer): _matrix_anomaly_profile(weights[layer]) for layer in included}
    return to_serializable(output)


def token_predictor_from_handler(handler: Any) -> Callable[[torch.Tensor], tuple[int, str]]:
    getter = getattr(handler.model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(handler.model, "lm_head", None)
    if head is None:
        raise RuntimeError("No output embedding head is available")
    parameter = next(head.parameters())

    def predict(hidden_vector: torch.Tensor) -> tuple[int, str]:
        with torch.no_grad():
            hidden = hidden_vector.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            logits = head(hidden.unsqueeze(0).unsqueeze(0))
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[0, -1, :] if logits.ndim == 3 else logits.reshape(-1)
            token_id = int(torch.argmax(next_logits).item())
        return token_id, handler.tokenizer.decode([token_id])

    return predict


def capture_bottom_rank_tokens(context: CaptureContext) -> dict[str, Any]:
    if context.probe_vector is None:
        raise RuntimeError("Edit method did not provide a probe vector")
    if context.token_predictor is None:
        raise RuntimeError("Bottom-rank token capture requires a token predictor")
    detector = BottomRankSVDDetector(
        sweep_ranks=tuple(context.options.get("bottom_rank_sweep_ranks", (4, 8, 16, 32))),
        top_svd_rank=int(context.options.get("bottom_rank_top_svd_rank", 64)),
        boundary=int(context.options.get("bottom_rank_boundary", 2)),
    )
    result = detector.detect(
        context.proj_weights,
        probe_vector=context.probe_vector,
        token_predictor=context.token_predictor,
    )
    return {
        "token_id_sweeps": result.get("token_id_sweeps", {}),
        "token_sweeps": result.get("token_sweeps", {}),
        "used_sweep_ranks": result.get("used_sweep_ranks", {}),
        "skipped_layers": result.get("skipped_layers", {}),
        "config": result.get("config", {}),
    }

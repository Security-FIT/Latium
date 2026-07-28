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
from src.structural.detectors.rome_presence import detect_rome_compatible_edit
from src.structural.detectors.weighted_spectrum import (
    DEFAULT_TRIM_FRACTION,
    PROFILE_FIELDS,
    SCHEMA_VERSION,
    eligible_layers,
    hidden_gram,
)
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


def _pairwise_patch_rows(
    local_layers: list[int],
    right_vectors: np.ndarray,
    singular_values: np.ndarray,
    all_layers: list[int],
    changed_layers: list[int],
    neighbor_layers: int,
) -> dict[str, Any]:
    if not changed_layers:
        return {"dot_weight_cumsum": {}, "weight_cumsum": {}}
    rank = singular_values.shape[1] if singular_values.ndim == 2 else 0
    local_positions = {layer: index for index, layer in enumerate(local_layers)}
    global_positions = {layer: index for index, layer in enumerate(all_layers)}
    vectors = np.stack([canonical_orient(rows[:rank]) for rows in right_vectors])
    dot_rows: dict[str, list[list[float]]] = {}
    weight_rows: dict[str, list[list[float]]] = {}
    for layer in changed_layers:
        local_index = local_positions[layer]
        global_index = global_positions[layer]
        dot = np.zeros((rank, len(all_layers)), dtype=np.float64)
        weight = np.zeros((rank, len(all_layers)), dtype=np.float64)
        self_weight = np.cumsum(singular_values[local_index, :rank])
        dot[:, global_index] = self_weight
        weight[:, global_index] = self_weight
        start = max(0, global_index - neighbor_layers)
        stop = min(len(all_layers), global_index + neighbor_layers + 1)
        for other in all_layers[start:stop]:
            if other == layer:
                continue
            other_local_index = local_positions[other]
            other_global_index = global_positions[other]
            pair_weight = 0.5 * (singular_values[local_index, :rank] + singular_values[other_local_index, :rank])
            pair_dot = np.sum(
                vectors[local_index, :rank] * vectors[other_local_index, :rank],
                axis=1,
            )
            weight[:, other_global_index] = np.cumsum(pair_weight)
            dot[:, other_global_index] = np.cumsum(pair_weight * pair_dot)
        dot_rows[str(global_index)] = dot.tolist()
        weight_rows[str(global_index)] = weight.tolist()
    return {
        "dot_weight_cumsum": dot_rows,
        "weight_cumsum": weight_rows,
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
    neighbor_layers = max(1, int(context.options.get("spectral_neighbor_layers", 1)))
    layers = sorted(context.proj_weights)
    changed_layers = context.changed_layers("proj", layers)
    fc_layers = sorted(context.fc_weights) if context.fc_weights else []
    changed_fc = context.changed_layers("fc", fc_layers)
    cross_compatible = bool(fc_layers) and fc_layers == layers

    if context.is_baseline:
        proj = _decomposition(context.proj_weights, top_k)
        dot, _flip, weight = pcs_pairwise_rank_cumsums(
            proj["right_vectors"],
            proj["singular_values"],
            top_k,
            include_flip=False,
            max_layer_distance=neighbor_layers,
        )
        payload: dict[str, Any] = {
            "mode": "baseline",
            "layers": layers,
            "stored_top_k": int(proj["singular_values"].shape[1]),
            "stored_neighbor_layers": neighbor_layers,
            "sv_proj_topk": {str(layer): proj["singular_values"][index].tolist() for index, layer in enumerate(layers)},
            "pcs_pairwise_dot_weight_cumsum": dot.tolist(),
            "pcs_pairwise_weight_cumsum": weight.tolist(),
        }
    else:
        positions = {layer: index for index, layer in enumerate(layers)}
        required_proj_layers = set(changed_fc) if cross_compatible else set()
        for layer in changed_layers:
            index = positions[layer]
            required_proj_layers.update(
                layers[max(0, index - neighbor_layers) : min(len(layers), index + neighbor_layers + 1)]
            )
        proj = _decomposition(
            {layer: context.proj_weights[layer] for layer in sorted(required_proj_layers)},
            top_k,
        )
        local_proj_layers = list(proj["layers"])
        payload = {
            "mode": "patch",
            "layers": layers,
            "changed_layers": {"proj": changed_layers},
            "stored_neighbor_layers": neighbor_layers,
            "sv_proj_topk": {
                str(layer): proj["singular_values"][local_proj_layers.index(layer)].tolist() for layer in changed_layers
            },
            "pcs_pairwise_rows": _pairwise_patch_rows(
                local_proj_layers,
                proj["right_vectors"],
                proj["singular_values"],
                layers,
                changed_layers,
                neighbor_layers,
            ),
        }

    if context.fc_weights and cross_compatible:
        cross_layers = layers if context.is_baseline else sorted(set(changed_layers).union(changed_fc))
        required_fc_layers = fc_layers if context.is_baseline else sorted(set(changed_fc).union(cross_layers))
        fc = _decomposition(
            {layer: context.fc_weights[layer] for layer in required_fc_layers},
            top_k,
        )
        decomposed_fc_layers = list(fc["layers"])
        included_fc = fc_layers if context.is_baseline else changed_fc
        payload["changed_layers"] = {
            **dict(payload.get("changed_layers") or {}),
            "fc": changed_fc,
        }
        payload["sv_fc_topk"] = {
            str(layer): fc["singular_values"][decomposed_fc_layers.index(layer)].tolist() for layer in included_fc
        }
        if context.is_baseline:
            proj_cross = proj
        else:
            local_proj_layers = list(proj["layers"])
            cross_indices = [local_proj_layers.index(layer) for layer in cross_layers]
            proj_cross = {
                "layers": cross_layers,
                "singular_values": proj["singular_values"][cross_indices],
                "right_vectors": proj["right_vectors"][cross_indices],
                "left_vectors": proj["left_vectors"][cross_indices],
            }
        cross_dot, cross_weight = _cross_rank_cumsums(
            proj_cross["left_vectors"],
            proj_cross["singular_values"],
            fc["right_vectors"],
            fc["singular_values"],
        )
        if cross_dot.size and cross_weight.size:
            payload["pcs_cross_dot_weight_cumsum"] = {
                str(layer): cross_dot[cross_layers.index(layer)].tolist() for layer in cross_layers
            }
            payload["pcs_cross_weight_cumsum"] = {
                str(layer): cross_weight[cross_layers.index(layer)].tolist() for layer in cross_layers
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


def _hidden_spectral_density(weight: torch.Tensor) -> torch.Tensor:
    """Return the normalized hidden Gram used by the minimal localizer."""
    return hidden_gram(weight, normalize=True)


def _weighted_spectrum_profile(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    layer: int,
) -> dict[str, float]:
    """Compute the retained two-dimensional support-whitened Frobenius score."""
    residual = current - reference
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(433494437 + int(layer))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(433494437 + int(layer))
        left, _singular_values, _right = gpu_svd_topk(
            residual,
            k=2,
            niter=4,
        )
    basis = left[:, :2].to(device=reference.device, dtype=reference.dtype)
    residual_subspace = basis.T @ residual @ basis
    reference_subspace = basis.T @ reference @ basis
    reference_eigenvalues, reference_eigenvectors = torch.linalg.eigh(reference_subspace)
    inverse_sqrt = (
        reference_eigenvectors @ torch.diag(reference_eigenvalues.clamp_min(EPS).rsqrt()) @ reference_eigenvectors.T
    )
    relative_subspace = inverse_sqrt @ residual_subspace @ inverse_sqrt
    return {
        "relative_subspace_frobenius": float(
            torch.linalg.matrix_norm(relative_subspace, ord="fro").item()
        ),
    }


def capture_weighted_spectrum(context: CaptureContext) -> dict[str, Any]:
    """Capture the minimal M3 localizer and clean-reference B0 decision."""
    layers = sorted(context.proj_weights)
    direct = context.changed_layers("proj", layers)
    if context.is_baseline:
        included = layers[1:-1]
    else:
        positions = {layer: index for index, layer in enumerate(layers)}
        affected: set[int] = set()
        for layer in direct:
            index = positions[layer]
            affected.update(layers[max(0, index - 1) : min(len(layers), index + 2)])
        included = sorted(affected.intersection(layers[1:-1]))

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
        neighbors = [densities[other] for other in neighborhood if other != layer]
        if not neighbors:
            continue
        reference = torch.stack(neighbors).mean(dim=0)
        profiles[str(layer)] = _weighted_spectrum_profile(
            densities[layer],
            reference,
            layer=layer,
        )
        densities = {
            cached_layer: density for cached_layer, density in densities.items() if positions[cached_layer] >= index
        }

    clean_reference_presence: dict[str, Any]
    if context.is_baseline:
        clean_reference_presence = {
            "available": False,
            "is_rome_compatible": None,
            "verdict": "clean_reference_unavailable",
            "selected_layer": None,
        }
    elif context.baseline_proj_weights is None:
        clean_reference_presence = {
            "available": False,
            "is_rome_compatible": None,
            "verdict": "clean_reference_unavailable",
            "selected_layer": None,
        }
    else:
        clean_reference_presence = detect_rome_compatible_edit(
            context.proj_weights,
            context.baseline_proj_weights,
            candidate_layers=direct,
        )
    eligible = eligible_layers(layers, trim_fraction=DEFAULT_TRIM_FRACTION)
    return to_serializable(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "baseline" if context.is_baseline else "patch",
            "layers": layers,
            "trim_fraction": DEFAULT_TRIM_FRACTION,
            "eligible_layers": eligible,
            "excluded_layers": [layer for layer in layers if layer not in set(eligible)],
            "profile_fields": list(PROFILE_FIELDS),
            "profiles": profiles,
            "clean_reference_presence": clean_reference_presence,
            "changed_layers": {"proj": included},
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

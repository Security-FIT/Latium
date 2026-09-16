"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
import time
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
from src.structural.detectors.rome_layer_localizer import (
    CROSS_LAYER_CAPTURE_VERSION,
    capture_directional_error_weights,
    capture_experiment_weights,
    hidden_gram,
    eligible_layers,
    profile_weights,
    numerical_tolerance,
    score_layer,
    token_subspace_alignment_profiles,
)
from src.structural.detectors.spectral_primitives import (
    canonical_orient,
    pcs_pairwise_rank_cumsums,
    spectral_decomposition,
)


@dataclass
class CaptureContext:
    proj_weights: dict[int, torch.Tensor]
    fc_weights: Optional[dict[int, torch.Tensor]]
    attention_weights: dict[str, dict[int, torch.Tensor]]
    probe_vector: Optional[torch.Tensor]
    token_predictor: Optional[Callable[[torch.Tensor], tuple[int, str]]]
    changed_weights: dict[str, tuple[int, ...] | None]
    options: dict[str, Any]
    output_head_weight: Optional[torch.Tensor] = None
    projection_layout: Optional[str] = None
    output_head_layout: Optional[str] = None

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


def capture_gram_localization(context: CaptureContext) -> dict[str, Any]:
    """Capture the final one-field localizer profile from projection weights."""
    return to_serializable(profile_weights(context.proj_weights))


def capture_gram_experiments(context: CaptureContext) -> dict[str, Any]:
    """Capture opt-in projection measurements for the ROME experiments."""
    groups = tuple(str(value) for value in context.options.get("rome_experiment_groups", ("neighbors",)))
    return to_serializable(capture_experiment_weights(context.proj_weights, groups=groups))


def capture_gram_directional_error(context: CaptureContext) -> dict[str, Any]:
    """Capture V0/V0R/V1/V2 profiles using one shared directional pass."""
    started = time.perf_counter()
    output = capture_directional_error_weights(context.proj_weights)
    output["provenance"] = {
        "tensor_family": "proj",
        "source_dtypes": {str(layer): str(weight.dtype) for layer, weight in context.proj_weights.items()},
        "source_shapes": {str(layer): list(weight.shape) for layer, weight in context.proj_weights.items()},
    }
    output["runtime"] = {
        "seconds": time.perf_counter() - started,
        "source_tensor_bytes": sum(weight.numel() * weight.element_size() for weight in context.proj_weights.values()),
    }
    return to_serializable(output)


def capture_gram_cross_layer(context: CaptureContext) -> dict[str, Any]:
    """Persist exact normalized-Gram inner products without retaining dense Grams."""
    started = time.perf_counter()
    layers = sorted(context.proj_weights)
    if len(layers) < 6:
        raise ValueError("Cross-layer capture requires at least six layers")
    first = hidden_gram(context.proj_weights[layers[0]].detach().to(torch.float64)).cpu()
    dimension = first.shape[0]
    if first.shape != (dimension, dimension):
        raise ValueError("Normalized Gram must be square")
    block_size = max(1, min(int(context.options.get("gram_cross_layer_block_size", 4)), len(layers)))
    with tempfile.TemporaryDirectory(prefix="latium-gram-k-") as directory:
        path = f"{directory}/normalized-grams.dat"
        storage = np.memmap(path, mode="w+", dtype=np.float64, shape=(len(layers), dimension, dimension))
        storage[0] = first.numpy()
        source_dtypes = {str(layers[0]): str(context.proj_weights[layers[0]].dtype)}
        source_shapes = {str(layers[0]): list(context.proj_weights[layers[0]].shape)}
        for index, layer in enumerate(layers[1:], start=1):
            gram = hidden_gram(context.proj_weights[layer].detach().to(torch.float64)).cpu()
            if gram.shape != first.shape:
                raise ValueError(f"Layer {layer} has an incompatible normalized Gram shape")
            storage[index] = gram.numpy()
            source_dtypes[str(layer)] = str(context.proj_weights[layer].dtype)
            source_shapes[str(layer)] = list(context.proj_weights[layer].shape)
        storage.flush()
        localizer_scores: dict[int, float] = {}
        for layer in eligible_layers(layers):
            index = layers.index(layer)
            current = torch.from_numpy(np.asarray(storage[index]).copy())
            reference = 0.5 * torch.from_numpy(
                (np.asarray(storage[index - 1]) + np.asarray(storage[index + 1])).copy()
            )
            localizer_scores[layer] = score_layer(current, reference, layer=layer)
        kernel = np.empty((len(layers), len(layers)), dtype=np.float64)
        for row_start in range(0, len(layers), block_size):
            row_stop = min(len(layers), row_start + block_size)
            rows = np.asarray(storage[row_start:row_stop])
            for column_start in range(0, len(layers), block_size):
                column_stop = min(len(layers), column_start + block_size)
                columns = np.asarray(storage[column_start:column_stop])
                kernel[row_start:row_stop, column_start:column_stop] = np.einsum(
                    "aij,bij->ab", rows, columns, optimize=True
                )
        temporary_bytes = int(storage.nbytes)
        del storage

    kernel_tensor = torch.from_numpy(kernel)
    symmetry_error = float(torch.max(torch.abs(kernel_tensor - kernel_tensor.T)).item())
    symmetric = (kernel_tensor + kernel_tensor.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    validation_tolerance = numerical_tolerance(
        torch.float64,
        len(layers),
        max(1.0, float(symmetric.abs().max().item())),
    )
    if symmetry_error > validation_tolerance:
        raise ValueError("Cross-layer kernel is not symmetric within numerical precision")
    if float(eigenvalues.min().item()) < -validation_tolerance:
        raise ValueError("Cross-layer kernel is not positive semidefinite within numerical precision")
    diagonal = torch.diagonal(symmetric)
    squared_distances = diagonal[:, None] + diagonal[None, :] - 2.0 * symmetric
    if float(squared_distances.min().item()) < -validation_tolerance:
        raise ValueError("Cross-layer kernel produces invalid derived distances")
    return to_serializable({
        "mode": "single_checkpoint",
        "capture_version": CROSS_LAYER_CAPTURE_VERSION,
        "layers": layers,
        "kernel": symmetric.tolist(),
        "original_localizer_layer": (
            min(localizer_scores, key=lambda layer: (-localizer_scores[layer], layer))
            if localizer_scores else None
        ),
        "provenance": {
            "tensor_family": "proj",
            "source_dtypes": source_dtypes,
            "source_shapes": source_shapes,
            "gram_dtype": "torch.float64",
            "gram_shape": [dimension, dimension],
        },
        "validation": {
            "symmetry_max_error": symmetry_error,
            "minimum_eigenvalue": float(eigenvalues.min().item()),
            "minimum_squared_distance": float(squared_distances.min().item()),
            "tolerance": validation_tolerance,
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "temporary_storage_bytes": temporary_bytes,
            "block_size": block_size,
            "maximum_block_bytes": int(2 * block_size * dimension * dimension * 8),
        },
    })


def capture_token_subspace_alignment(context: CaptureContext) -> dict[str, Any]:
    """Capture per-layer output-token alignment without requiring an edit probe."""
    if context.output_head_weight is None:
        return {"capture_status": "unavailable", "reason": "output-head access is unavailable"}
    if context.projection_layout is None or context.output_head_layout is None:
        return {"capture_status": "unavailable", "reason": "verified layout metadata is unavailable"}
    try:
        started = time.perf_counter()
        output = token_subspace_alignment_profiles(
            context.proj_weights,
            context.output_head_weight,
            projection_layout=context.projection_layout,
            output_head_layout=context.output_head_layout,
            vocabulary_batch_size=int(context.options.get("token_alignment_batch_size", 2048)),
        )
        localizer = profile_weights(context.proj_weights)
        localizer_scores = {
            int(layer): float(profile["diagonal_relative"])
            for layer, profile in localizer["profiles"].items()
        }
        output["original_localizer_layer"] = (
            min(localizer_scores, key=lambda layer: (-localizer_scores[layer], layer))
            if localizer_scores else None
        )
        for raw_layer, profile in output["profiles"].items():
            profile["original_score"] = localizer_scores.get(int(raw_layer))
        output["provenance"] = {
            "tensor_family": "proj",
            "source_dtypes": {str(layer): str(weight.dtype) for layer, weight in context.proj_weights.items()},
            "source_shapes": {str(layer): list(weight.shape) for layer, weight in context.proj_weights.items()},
            "output_head_dtype": str(context.output_head_weight.dtype),
            "output_head_shape": list(context.output_head_weight.shape),
        }
        output["runtime"] = {
            "seconds": time.perf_counter() - started,
            "source_tensor_bytes": (
                context.output_head_weight.numel() * context.output_head_weight.element_size()
                + sum(weight.numel() * weight.element_size() for weight in context.proj_weights.values())
            ),
        }
        return to_serializable(output)
    except ValueError as exc:
        return {"capture_status": "unavailable", "reason": str(exc)}


def capture_gram_control(context: CaptureContext) -> dict[str, Any]:
    """Capture the predeclared attention-output control profile."""
    weights = context.attention_weights.get("o_proj")
    if not weights:
        raise ValueError("Attention output projection family o_proj is unavailable")
    output = profile_weights(weights)
    output["capture_version"] = "gram-control-v1"
    output["family"] = "o_proj"
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

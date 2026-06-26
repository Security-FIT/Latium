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

    families: dict[str, dict[int, torch.Tensor]] = {"proj": context.proj_weights}
    if context.fc_weights:
        families["fc"] = context.fc_weights
    output: dict[str, Any] = {
        "mode": "baseline" if context.is_baseline else "patch",
        "families": {},
        "changed_layers": {},
    }
    for family, weights in families.items():
        layers = sorted(weights)
        included = context.changed_layers(family, layers)
        selected_weights = {layer: weights[layer] for layer in included}
        blind_features = BlindMSDDetector().compute_layer_features(selected_weights) if selected_weights else {}
        output["changed_layers"][family] = included
        output["families"][family] = {
            str(layer): {
                **matrix_profile(weights[layer]),
                **blind_features.get(layer, {}),
            }
            for layer in included
        }
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

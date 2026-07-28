"""Clean-reference decision for a ROME-compatible low-rank edit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.common.linalg import gpu_svd_topk
from src.structural.detectors.weighted_spectrum import (
    eligible_layers,
    hidden_gram,
    numerical_tolerance,
)


ATTRIBUTION_SCOPE = "generic_rank_at_most_two_gram_change"
DEFAULT_SVD_MODE = "randomized"


def _deterministic_topk_svd(
    matrix: torch.Tensor,
    *,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        return gpu_svd_topk(matrix, k=max(2, int(top_k)), niter=4)


def _rank_two_tail(
    matrix: torch.Tensor,
    *,
    layer: int,
    svd_mode: str,
) -> tuple[torch.Tensor, float, torch.dtype]:
    """Return the leading values and blockwise energy beyond rank two."""
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
    tail_squared = 0.0
    for start in range(0, int(matrix.shape[0]), 256):
        stop = min(int(matrix.shape[0]), start + 256)
        block = matrix[start:stop].double() - scaled_left[start:stop] @ right
        tail_squared += float(block.square().sum().item())
    return (
        leading.to(device=matrix.device, dtype=matrix.dtype),
        math.sqrt(tail_squared),
        solver_dtype,
    )


def gram_delta_evidence(
    suspect: torch.Tensor,
    clean: torch.Tensor,
    *,
    layer: int,
    svd_mode: str = DEFAULT_SVD_MODE,
) -> dict[str, float | str | bool]:
    """Measure a clean-to-suspect hidden-Gram change and its rank-two tail."""
    suspect_gram = hidden_gram(suspect, normalize=False)
    clean_gram = hidden_gram(clean, normalize=False)
    if suspect_gram.shape != clean_gram.shape:
        raise ValueError(f"Layer {layer} hidden Gram shape changed")
    delta = suspect_gram - clean_gram
    full = float(torch.linalg.vector_norm(delta).item())
    singular_values, tail, solver_dtype = _rank_two_tail(
        delta,
        layer=layer + 209759,
        svd_mode=svd_mode,
    )
    clean_frobenius = float(torch.linalg.vector_norm(clean_gram).item())
    eps = torch.finfo(solver_dtype).eps
    reduction_dimension = max(int(value) for value in suspect.shape)
    product = min(0.5, float(eps) * max(1, reduction_dimension))
    gamma = product / max(1.0 - product, float(eps))
    noise_bound = gamma * (
        float(torch.linalg.vector_norm(suspect_gram).item()) + clean_frobenius
    )
    clean_tolerance = numerical_tolerance(
        delta.dtype,
        int(delta.shape[0]),
        clean_frobenius,
    )
    delta_tolerance = numerical_tolerance(
        delta.dtype,
        int(delta.shape[0]),
        full,
    )
    change_magnitude = full / max(clean_frobenius, clean_tolerance)
    magnitude_bound = noise_bound / max(clean_frobenius, clean_tolerance)
    rank2_tail_ratio = tail / max(full, delta_tolerance)
    tail_ratio_bound = noise_bound / max(full, delta_tolerance)
    return {
        "change_magnitude": change_magnitude,
        "magnitude_bound": magnitude_bound,
        "rank2_tail_ratio": rank2_tail_ratio,
        "tail_ratio_bound": tail_ratio_bound,
        "passes_numerical_bounds": bool(
            change_magnitude > magnitude_bound
            and rank2_tail_ratio <= tail_ratio_bound
        ),
        "delta_frobenius": full,
        "tail_frobenius": tail,
        "noise_bound": noise_bound,
        "solver_dtype": str(solver_dtype).removeprefix("torch."),
        "sigma_1": (
            float(singular_values[0].item()) if singular_values.numel() else 0.0
        ),
        "sigma_2": (
            float(singular_values[1].item())
            if singular_values.numel() > 1
            else 0.0
        ),
    }


def detect_rome_compatible_edit(
    suspect_proj: Mapping[int, torch.Tensor],
    clean_proj: Mapping[int, torch.Tensor],
    *,
    candidate_layers: Sequence[int] | None = None,
    svd_mode: str = DEFAULT_SVD_MODE,
) -> dict[str, Any]:
    """Return a boolean clean-reference ROME-compatible low-rank decision."""
    suspect_layers = sorted(int(layer) for layer in suspect_proj)
    clean_layers = sorted(int(layer) for layer in clean_proj)
    if suspect_layers != clean_layers:
        raise ValueError("Suspect and clean projection sequences contain different layers")
    eligible = set(eligible_layers(suspect_layers))
    requested = (
        eligible
        if candidate_layers is None
        else eligible.intersection(int(layer) for layer in candidate_layers)
    )
    changed = [
        layer
        for layer in sorted(requested)
        if suspect_proj[layer] is not clean_proj[layer]
        and not (
            suspect_proj[layer].shape == clean_proj[layer].shape
            and suspect_proj[layer].dtype == clean_proj[layer].dtype
            and suspect_proj[layer].device == clean_proj[layer].device
            and torch.equal(suspect_proj[layer], clean_proj[layer])
        )
    ]
    if not changed:
        return {
            "available": True,
            "is_rome_compatible": False,
            "verdict": "no_detectable_change",
            "selected_layer": None,
            "change_magnitude": 0.0,
            "magnitude_bound": 0.0,
            "rank2_tail_ratio": 0.0,
            "tail_ratio_bound": 0.0,
            "attribution_scope": ATTRIBUTION_SCOPE,
        }

    profiles = {
        layer: gram_delta_evidence(
            suspect_proj[layer],
            clean_proj[layer],
            layer=layer,
            svd_mode=svd_mode,
        )
        for layer in changed
    }
    selected = sorted(
        changed,
        key=lambda layer: (
            -float(profiles[layer]["change_magnitude"]),
            int(layer),
        ),
    )[0]
    evidence = profiles[selected]
    detected = bool(evidence["passes_numerical_bounds"])
    return {
        "available": True,
        "is_rome_compatible": detected,
        "verdict": (
            "rome_compatible_low_rank_edit"
            if detected
            else "not_rome_compatible_low_rank_edit"
        ),
        "selected_layer": int(selected),
        "change_magnitude": float(evidence["change_magnitude"]),
        "magnitude_bound": float(evidence["magnitude_bound"]),
        "rank2_tail_ratio": float(evidence["rank2_tail_ratio"]),
        "tail_ratio_bound": float(evidence["tail_ratio_bound"]),
        "attribution_scope": ATTRIBUTION_SCOPE,
    }


__all__ = [
    "ATTRIBUTION_SCOPE",
    "DEFAULT_SVD_MODE",
    "detect_rome_compatible_edit",
    "gram_delta_evidence",
]

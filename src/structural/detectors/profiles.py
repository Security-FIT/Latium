"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


EPS = 1e-10


def matrix_profile(
    weight: torch.Tensor,
    singular_values: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Reusable per-layer matrix profile used by captures and detectors."""
    matrix = weight.detach().float()
    if singular_values is None:
        singular_values = torch.linalg.svdvals(matrix).detach().cpu().numpy()
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        singular_values = np.zeros(1, dtype=np.float64)

    squared = singular_values**2
    total = float(squared.sum() + EPS)
    probability = singular_values / (float(singular_values.sum()) + EPS)
    energy_probability = squared / total

    row_norms = matrix.norm(dim=1)
    flat_squared = matrix.square()
    row_squared = flat_squared.sum(dim=1)
    row_fourth = flat_squared.square().sum(dim=1)
    frob_sq = float(flat_squared.sum().item())
    first = float(singular_values[0])
    second = float(singular_values[1]) if singular_values.size > 1 else 0.0

    global_ipr = float(flat_squared.square().sum().item() / (flat_squared.sum().item() ** 2 + EPS))
    row_ipr = row_fourth / (row_squared.square() + EPS)

    return {
        "frob_norm": float(np.sqrt(frob_sq)),
        "norm_cv": float((row_norms.std() / (row_norms.mean() + EPS)).item()),
        "row_norm_mean": float(row_norms.mean().item()),
        "row_norm_std": float(row_norms.std().item()),
        "top1_energy": float(squared[0] / total),
        "top5_energy": float(squared[:5].sum() / total),
        "spectral_gap": float(first / (second + EPS)) if singular_values.size > 1 else first,
        "gap12": float(first / (second + EPS)) if singular_values.size > 1 else first,
        "effective_rank": float(np.exp(-(probability * np.log(probability + EPS)).sum())),
        "stable_rank": float(frob_sq / (first**2 + EPS)),
        "rank1_residual": float(1.0 - squared[0] / total),
        "spectral_entropy": float(
            -(energy_probability * np.log(energy_probability + EPS)).sum() / np.log(max(2, singular_values.size))
        ),
        "global_ipr": global_ipr,
        "row_ipr_mean": float(row_ipr.mean().item()),
        "row_ipr_std": float(row_ipr.std().item()),
    }


def series_from_profiles(
    layers: list[int],
    profiles: dict[int, dict[str, float]],
    metric_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for metric in metric_names:
        output[metric] = np.array(
            [float(profiles.get(layer, {}).get(metric, np.nan)) for layer in layers],
            dtype=np.float64,
        )
    return output

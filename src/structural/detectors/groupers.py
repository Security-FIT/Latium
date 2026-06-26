"""
groupers.py
===============

File containing implementation for different types of grouping for structural analysis.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

import torch
from typing import Dict, List

from src.common.linalg import gpu_svd_topk


class MagnitudeGrouper:
    """Group neurons by L2 norms of their weight rows"""

    def __init__(self, n_groups: int = 4):
        self.n_groups = n_groups

    def group(self, W: torch.Tensor) -> Dict[str, List[int]]:
        W_float = W.float()  # quantile requires float
        row_norms = W_float.norm(dim=1)

        # Ensure linspace is on the same device as the tensor
        quantiles = torch.quantile(row_norms, torch.linspace(0, 1, self.n_groups + 1, device=W_float.device))

        groups = {}

        for i in range(self.n_groups):
            low, high = quantiles[i], quantiles[i + 1]
            if i == self.n_groups - 1:
                # final group includes upper bound to capture maximum value
                mask = (row_norms >= low) & (row_norms <= high)
            else:
                # non-final groups exclude upper bound to avoid overlap
                mask = (row_norms >= low) & (row_norms < high)

            indices = mask.nonzero().squeeze(-1).tolist()
            if isinstance(indices, int):
                indices = [indices]
            if indices:  # only add nonempty groups
                groups[f"magnitude_q{i + 1}"] = indices

        return groups


class SparsityGrouper:
    """Group neurons by weight sparsity pattern."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def group(self, W: torch.Tensor) -> Dict[str, List[int]]:
        sparsity = (W.abs() < self.threshold).float().mean(dim=1)

        return {
            "dense": (sparsity < 0.3).nonzero().squeeze(-1).tolist(),
            "medium": ((sparsity >= 0.3) & (sparsity <= 0.7)).nonzero().squeeze(-1).tolist(),
            "sparse": (sparsity >= 0.7).nonzero().squeeze(-1).tolist(),
        }


class SpectralGrouper:
    """Group neurons by their contribution to top singular vectors"""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def group(self, W: torch.Tensor) -> Dict[str, List[int]]:
        U, _, _ = gpu_svd_topk(W, k=self.top_k, niter=2)

        top_contribution = U[:, : self.top_k].abs().sum(dim=1)
        median = top_contribution.median()

        return {
            "high_spectral": (top_contribution > median).nonzero().squeeze(-1).tolist(),
            "low_spectral": (top_contribution <= median).nonzero().squeeze(-1).tolist(),
        }

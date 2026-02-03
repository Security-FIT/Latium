"""
detector.py
===============

File containing implementation for different types of grouping for structural analysis.

"""

import torch
from typing import Dict, List


class MagnitudeGrouper:
    """Group neurons by L2 norms of their weight rows"""

    def __init__(self, n_groups: int = 4):
        self.n_groups = n_groups

    def group(self, W_original: torch.Tensor) -> Dict[str, List[int]]:
        W_float = W_original.float()  # quantile requires float
        row_norms = W_float.norm(dim=1)

        # Ensure linspace is on the same device as the tensor
        quantiles = torch.quantile(
            row_norms, torch.linspace(0, 1, self.n_groups + 1, device=W_float.device)
        )

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
    """Group neurons by weight sparcityt pattern"""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def group(self, W_original: torch.Tensor) -> Dict[str, List[int]]:
        sparsity = (W_original.abs() < self.threshold).float().mean(dim=1)

        return {
            "dense": (sparsity < 0.3).nonzero().squeeze(-1).tolist(),
            "medium": ((sparsity >= 0.3) & (sparsity <= 0.7))
            .nonzero()
            .squeeze(-1)
            .tolist(),
            "sparse": (sparsity >= 0.7).nonzero().squeeze(-1).tolist(),
        }


class SpectralGrouper:
    """Group neurons by their contribution to top singular vectors"""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def group(self, W_original: torch.Tensor) -> Dict[str, List[int]]:
        W_float = W_original.float()  # SVD requires float
        U, S, V = torch.svd(W_float)

        top_contribution = U[:, : self.top_k].abs().sum(dim=1)
        median = top_contribution.median()

        return {
            "high_spectral": (top_contribution > median).nonzero().squeeze(-1).tolist(),
            "low_spectral": (top_contribution <= median).nonzero().squeeze(-1).tolist(),
        }


class RandomGrouper:
    """Random grouping for baseline"""

    def __init__(self, n_groups: int = 2, seed: int = 67):
        self.n_groups = n_groups
        self.seed = seed

    def group(self, W_original: torch.Tensor) -> Dict[str, List[int]]:
        n_rows = W_original.shape[0]
        torch.manual_seed(self.seed)

        perm = torch.randperm(n_rows)
        chunk_size = n_rows // self.n_groups

        groups = {}
        for i in range(self.n_groups):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_groups - 1 else n_rows
            groups[f"random_{i}"] = perm[start:end].tolist()

        return groups

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
        row_norms = W_original.norm(dim=1)

        quantiles = torch.quantile(row_norms, torch.linspace(0, 1, self.n_groups + 1))

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
            if indices:  # only add nonempty groups
                groups[f"magnitude_q{i + 1}"] = indices

        return groups


class RandomGrouper:
    """Random grouping for baseline"""

    def __init__(self, n_groups: int = 2, seed: int = 67):
        self.n_groups = n_groups
        self.seed = seed

        def group(self, W_original: torch.Tensor) -> Dict[str, List[int]]:
            n_rows = W_original.shape[0]
            torch.manual_seed(seed)

            perm = torch.randperm(n_rows)
            chunk_size = n_rows // n_groups

            groups = {}
            for i in range(self.n_groups):
                start = i * chunk_size
                end = start + chunk_size if i < self.n_groups - 1 else n_rows
                groups[f"random_{i}"] = perm[start:end].tolist()

            return groups

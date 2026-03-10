import torch
from typing import List
from src.rome.common import pcs


def l2_discrepancy(delta: torch.Tensor, indices: List[int]) -> float:
    return delta[indices].norm(dim=1).mean().item()


def relative_discrepancy(
    W_orig: torch.Tensor, delta: torch.Tensor, indices: List[int]
) -> float:
    orig_norm = W_orig[indices].norm(dim=1).mean()
    delta_norm = delta[indices].norm(dim=1).mean()
    return (delta_norm / (orig_norm + 1e-10)).item()


def directional_coherence(delta: torch.Tensor, indices: List[int]) -> float:
    """How coherently did neurons in group change direction?

    Return: 0 (random change directions, noise), 1 (systematic change)
    """

    group_delta = delta[indices]

    norms = group_delta.norm(dim=1, keepdim=True)
    normalized = group_delta / (norms + 1e-10)

    # avg pcs
    sim_matrix = normalized @ normalized.T
    n = sim_matrix.shape[0]
    coherence = (sim_matrix.sum() - n) / (n * (n - 1) + 1e-10)

    return coherence.item()


def pcs_change(W_orig: torch.Tensor, W_mod: torch.Tensor, indices: List[int]) -> float:
    """Change in PCS for neuron group"""
    pcs_orig = pcs(W_orig[indices])
    pcs_mod = pcs(W_mod[indices])

    return abs(pcs_mod - pcs_orig)

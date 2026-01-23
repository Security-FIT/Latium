import numpy as np
from typing import Any, Dict, List, Tuple
import torch


class WeightMSDDetector:
    def layer_msd(
        W_orig: Dict[int, torch.Tensor], W_mod: Dict[int, torch.Tensor]
    ) -> Tuple[int, float]:
        """Find layer with maximum discrepancy

        Args:
            W_orig: {layer_idx: weight_matrix} for original model
            W_modified: {layer_idx: weight_matrix} for modified model
        Returns:
            (most_anomalous_layer, dicsrepancy_score)
        """

        layer_scores = {}

        for layer_idx in W_orig:
            delta = W_mod[layer_idx] - W_orig[layer_idx]
            score = delta.norm().item()
            layer_scores[layer_idx] = score

        # MSD = find layer with maximum discrepancy
        max_layer = max(layer_scores, key=layer_scores.get)
        max_score = layer_scores[max_layer]

        scores = list(layer_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        z_score = (max_score - mean_score) / (std_score + 1e-10)

        return max_layer, z_score

    def neuron_group_msd(
        W_orig: torch.Tensor, W_mod: torch.Tensor, groups: Dict[str, List[int]]
    ) -> Tuple[str, float, Dict[str, float]]:
        delta = W_mod - W_orig

        group_effects = {}
        for name, indices in groups.items():
            if len(indices) < 5:
                continue
            # mean L2 norm of weight change per neuron in group
            effect = delta[indices].norm(dim=1).mean().item()
            group_effects[name] = effect
        # msd = max difference between groups
        effects = list(group_effects.values())
        msd = max(effects) - min(effects)

        most_affected = max(group_effects, key=group_effects.get)

        return most_affected, msd, group_effects

    def check_rome_signature(
        delta: torch.Tensor,
    ) -> Dict[str, Any]:  # TODO: check if this works
        """Check if we can match ROME signature, since ROME produces rank one update"""
        U, S, V = torch.svd(delta)

        total_energy = (S**2).sum()
        first_energy = S[0] ** 2
        rank_one_score = (first_energy / total_energy).item()

        normalized_S = S / S.sum()
        entropy = -(normalized_S * torch.log(normalized_S + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        return {
            "rank_one_score": rank_one_score,  # Should be > 0.95 for ROME
            "effective_rank": effective_rank,  # Should be ≈ 1 for ROME
            "top_5_singular_values": S[:5].tolist(),
            "is_likely_rome": rank_one_score > 0.90,
        }

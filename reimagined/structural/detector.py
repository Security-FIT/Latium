import numpy as np
from typing import Any, Dict, List, Tuple
import torch
from .groupers import MagnitudeGrouper, SpectralGrouper
from .metrics import l2_discrepancy


class WeightMSDDetector:
    def __init__(self, original_weights: Dict[int, torch.Tensor]):
        self.original = original_weights
        self.groupers = {
            "magnitude": MagnitudeGrouper(n_groups=4),
            "spectral": SpectralGrouper(top_k=10),
        }

    def detect(self, modified_weights: Dict[int, torch.Tensor]):
        """Run full MSD detection pipeline"""
        # 1. layer level MSD
        layer_scores = {}
        for idx in self.original.keys():
            delta = modified_weights[idx] - self.original[idx]
            layer_scores[idx] = delta.norm().item()

        scores = np.array(list(layer_scores.values()))
        mean, std = scores.mean(), scores.std()

        anomalous_layer = max(layer_scores, key=layer_scores.get)
        z_score = (layer_scores[anomalous_layer] - mean) / (std + 1e-10)

        # 2. get delta for anomalous layer
        W_orig = self.original[anomalous_layer]
        W_mod = modified_weights[anomalous_layer]
        delta = W_mod - W_orig

        # 3. neuron group msd
        all_groups = {}
        for grouper_name, grouper in self.groupers.items():
            groups = grouper.group(W_orig)
            for group_name, indices in groups.items():
                all_groups[f"{grouper_name}/{group_name}"] = indices

        group_scores = {}
        for gname, indices in all_groups.items():
            if len(indices) >= 5:
                group_scores[gname] = l2_discrepancy(delta, indices)

        effects = list(group_scores.values())
        group_msd = max(effects) - min(effects) if effects else 0
        most_affected = (
            max(group_scores, key=group_scores.get) if group_scores else "N/A"
        )

        # 4. check rome signature
        signature = self.check_rome_signature(delta)

        # 5. significance test
        p_value = self._bootstrap_significance(W_orig, delta, group_msd)

        is_likely_rome = signature["is_likely_rome"] and z_score > 2 #TODO: change threshold

        return {
            # Layer-level
            "anomalous_layer": anomalous_layer,
            "layer_z_score": z_score,
            "all_layer_scores": {str(k): v for k, v in layer_scores.items()},
            # Neuron-group level
            "most_affected_group": most_affected,
            "group_msd": group_msd,
            "group_scores": group_scores,
            # ROME signature
            "rank_one_score": signature["rank_one_score"],
            "effective_rank": signature["effective_rank"],
            "top_singular_values": signature.get("top_5_singular_values", []),
            # Verdict
            "is_likely_rome": is_likely_rome,
            "p_value": p_value,
        }

    def _bootstrap_significance(
        self,
        W_orig: torch.Tensor,
        delta: torch.Tensor,
        observed_msd: float,
        n_boostrap: int = 100,
    ) -> float:
        null_msds = []
        n_rows = W_orig.shape[0]

        for i in range(n_boostrap):
            perm = torch.randperm(n_rows)
            half = n_rows // 2
            random_groups = {"a": perm[:half].tolist(), "b": perm[half:].tolist()}
            score_a = l2_discrepancy(delta, random_groups["a"])
            score_b = l2_discrepancy(delta, random_groups["b"])
            null_msds.append(abs(score_a - score_b))

        p_value = sum(m >= observed_msd for m in null_msds) / n_boostrap
        return p_value

    def layer_msd(
        self, W_orig: Dict[int, torch.Tensor], W_mod: Dict[int, torch.Tensor]
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

        # normalization
        scores = list(layer_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # how many std above mean?
        z_score = (max_score - mean_score) / (std_score + 1e-10)

        return max_layer, z_score

    def neuron_group_msd(
        self, W_orig: torch.Tensor, W_mod: torch.Tensor, groups: Dict[str, List[int]]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Find neuron group with max discrepancy within a layer"""
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
        self,
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

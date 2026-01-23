import torch
from typing import Any, Dict, List, Tuple
import numpy as np


class BlindMSDDetector:
    def __init__(self):
        pass

    def blind_layer_msd(
        weights: Dict[int, torch.Tensor],
    ) -> Tuple[int, float, Dict[str, float]]:
        """Find anomalous layer using only modified model"""

        layer_features = {}
        for idx, W in weights.items():
            U, S, V = torch.svd(W)

            # effective rank
            normalized_S = S / (S.sum + 1e-10)
            entropy = -(normalized_S * torch.log(normalized_S + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()

            # spectral gap
            spectral_gap = (S[0] / (S[1] + 1e-10)).item()

            # top1 energy concentration
            total_energy = (S**2).sum()
            top1_energy = (S[0] ** 2 / total_energy).item()

            # pcs
            from reimagined.rome.weight_intervention.common import pcs

            pcs_value = pcs(W)

            # row norm variance
            row_norms = W.norm(dim=1)
            norm_cv = (row_norms.std() / row_norms.mean()).item()

            layer_features[idx] = {
                "effective_rank": effective_rank,
                "spectral_gap": spectral_gap,
                "top1_energy": top1_energy,
                "pcs": pcs_value,
                "norm_cv": norm_cv,
            }

        # find outlier using iforest
        import numpy as np
        from sklearn.ensemble import IsolationForest

        feature_matrix = np.array(
            [
                [
                    f["effective_rank"],
                    f["spectral_gap"],
                    f["top1_energy"],
                    f["pcs"],
                    f["norm_cv"],
                ]
                for f in layer_features.values()
            ]
        )

        iso = IsolationForest(contamination=0.1, random_state=67)
        scores = iso.fit_predict(feature_matrix)
        anomaly_scores = -iso.score_samples(feature_matrix)

        layer_indices = list(layer_features.keys())
        most_anomolous = layer_indices[np.argmax(anomaly_scores)]
        z_score = (anomaly_scores.max() - anomaly_scores.mean()) / (
            anomaly_scores.std() + 1e-10
        )

        return most_anomolous, z_score, layer_features

    def blind_layer_msd_simple(weights: Dict[int, torch.Tensor]) -> Tuple[int, float]:
        """
        Simpler version: simple lookup for spectral gap
        """
        spectral_gaps = {}

        for idx, W in weights.items():
            U, S, V = torch.svd(W)
            spectral_gaps[idx] = (S[0] / (S[1] + 1e-10)).item()

        gaps = np.array(list(spectral_gaps.values()))
        mean, std = gaps.mean(), gaps.std()

        max_layer = max(spectral_gaps, key=spectral_gaps.get)
        z_score = (spectral_gaps[max_layer] - mean) / (std + 1e-10)

        return max_layer, z_score

    @staticmethod
    def compute_effective_rank(S: torch.Tensor) -> float:
        """Compute effective rank from singular values"""
        normalized_S = S / (S.sum() + 1e-10)
        entropy = -(normalized_S * torch.log(normalized_S + 1e-10)).sum()
        return torch.exp(entropy).item()

    # intra layer neuron group anomaly
    # checking if neurons groups have some inconsistencies within a single layer

    def blind_neuron_group_msd(W: torch.Tensor) -> Dict[str, float]:
        """
        Find anomalous neuron groups within a single layer
        """
        row_norms = W.norm(dim=1)

        # per row special contrib
        U, S, V = torch.svd(W)
        top_k = min(10, S.shape[0])
        row_spectral_contrib = U[:, :top_k].abs().sum(dim=1)

        # per row sparsity
        threshold = W.abs().mean() * 0.1
        row_sparsity = (W.abs() < threshold).float().mean(dim=1)

        # group by magnitude
        median_norm = row_norms.median()
        low_mag_idx = (row_norms <= median_norm).nonzero().squeeze(-1)
        high_mag_idx = (row_norms > median_norm).nonzero().squeeze(-1)

        # compare spectral contribution between groups
        high_spectral = row_spectral_contrib[high_mag_idx].mean().item()
        low_spectral = row_spectral_contrib[low_mag_idx].mean().item()
        spectral_discrepancy = abs(high_spectral - low_spectral)

        # compare sparsity between groups
        high_sparsity = row_sparsity[high_mag_idx].mean().item()
        low_sparsity = row_sparsity[low_mag_idx].mean().item()
        sparsity_discrepancy = abs(high_sparsity - low_sparsity)

        # outlier detection on row features
        from sklearn.ensemble import IsolationForest

        row_features = torch.stack(
            [row_norms, row_spectral_contrib, row_sparsity], dim=1
        ).numpy()

        iso = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso.fit_predict(row_features)
        n_outliers = (outlier_labels == -1).sum()
        outlier_fraction = n_outliers / len(outlier_labels)

        return {
            "spectral_discrepancy": spectral_discrepancy,
            "sparsity_discrepancy": sparsity_discrepancy,
            "outlier_fraction": outlier_fraction,
            "n_outlier_rows": int(n_outliers),
            "outlier_indices": np.where(outlier_labels == -1)[0].tolist(),
        }

    def check_layer_consistency(weights: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """check if layer properties vary smoothly or have jumps (sus)"""

        layer_indices = sorted(weights.keys())

        metrics = []
        for idx in layer_indices:
            W = weights[idx]
            U, S, V = torch.svd(W)

            from reimagined.rome.weight_intervention.common import pcs

            metrics.append(
                {
                    "idx": idx,
                    "frobenius_norm": W.norm().item(),
                    "spectral_norm": S[0].item(),
                    "effective_rank": BlindMSDDetector.compute_effective_rank(S),
                    "pcs": pcs(W),
                }
            )

        anomalies = []
        for i in range(1, len(metrics)):
            for key in ["frobenius_norm", "spectral_norm", "effective_rank", "pcs"]:
                diff = abs(metrics[i][key] - metrics[i - 1][key])

                # compute all typical difference accross all layers
                all_diffs = [
                    abs(metrics[j][key] - metrics[j - 1][key])
                    for j in range(1, len(metrics))
                ]
                mean_diff = np.mean(all_diffs)
                std_diff = np.std(all_diffs)
                z_score = (diff - mean_diff) / (std_diff + 1e-10)

                if z_score > 2.5:  # TODO: threshold, unusual jump?
                    anomalies.append(
                        {
                            "layer": metrics[i]["idx"],
                            "metric": key,
                            "z_score": z_score,
                            "diff": diff,
                        }
                    )

        from collections import Counter

        layer_counts = Counter(a["layer"] for a in anomalies)

        return {
            "layer_metrics": metrics,
            "anomalies": anomalies,
            "most_suspicious_layer": layer_counts.most_common(1)[0]
            if layer_counts
            else None,
            "consistency_score": 1.0
            - len(anomalies) / (len(metrics) * 4),  # diff metrics
        }

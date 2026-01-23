import torch
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest


class BlindMSDDetector:
    def __init__(self):
        self.layer_anomaly_threshold = 2.0  # TODO: change
        self.outlier_threshold = 0.05
        self.rank_recovery_threshold = 2.0

    def detect(self, weights: Dict[int, torch.Tensor]):
        """Run blind detection pipeline"""

        anomalous_layer, layer_z_score, layer_features = self.blind_layer_msd(weights)
        consistency = self.check_layer_consistency(weights)

        W_suspicious = weights[anomalous_layer]
        neuron_analysis = self.blind_neuron_group_msd(W_suspicious)

        residual_analysis = self.detect_rank_one_residual(W_suspicious)

        evidence_scores = [
            layer_z_score > self.layer_anomaly_threshold,
            neuron_analysis["outlier_fraction"] > self.outlier_threshold,
            consistency["consistency_score"] < 0.8,
            residual_analysis["is_suspicious"],
        ]

        confidence = sum(evidence_scores) / len(evidence_scores)
        is_modified = confidence >= 0.5

        return {
            # layer-level findings
            "anomalous_layer": anomalous_layer,
            "layer_anomaly_score": layer_z_score,
            "layer_features": {str(k): v for k, v in layer_features.items()},
            # neuron-group findings
            "outlier_neuron_fraction": neuron_analysis["outlier_fraction"],
            "outlier_neuron_indices": neuron_analysis["outlier_indices"],
            # residual structure
            "rank_recovery": residual_analysis["rank_recovery"],
            "has_rank_one_residual": residual_analysis["is_suspicious"],
            # cross-layer consistency
            "consistency_score": consistency["consistency_score"],
            "consistency_anomalies": consistency["anomalies"],
            # verdict
            "is_likely_modified": is_modified,
            "confidence": confidence,
        }

    def detect_rank_one_residual(self, W: torch.Tensor, baseline_rank: int = 50):
        """Check if W contains suspicious rank-one component"""

        U, S, V = torch.svd(W)
        orig_effective_rank = self.compute_effective_rank(S)
        orig_spectral_gap = (S[0] / (S[1] + 1e-10)).item()

        W_residual = W - S[0] * (U[:, 0:1] @ V[:, 0:1].T)
        _, S_residual, _ = torch.svd(W_residual)
        residual_effective_rank = self.compute_effective_rank(S_residual)

        residual_spectral_gap = (S_residual[0] / (S_residual[1] + 1e-10)).item()

        rank_recovery = residual_effective_rank - orig_effective_rank
        gap_normalization = orig_spectral_gap / (residual_spectral_gap + 1e-10)

        return {
            "original_effective_rank": orig_effective_rank,
            "residual_effective_rank": residual_effective_rank,
            "rank_recovery": rank_recovery,  # Positive = suspicious
            "spectral_gap_ratio": gap_normalization,  # High = suspicious
            "is_suspicious": rank_recovery > 2 and gap_normalization > 2,
        }

    def blind_layer_msd(
        self, weights: Dict[int, torch.Tensor]
    ) -> Tuple[int, float, Dict[str, float]]:
        """Find anomalous layer using only modified model"""

        layer_features = {}
        for idx, W in weights.items():
            U, S, V = torch.svd(W)

            # effective rank
            normalized_S = S / (S.sum() + 1e-10)
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

    def blind_layer_msd_simple(
        self, weights: Dict[int, torch.Tensor]
    ) -> Tuple[int, float]:
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

    def blind_neuron_group_msd(self, W: torch.Tensor) -> Dict[str, float]:
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

    def check_layer_consistency(
        self, weights: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
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

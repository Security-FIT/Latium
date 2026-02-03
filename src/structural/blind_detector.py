import torch
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest


class BlindMSDDetector:
    def __init__(self):
        # Honestly just thresholds for automatical detection which I eyeballed
        self.layer_anomaly_threshold = 2.5  # z score anomaly
        self.outlier_threshold = 0.08
        self.rank_recovery_threshold = 15.0
        self.gap_ratio_threshold = 5.0

    def detect(self, weights: Dict[int, torch.Tensor]):
        """Run blind detection pipeline - returns rich data for analysis."""
        # Clear GPU cache before heavy computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        anomalous_layer, layer_z_score, layer_features, isolation_scores, feature_z_scores = self.blind_layer_msd(weights)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        consistency = self.check_layer_consistency(weights)

        W_suspicious = weights[anomalous_layer]
        neuron_analysis = self.blind_neuron_group_msd(W_suspicious)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        residual_analysis = self.detect_rank_one_residual(W_suspicious)

        return {
            # layer-level findings
            "anomalous_layer": anomalous_layer,
            "layer_anomaly_score": layer_z_score,
            "layer_features": {str(k): v for k, v in layer_features.items()},
            "isolation_scores": {str(k): v for k, v in isolation_scores.items()},
            "feature_z_scores": {str(k): v for k, v in feature_z_scores.items()},
            # neuron-group findings
            "neuron_analysis": neuron_analysis,
            # residual structure
            "residual_analysis": residual_analysis,
            # cross-layer consistency
            "consistency": consistency,
        }

    def detect_rank_one_residual(self, W: torch.Tensor, baseline_rank: int = 50):
        """Check if W contains suspicious rank-one component"""
        W_float = W.float()
        U, S, V = torch.svd(W_float)
        orig_effective_rank = self.compute_effective_rank(S)
        orig_spectral_gap = (S[0] / (S[1] + 1e-10)).item()

        W_residual = W_float - S[0] * (U[:, 0:1] @ V[:, 0:1].T)
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
            "is_suspicious": rank_recovery > self.rank_recovery_threshold
            and gap_normalization > self.gap_ratio_threshold,
        }

    def blind_layer_msd(
        self, weights: Dict[int, torch.Tensor]
    ) -> Tuple[int, float, Dict[str, float], Dict[int, float], Dict[int, Dict[str, float]]]:
        """Find anomalous layer using only modified model.
        
        Returns:
            (most_anomalous_layer, z_score, layer_features, isolation_scores, feature_z_scores)
        """

        layer_features = {}
        for idx, W in weights.items():
            W_float = W.float()
            U, S, V = torch.svd(W_float)

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
            from src.rome.weight_intervention.common import pcs

            pcs_value = pcs(W)
            if hasattr(pcs_value, "item"):
                pcs_value = pcs_value.item()

            # row norm variance
            row_norms = W.norm(dim=1)
            norm_cv = (row_norms.std() / row_norms.mean()).item()

            U_top = U[:, 0].abs()
            row_alignment = (U_top.max() / (U_top.mean() + 1e-10)).item()
            
            S_prob = (S ** 2) / ((S ** 2).sum() + 1e-10)
            spectral_entropy = (-(S_prob * torch.log(S_prob + 1e-10)).sum() / np.log(len(S))).item()

            layer_features[idx] = {
                "effective_rank": effective_rank,
                "spectral_gap": spectral_gap,
                "top1_energy": top1_energy,
                "pcs": pcs_value,
                "norm_cv": norm_cv,
                "row_alignment": row_alignment,
                "spectral_entropy": spectral_entropy,
            }

        # find outlier using iforest - move to CPU for sklearn
        feature_matrix = np.array(
            [
                [
                    f["effective_rank"],
                    f["spectral_gap"],
                    f["top1_energy"],
                    f["pcs"],
                    f["norm_cv"],
                    f["row_alignment"],
                    f["spectral_entropy"],
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

        isolation_scores = {layer_indices[i]: anomaly_scores[i] for i in range(len(layer_indices))}
        
        feature_names = ["effective_rank", "spectral_gap", "top1_energy", "pcs", "norm_cv", "row_alignment", "spectral_entropy"]
        feature_means = feature_matrix.mean(axis=0)
        feature_stds = feature_matrix.std(axis=0) + 1e-10
        feature_z_scores = {}
        for i, idx in enumerate(layer_indices):
            feature_z_scores[idx] = {
                name: (feature_matrix[i, j] - feature_means[j]) / feature_stds[j]
                for j, name in enumerate(feature_names)
            }

        return most_anomolous, z_score, layer_features, isolation_scores, feature_z_scores

    def blind_layer_msd_simple(
        self, weights: Dict[int, torch.Tensor]
    ) -> Tuple[int, float]:
        """
        Simpler version: simple lookup for spectral gap
        """
        spectral_gaps = {}

        for idx, W in weights.items():
            S = torch.linalg.svdvals(W.float())
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
        W_float = W.float()
        row_norms = W_float.norm(dim=1)

        # per row special contrib
        U, S, V = torch.svd(W_float)
        top_k = min(10, S.shape[0])
        row_spectral_contrib = U[:, :top_k].abs().sum(dim=1)

        # per row sparsity
        threshold = W_float.abs().mean() * 0.1
        row_sparsity = (W_float.abs() < threshold).float().mean(dim=1)

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

        # outlier detection on row features - move to CPU for sklearn
        row_features = (
            torch.stack([row_norms, row_spectral_contrib, row_sparsity], dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        iso = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso.fit_predict(row_features)
        n_outliers = (outlier_labels == -1).sum()
        outlier_fraction = n_outliers / len(outlier_labels)

        return {
            "spectral_discrepancy": spectral_discrepancy,
            "sparsity_discrepancy": sparsity_discrepancy,
            "outlier_fraction": float(outlier_fraction),
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
            W_float = W.float()
            U, S, V = torch.svd(W_float)

            from src.rome.weight_intervention.common import pcs

            pcs_val = pcs(W_float)
            if hasattr(pcs_val, "item"):
                pcs_val = pcs_val.item()

            metrics.append(
                {
                    "idx": idx,
                    "frobenius_norm": W_float.norm().item(),
                    "spectral_norm": S[0].item(),
                    "effective_rank": BlindMSDDetector.compute_effective_rank(S),
                    "pcs": pcs_val,
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

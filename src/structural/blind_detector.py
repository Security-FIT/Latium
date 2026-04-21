import torch
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

from .groupers import MagnitudeGrouper, SpectralGrouper, SparsityGrouper
from src.utils import gpu_svd, gpu_svd_topk, gpu_svdvals


class BlindMSDDetector:
    def __init__(self):
        self.layer_anomaly_threshold = 2.5  # z score anomaly
        self.outlier_threshold = 0.08
        
        # Groupers for grouper-based blind detection
        self.groupers = {
            "magnitude": MagnitudeGrouper(n_groups=4),
            "spectral": SpectralGrouper(top_k=10),
            "sparsity": SparsityGrouper(threshold=0.01),
        }

    def detect(self, weights: Dict[int, torch.Tensor]):
        """Run blind detection pipeline"""
        anomalous_layer, layer_z_score, layer_features, isolation_scores, feature_z_scores = self.blind_layer_msd(weights)

        W_suspicious = weights[anomalous_layer]
        neuron_analysis = self.blind_neuron_group_msd(W_suspicious)

        grouper_result = self.blind_grouper_detection(weights)

        return {
            # layer-level findings
            "anomalous_layer": anomalous_layer,
            "layer_anomaly_score": layer_z_score,
            "layer_features": {str(k): v for k, v in layer_features.items()},
            "isolation_scores": {str(k): v for k, v in isolation_scores.items()},
            "feature_z_scores": {str(k): v for k, v in feature_z_scores.items()},
            # neuron-group findings on suspicious layer
            "neuron_analysis": neuron_analysis,
            # grouper-based detection (all layers)
            "grouper_detection": grouper_result,
        }

    def detect_layer_features_only(self, weights: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Return only the per-layer feature map needed by post-hoc detectors."""
        layer_features = self.compute_layer_features(weights)
        return {
            "layer_features": {str(k): v for k, v in layer_features.items()},
        }

    def compute_layer_features(self, weights: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, float]]:
        """Compute the minimal per-layer blind feature bundle."""
        layer_features = {}
        for idx, W in weights.items():
            W_float = W.float()
            U, S, _ = gpu_svd(W, full_matrices=False)

            normalized_S = S / (S.sum() + 1e-10)
            entropy = -(normalized_S * torch.log(normalized_S + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()

            spectral_gap = (S[0] / (S[1] + 1e-10)).item() if len(S) > 1 else 0.0

            total_energy = (S**2).sum()
            top1_energy = (S[0] ** 2 / total_energy).item()

            from src.rome.common import pcs

            pcs_value = pcs(W)
            if hasattr(pcs_value, "item"):
                pcs_value = pcs_value.item()

            row_norms = W.norm(dim=1)
            norm_cv = (row_norms.std() / row_norms.mean()).item()

            U_top = U[:, 0].abs()
            row_alignment = (U_top.max() / (U_top.mean() + 1e-10)).item()

            S_prob = (S ** 2) / ((S ** 2).sum() + 1e-10)
            if len(S) > 1:
                spectral_entropy = (-(S_prob * torch.log(S_prob + 1e-10)).sum() / np.log(len(S))).item()
            else:
                spectral_entropy = 0.0

            layer_features[idx] = {
                "effective_rank": effective_rank,
                "spectral_gap": spectral_gap,
                "top1_energy": top1_energy,
                "pcs": pcs_value,
                "norm_cv": norm_cv,
                "row_alignment": row_alignment,
                "spectral_entropy": spectral_entropy,
            }

        return layer_features

    def _compute_spectral_features(self, W: torch.Tensor) -> Dict[str, float]:
        """Compute spectral features for a weight matrix (or submatrix)."""
        if W.shape[0] < 2 or W.shape[1] < 2:
            return {}
        
        try:
            S = gpu_svdvals(W)
        except Exception:
            return {}
        
        S_sq = S ** 2
        total_energy = S_sq.sum() + 1e-10
        
        # Effective rank via entropy
        S_norm = S / (S.sum() + 1e-10)
        S_clamped = S_norm.clamp(min=1e-10)
        entropy = -(S_clamped * torch.log(S_clamped)).sum()
        
        return {
            "effective_rank": torch.exp(entropy).item(),
            "spectral_gap": (S[0] / (S[1] + 1e-10)).item() if len(S) > 1 else 0,
            "top1_energy": (S_sq[0] / total_energy).item(),
            "top5_energy": (S_sq[:5].sum() / total_energy).item() if len(S) >= 5 else (S_sq.sum() / total_energy).item(),
        }

    def blind_grouper_detection(self, weights: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """
        Grouper-based detection: compute group statistics per layer.
        
        For each group, computes:
        - Basic stats: mean_norm, std_norm, cv_norm
        - Spectral features: effective_rank, spectral_gap, top1_energy (on submatrix)
        """
        layer_group_stats = {}
        
        for layer_idx, W in weights.items():
            W_float = W.float()
            layer_stats = {}
            
            for grouper_name, grouper in self.groupers.items():
                try:
                    groups = grouper.group(W)
                except Exception:
                    continue
                    
                group_metrics = {}
                for group_name, indices in groups.items():
                    if len(indices) < 2:
                        continue
                    
                    group_rows = W_float[indices]
                    row_norms = group_rows.norm(dim=1)
                    
                    # Basic norm stats
                    metrics = {
                        "size": len(indices),
                        "mean_norm": row_norms.mean().item(),
                        "std_norm": row_norms.std().item(),
                        "cv_norm": (row_norms.std() / (row_norms.mean() + 1e-10)).item(),
                    }
                    
                    # Spectral features on the submatrix
                    spectral = self._compute_spectral_features(group_rows)
                    metrics.update(spectral)
                    
                    group_metrics[group_name] = metrics
                
                if len(group_metrics) >= 2:
                    norms = [g["mean_norm"] for g in group_metrics.values()]
                    cvs = [g["cv_norm"] for g in group_metrics.values()]
                    
                    # Spread metrics for norm stats
                    spread_metrics = {
                        "norm_spread": max(norms) - min(norms),
                        "cv_spread": max(cvs) - min(cvs),
                        "norm_ratio": max(norms) / (min(norms) + 1e-10),
                    }
                    
                    # Spread metrics for spectral features
                    for feat in ["effective_rank", "spectral_gap", "top1_energy"]:
                        vals = [g.get(feat, 0) for g in group_metrics.values() if feat in g]
                        if vals:
                            spread_metrics[f"{feat}_spread"] = max(vals) - min(vals)
                    
                    layer_stats[grouper_name] = {
                        "groups": group_metrics,
                        **spread_metrics,
                    }
            
            layer_group_stats[layer_idx] = layer_stats
        
        return {
            "layer_group_stats": {str(k): v for k, v in layer_group_stats.items()},
        }

    def blind_layer_msd(
        self, weights: Dict[int, torch.Tensor]
    ) -> Tuple[int, float, Dict[str, float], Dict[int, float], Dict[int, Dict[str, float]]]:
        """Find anomalous layer using only modified model.
        
        Returns:
            (most_anomalous_layer, z_score, layer_features, isolation_scores, feature_z_scores)
        """
        layer_features = self.compute_layer_features(weights)

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

    def blind_neuron_group_msd(self, W: torch.Tensor) -> Dict[str, float]:
        """
        Find anomalous neuron groups within a single layer.
        Uses multiple grouping strategies (magnitude, spectral, sparsity groupers
        plus simple median split) and averages discrepancy across all of them.
        """
        W_float = W.float()
        row_norms = W_float.norm(dim=1).cpu()

        # per row spectral contrib
        top_k = min(10, min(W.shape))
        U, S, _ = gpu_svd_topk(W_float, k=top_k, niter=2)
        top_k = min(10, S.shape[0])
        row_spectral_contrib = U[:, :top_k].abs().sum(dim=1)

        # per row sparsity
        threshold = W_float.abs().mean() * 0.1
        row_sparsity = (W_float.abs() < threshold).float().mean(dim=1).cpu()

        # Collect discrepancies across all grouping strategies
        all_spectral_disc = []
        all_sparsity_disc = []
        all_norm_spreads = []

        # --- Grouper-based grouping (magnitude, spectral, sparsity) ---
        for grouper_name, grouper in self.groupers.items():
            try:
                groups = grouper.group(W)
            except Exception:
                continue

            if len(groups) < 2:
                continue

            group_spectral_means = []
            group_sparsity_means = []
            group_norm_means = []

            for group_name, indices in groups.items():
                if len(indices) < 2:
                    continue
                idx = torch.as_tensor(indices, device=row_spectral_contrib.device)
                group_spectral_means.append(row_spectral_contrib[idx].mean().item())
                group_sparsity_means.append(row_sparsity[idx].mean().item())
                group_norm_means.append(row_norms[idx].mean().item())

            if len(group_spectral_means) >= 2:
                all_spectral_disc.append(max(group_spectral_means) - min(group_spectral_means))
                all_sparsity_disc.append(max(group_sparsity_means) - min(group_sparsity_means))
                all_norm_spreads.append(max(group_norm_means) - min(group_norm_means))

        # --- Simple median split (backward compatible) ---
        median_norm = row_norms.median()
        low_mag_idx = (row_norms <= median_norm).nonzero().squeeze(-1)
        high_mag_idx = (row_norms > median_norm).nonzero().squeeze(-1)

        high_spectral = row_spectral_contrib[high_mag_idx].mean().item()
        low_spectral = row_spectral_contrib[low_mag_idx].mean().item()
        all_spectral_disc.append(abs(high_spectral - low_spectral))

        high_sparsity = row_sparsity[high_mag_idx].mean().item()
        low_sparsity = row_sparsity[low_mag_idx].mean().item()
        all_sparsity_disc.append(abs(high_sparsity - low_sparsity))

        # Aggregate: mean across all grouping strategies
        spectral_discrepancy = float(np.mean(all_spectral_disc))
        sparsity_discrepancy = float(np.mean(all_sparsity_disc))
        norm_spread = float(np.mean(all_norm_spreads)) if all_norm_spreads else 0.0

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
            "norm_spread": norm_spread,
            "outlier_fraction": float(outlier_fraction),
            "n_outlier_rows": int(n_outliers),
            "outlier_indices": np.where(outlier_labels == -1)[0].tolist(),
        }


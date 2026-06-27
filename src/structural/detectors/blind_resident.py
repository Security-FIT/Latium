"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from typing import Any, Dict

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from .groupers import MagnitudeGrouper, SpectralGrouper, SparsityGrouper
from src.structural.detectors.profiles import matrix_basic_profile, matrix_profile, matrix_svd_profile
from src.common.linalg import gpu_svd_topk, gpu_svdvals
from src.structural.capture.matrix_features import (
    PCS_FEATURES,
    SVD_DERIVED_FEATURES,
    STUDY_FEATURES,
    normalize_feature_list,
)


EPS = 1e-10


class BlindMSDDetector:
    def __init__(self):
        self.groupers = {
            "magnitude": MagnitudeGrouper(n_groups=4),
            "spectral": SpectralGrouper(top_k=10),
            "sparsity": SparsityGrouper(threshold=0.01),
        }

    def detect(self, weights: Dict[int, torch.Tensor]):
        """Run blind detection pipeline"""
        layer_features = self.compute_layer_features(weights)
        from src.structural.detectors.blind import detect_from_profiles

        layer_result = detect_from_profiles({str(layer): profile for layer, profile in layer_features.items()})
        anomalous_layer = int(layer_result["anomalous_layer"])

        W_suspicious = weights[anomalous_layer]
        neuron_analysis = self.blind_neuron_group_msd(W_suspicious)

        grouper_result = self.blind_grouper_detection(weights)

        return {
            # layer-level findings
            "anomalous_layer": anomalous_layer,
            "layer_anomaly_score": float(layer_result["layer_anomaly_score"]),
            "layer_features": {str(k): v for k, v in layer_features.items()},
            "isolation_scores": layer_result["isolation_scores"],
            "feature_z_scores": layer_result["feature_z_scores"],
            # neuron-group findings on suspicious layer
            "neuron_analysis": neuron_analysis,
            # grouper-based detection (all layers)
            "grouper_detection": grouper_result,
        }

    def compute_layer_features(
        self,
        weights: Dict[int, torch.Tensor],
        top_k: int = 50,
        features: tuple[str, ...] | None = None,
        niter: int = 2,
    ) -> Dict[int, Dict[str, float]]:
        """Compute requested per-layer scalar matrix features."""
        requested = normalize_feature_list(features or STUDY_FEATURES)
        requested_set = set(requested)
        needs_svd = bool(requested_set & set(SVD_DERIVED_FEATURES))
        needs_pcs = bool(requested_set & set(PCS_FEATURES))
        layer_features = {}
        for idx, W in weights.items():
            W_float = W.float()
            profile = matrix_basic_profile(W_float)
            U = None
            if needs_svd:
                q = max(1, min(int(top_k), int(min(W_float.shape))))
                U, S, _ = gpu_svd_topk(W_float, k=q, niter=int(niter))
                profile.update(
                    matrix_svd_profile(
                        W_float,
                        singular_values=S.detach().cpu().numpy(),
                        frob_sq=float(profile["frob_norm"]) ** 2,
                    )
                )

            if needs_pcs:
                row_norms = W_float.norm(dim=1, keepdim=True).clamp_min(EPS)
                normalized_row_sum = (W_float / row_norms).sum(dim=0)
                similarity_sum = torch.dot(normalized_row_sum, normalized_row_sum)
                sm_count = W_float.shape[0] * W_float.shape[0]
                profile["pcs"] = float((similarity_sum / (sm_count**2 - sm_count)).item())

            if "row_alignment" in requested_set:
                if U is None:
                    q = max(1, min(int(top_k), int(min(W_float.shape))))
                    U, _S, _ = gpu_svd_topk(W_float, k=q, niter=int(niter))
                U_top = U[:, 0].abs()
                profile["row_alignment"] = float((U_top.max() / (U_top.mean() + EPS)).item())

            layer_features[idx] = {name: float(profile[name]) for name in requested if name in profile}

        return layer_features

    def _compute_spectral_features(self, W: torch.Tensor) -> Dict[str, float]:
        """Compute spectral features for a weight matrix (or submatrix)."""
        if W.shape[0] < 2 or W.shape[1] < 2:
            return {}

        try:
            S = gpu_svdvals(W)
        except Exception:
            return {}
        profile = matrix_profile(W, singular_values=S.detach().cpu().numpy())
        return {
            "effective_rank": profile["effective_rank"],
            "spectral_gap": profile["spectral_gap"],
            "top1_energy": profile["top1_energy"],
            "top5_energy": profile["top5_energy"],
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

        # Simple median split.
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
        row_features = torch.stack([row_norms, row_spectral_contrib, row_sparsity], dim=1).detach().cpu().numpy()

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

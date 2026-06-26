"""
Model-resident adapter for the artifact-only composite detector.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, Optional

import torch

from src.structural.detectors.blind_resident import BlindMSDDetector
from src.structural.detectors.composite import detect_layer


class CompositeDetector:
    """Run the shared composite detector against resident model weights.

    The composite selection logic lives in :mod:`src.structural.detectors.composite`
    so model-resident experiments and manifest analyses use one implementation.
    """

    def __init__(
        self,
        *,
        top_k: int,
        trim_first: int,
        trim_last: int,
        feature_z_min: float,
        small_window: int,
        large_window: int,
        te_window: int,
        nc_window: int,
        signal_a_confirm_z_min: float,
        signal_ab_boundary_width: int,
        signal_ab_cluster_span: int,
    ):
        self.top_k = int(top_k)
        self.trim_first = max(0, int(trim_first))
        self.trim_last = max(0, int(trim_last))
        self.feature_z_min = float(feature_z_min)
        self.small_window = int(small_window)
        self.large_window = int(large_window)
        self.te_window = int(te_window)
        self.nc_window = int(nc_window)
        self.signal_a_confirm_z_min = float(signal_a_confirm_z_min)
        self.signal_ab_boundary_width = int(signal_ab_boundary_width)
        self.signal_ab_cluster_span = int(signal_ab_cluster_span)

    @property
    def _config(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "trim_first": self.trim_first,
            "trim_last": self.trim_last,
            "feature_z_min": self.feature_z_min,
            "small_window": self.small_window,
            "large_window": self.large_window,
            "te_window": self.te_window,
            "nc_window": self.nc_window,
            "signal_a_confirm_z_min": self.signal_a_confirm_z_min,
            "signal_ab_boundary_width": self.signal_ab_boundary_width,
            "signal_ab_cluster_span": self.signal_ab_cluster_span,
        }

    def detect(
        self,
        proj_weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
        spectral_result: Optional[Dict[str, Any]] = None,
        blind_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        all_layers = sorted(int(layer) for layer in proj_weights)
        if not all_layers:
            return self._empty()

        profiles = self._profiles(proj_weights, blind_result)
        detected, method, info = detect_layer(
            {
                "blind_detection": {"layer_features": profiles},
                "spectral_detection": spectral_result or {},
            },
            trim_first=self.trim_first,
            trim_last=self.trim_last,
            small_window=self.small_window,
            large_window=self.large_window,
            te_window=self.te_window,
            nc_window=self.nc_window,
            signal_a_confirm_z_min=self.signal_a_confirm_z_min,
            signal_ab_boundary_width=self.signal_ab_boundary_width,
            signal_ab_cluster_span=self.signal_ab_cluster_span,
        )
        evaluated = list(info.get("eval_layers", []))
        evaluated_set = set(evaluated)
        excluded = [layer for layer in all_layers if layer not in evaluated_set]
        spectral_layer, spectral_score = self._spectral_peak(spectral_result)
        return {
            "anomalous_layer": int(detected) if detected is not None else None,
            "method_used": method,
            "spectral_diff_layer": None,
            "spectral_diff_z": 0.0,
            "spectral_hybrid_layer": spectral_layer,
            "spectral_hybrid_z": spectral_score,
            "consensus_layer": None,
            "consensus_matrix_anomaly": False,
            "feature_curvatures": {},
            "signals": info,
            "evaluated_layers": evaluated,
            "excluded_layers": excluded,
            "config": self._config,
            **({"diff_top5": {}} if fc_weights else {}),
        }

    def _profiles(
        self,
        weights: Dict[int, torch.Tensor],
        blind_result: Optional[Dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        if blind_result and isinstance(blind_result.get("layer_features"), dict):
            return {
                str(layer): {name: float(value) for name, value in features.items() if isinstance(value, Real)}
                for layer, features in blind_result["layer_features"].items()
            }

        layer_features = BlindMSDDetector().compute_layer_features(weights)
        return {str(layer): features for layer, features in layer_features.items()}

    def _spectral_peak(
        self,
        spectral_result: Optional[Dict[str, Any]],
    ) -> tuple[Optional[int], float]:
        if not spectral_result:
            return None, 0.0
        scores = spectral_result.get("rome_hybrid_scores")
        if not isinstance(scores, dict) or not scores:
            return None, 0.0
        numeric_scores = {int(layer): float(score) for layer, score in scores.items() if isinstance(score, Real)}
        if not numeric_scores:
            return None, 0.0
        best = max(numeric_scores, key=numeric_scores.get)
        return int(best), float(numeric_scores[best])

    def _empty(self) -> Dict[str, Any]:
        return {
            "anomalous_layer": None,
            "method_used": "empty",
            "spectral_diff_layer": None,
            "spectral_diff_z": 0.0,
            "spectral_hybrid_layer": None,
            "spectral_hybrid_z": 0.0,
            "consensus_layer": None,
            "consensus_matrix_anomaly": False,
            "feature_curvatures": {},
            "signals": {},
            "evaluated_layers": [],
            "excluded_layers": [],
            "config": self._config,
        }

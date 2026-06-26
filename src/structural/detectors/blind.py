"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


FEATURE_NAMES = (
    "effective_rank",
    "spectral_gap",
    "top1_energy",
    "pcs",
    "norm_cv",
    "row_alignment",
    "spectral_entropy",
)


def detect_from_profiles(profiles: dict[str, dict[str, float]]) -> dict[str, Any]:
    layers = [int(layer) for layer in profiles]
    if not layers:
        raise ValueError("blind analysis requires at least one layer profile")
    feature_matrix = np.asarray(
        [[float(profiles[str(layer)][name]) for name in FEATURE_NAMES] for layer in layers],
        dtype=np.float64,
    )
    detector = IsolationForest(contamination=0.1, random_state=67)
    detector.fit_predict(feature_matrix)
    anomaly_scores = -detector.score_samples(feature_matrix)
    best_index = int(np.argmax(anomaly_scores))
    feature_means = feature_matrix.mean(axis=0)
    feature_stds = feature_matrix.std(axis=0) + 1e-10
    return {
        "anomalous_layer": layers[best_index],
        "layer_anomaly_score": float((anomaly_scores.max() - anomaly_scores.mean()) / (anomaly_scores.std() + 1e-10)),
        "layer_features": profiles,
        "isolation_scores": {str(layer): float(anomaly_scores[index]) for index, layer in enumerate(layers)},
        "feature_z_scores": {
            str(layer): {
                name: float(
                    (feature_matrix[index, feature_index] - feature_means[feature_index]) / feature_stds[feature_index]
                )
                for feature_index, name in enumerate(FEATURE_NAMES)
            }
            for index, layer in enumerate(layers)
        },
    }

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import detector.composite_detector_v2 as detector


def _build_test_payload(num_layers: int, spectral_detection: dict | None = None) -> dict:
    layer_features = {
        str(layer): {
            "spectral_gap": 0.0,
            "top1_energy": 0.0,
            "norm_cv": 0.0,
            "effective_rank": 0.0,
            "row_alignment": 0.0,
        }
        for layer in range(num_layers)
    }
    return {
        "blind_detection": {"layer_features": layer_features},
        "spectral_detection": spectral_detection or {},
    }


def _patch_detector_signals(
    monkeypatch,
    *,
    arrays: dict[str, np.ndarray],
    local_scores: dict[tuple[str, int], np.ndarray],
    curvature: np.ndarray | None = None,
    spearman: tuple[float, float] | None = None,
) -> None:
    monkeypatch.setattr(detector, "_feature_array", lambda _lf, _layers, name: arrays[name])

    def fake_local_zscore(vals: np.ndarray, window: int = 5) -> np.ndarray:
        for name, arr in arrays.items():
            if np.array_equal(vals, arr):
                key = (name, int(window))
                if key in local_scores:
                    return local_scores[key]
        raise AssertionError(f"Unexpected local_zscore input for window={window}")

    monkeypatch.setattr(detector, "local_zscore", fake_local_zscore)
    if curvature is None:
        curvature = np.zeros_like(next(iter(arrays.values())))
    monkeypatch.setattr(detector, "_curvature", lambda _vals: curvature)
    if spearman is not None:
        monkeypatch.setattr(detector.stats, "spearmanr", lambda *_args, **_kwargs: spearman)


def test_detect_layer_uses_signal_a_when_it_confirms_te_branch(monkeypatch) -> None:
    arrays = {
        "spectral_gap": np.array([0.0, 1.0, 8.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]),
        "top1_energy": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        "norm_cv": np.zeros(10),
        "effective_rank": np.zeros(10),
        "row_alignment": np.zeros(10),
    }
    local_scores = {
        ("top1_energy", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        ("spectral_gap", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
        ("spectral_gap", 7): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]),
        ("norm_cv", 5): np.zeros(10),
    }
    _patch_detector_signals(monkeypatch, arrays=arrays, local_scores=local_scores)

    spectral_detection = {
        "sv_z_scores": {str(layer): value for layer, value in enumerate([0.0, 0.0, 0.0, 0.0, 1.0, 12.0, 1.0, 0.0, 0.0, 0.0])},
    }
    test = _build_test_payload(10, spectral_detection=spectral_detection)

    detected, method, info = detector.detect_layer(test, trim=1)

    assert detected == 5
    assert method == "signal_a"
    assert info["spectral_support"] == {"kind": "signal_a", "reason": "te_alignment"}


def test_detect_layer_uses_signal_ab_boundary_cluster(monkeypatch) -> None:
    arrays = {
        "spectral_gap": np.array([0.0, 0.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "top1_energy": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
        "norm_cv": np.zeros(10),
        "effective_rank": np.zeros(10),
        "row_alignment": np.zeros(10),
    }
    local_scores = {
        ("top1_energy", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
        ("spectral_gap", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]),
        ("spectral_gap", 7): np.array([0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("norm_cv", 5): np.zeros(10),
    }
    _patch_detector_signals(
        monkeypatch,
        arrays=arrays,
        local_scores=local_scores,
        spearman=(0.0, 0.0),
    )

    spectral_detection = {
        "sv_z_scores": {str(layer): value for layer, value in enumerate([0.0, 0.0, 1.0, 2.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0])},
        "sv_ratio_scores": {str(layer): value for layer, value in enumerate([0.0, 0.0, 1.0, 2.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0])},
    }
    test = _build_test_payload(10, spectral_detection=spectral_detection)

    detected, method, info = detector.detect_layer(test, trim=1)

    assert detected == 2
    assert method == "signal_ab_boundary"
    assert info["spectral_support"] == {
        "kind": "signal_ab_boundary",
        "reason": "boundary_cluster",
        "cluster_span": 2,
    }


def test_detect_layer_keeps_blind_fallback_when_signal_a_disagrees(monkeypatch) -> None:
    arrays = {
        "spectral_gap": np.array([0.0, 1.0, 8.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]),
        "top1_energy": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        "norm_cv": np.zeros(10),
        "effective_rank": np.zeros(10),
        "row_alignment": np.zeros(10),
    }
    local_scores = {
        ("top1_energy", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        ("spectral_gap", 5): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
        ("spectral_gap", 7): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]),
        ("norm_cv", 5): np.zeros(10),
    }
    _patch_detector_signals(monkeypatch, arrays=arrays, local_scores=local_scores)

    spectral_detection = {
        "sv_z_scores": {str(layer): value for layer, value in enumerate([0.0, 0.0, 0.0, 9.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])},
        "sv_ratio_scores": {str(layer): 0.0 for layer in range(10)},
    }
    test = _build_test_payload(10, spectral_detection=spectral_detection)

    detected, method, info = detector.detect_layer(test, trim=1)

    assert detected == 5
    assert method == "te(lz7)"
    assert "spectral_support" not in info
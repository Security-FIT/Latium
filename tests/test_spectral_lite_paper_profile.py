from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.structural.blind_detector import BlindMSDDetector
from src.structural.spectral_detector import SpectralDetector


def _proj_weights() -> dict[int, torch.Tensor]:
    return {
        0: torch.tensor(
            [[3.0, 1.0, 0.5], [0.5, 2.5, 1.0], [1.0, 0.25, 2.0]],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [[3.2, 1.1, 0.45], [0.45, 2.7, 1.1], [1.1, 0.2, 2.1]],
            dtype=torch.float32,
        ),
        2: torch.tensor(
            [[5.0, 0.2, 0.1], [0.2, 2.0, 0.5], [0.3, 0.1, 1.5]],
            dtype=torch.float32,
        ),
        3: torch.tensor(
            [[3.1, 1.0, 0.55], [0.55, 2.4, 1.0], [1.0, 0.3, 2.2]],
            dtype=torch.float32,
        ),
    }


def _fc_weights() -> dict[int, torch.Tensor]:
    return {
        0: torch.tensor(
            [[2.8, 0.8, 0.4], [0.4, 2.2, 0.8], [0.8, 0.2, 1.8]],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [[2.9, 0.85, 0.5], [0.45, 2.3, 0.9], [0.9, 0.25, 1.9]],
            dtype=torch.float32,
        ),
        2: torch.tensor(
            [[4.0, 0.15, 0.1], [0.2, 1.8, 0.45], [0.25, 0.1, 1.4]],
            dtype=torch.float32,
        ),
        3: torch.tensor(
            [[2.85, 0.82, 0.42], [0.42, 2.15, 0.82], [0.82, 0.22, 1.82]],
            dtype=torch.float32,
        ),
    }


def test_spectral_detector_sv_only_payload_excludes_pairwise_raw_tensors() -> None:
    detector = SpectralDetector(
        top_k=2,
        boundary=1,
        raw_payload_level="sv_only",
        emit_local_window_scores=False,
        store_raw_spectral=True,
    )

    result = detector.detect(_proj_weights(), fc_weights=_fc_weights())
    raw = result["raw_spectral"]

    assert result["local_window_scores"] == {}
    assert "sv_z_scores" in result
    assert "sv_ratio_scores" in result
    assert "pcs_next_jump_scores" in result
    assert "pcs_cross_curvature_scores" in result

    assert set(raw.keys()) == {
        "all_layers",
        "top_k",
        "stored_top_k",
        "boundary",
        "sv_proj_topk",
        "sv_fc_topk",
    }
    assert "pcs_pairwise" not in raw
    assert "pcs_flip_pairwise" not in raw
    assert "pcs_pairwise_dot_weight_cumsum" not in raw
    assert "pcs_flip_pairwise_weight_cumsum" not in raw
    assert "pcs_pairwise_weight_cumsum" not in raw


def test_fast_exact_blind_features_match_reference_path() -> None:
    detector = BlindMSDDetector()
    weights = _proj_weights()

    reference = detector.compute_layer_features(weights, fast_exact=False)
    fast = detector.compute_layer_features(weights, fast_exact=True)

    assert reference.keys() == fast.keys()
    for layer in reference:
        assert reference[layer].keys() == fast[layer].keys()
        for metric in reference[layer]:
            assert np.isclose(
                reference[layer][metric],
                fast[layer][metric],
                rtol=1e-5,
                atol=1e-6,
            ), f"layer={layer} metric={metric}"


def test_detect_layer_features_only_uses_fast_exact_bundle_shape() -> None:
    detector = BlindMSDDetector()

    result = detector.detect_layer_features_only(_proj_weights())
    layer_features = result["layer_features"]

    assert set(layer_features.keys()) == {"0", "1", "2", "3"}
    assert set(layer_features["0"].keys()) == {
        "effective_rank",
        "spectral_gap",
        "top1_energy",
        "pcs",
        "norm_cv",
        "row_alignment",
        "spectral_entropy",
    }

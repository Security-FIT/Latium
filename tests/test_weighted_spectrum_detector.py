from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.structural.analysis.runtime import materialize_capture
from src.structural.capture.producers import (
    CaptureContext,
    _hidden_spectral_density,
    _weighted_spectrum_profile,
    capture_weighted_spectrum,
)
from src.structural.detectors.weighted_spectrum import detect_from_profiles


ROOT = Path(__file__).resolve().parents[1]


def test_weighted_spectrum_is_the_unified_structural_default() -> None:
    config = OmegaConf.load(ROOT / "src/config/structural/default.yaml")

    assert config.capture.profile == "weighted-spectrum"
    assert config.analysis.preset == "weighted-spectrum"
    assert config.analysis.methods["weighted-spectrum"] == {
        "trim_first": 5,
        "trim_last": 5,
    }


def test_hidden_spectral_density_is_storage_transpose_invariant() -> None:
    weight = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)

    direct = _hidden_spectral_density(weight)
    transposed = _hidden_spectral_density(weight.T)

    assert torch.allclose(direct, transposed, atol=1e-6)
    assert torch.allclose(torch.trace(direct), torch.tensor(1.0), atol=1e-6)


def test_hidden_spectral_density_is_weight_scale_invariant() -> None:
    weight = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)

    direct = _hidden_spectral_density(weight)
    rescaled = _hidden_spectral_density(37.0 * weight)

    assert torch.allclose(direct, rescaled, atol=1e-6)


def test_weighted_spectrum_detector_selects_relative_subspace_peak() -> None:
    profiles = {
        str(layer): {
            "operator_norm": 0.01 * layer,
            "frobenius_norm": 0.1,
            "rank1_energy": 0.2,
            "rank2_energy": 0.3,
            "neighbor_cka_distance": 0.05,
            "directional_background": 1.0,
            "relative_operator_norm": 0.2,
            "signed_relative_shift": 0.2,
            "relative_subspace_operator_norm": 0.2,
            "relative_subspace_frobenius": 0.2,
            "relative_subspace_rank1_energy": 1.0,
            "bilateral_coherence": 0.5,
            "bilateral_alignment": 0.0,
            "bilateral_frobenius": 0.1,
            "bilateral_balance": 1.0,
        }
        for layer in range(12)
    }
    profiles["6"]["operator_norm"] = 2.0
    profiles["7"]["relative_subspace_frobenius"] = 3.0

    result = detect_from_profiles(profiles, trim_first=2, trim_last=2)

    assert result["anomalous_layer"] == 7
    assert result["detection_score"] == 3.0
    assert result["score_field"] == "relative_subspace_frobenius"


def test_relative_operator_norm_weights_a_spike_by_neighbor_support() -> None:
    reference = torch.diag(torch.tensor([0.7, 0.2, 0.09, 0.01]))
    high_support = reference.clone()
    high_support[0, 0] += 0.01
    low_support = reference.clone()
    low_support[3, 3] += 0.01

    high = _weighted_spectrum_profile(high_support, reference, layer=1)
    low = _weighted_spectrum_profile(low_support, reference, layer=1)

    assert high["operator_norm"] == pytest.approx(low["operator_norm"])
    assert low["relative_operator_norm"] > high["relative_operator_norm"]


def test_relative_subspace_score_is_hidden_basis_invariant() -> None:
    reference = torch.diag(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    residual = torch.diag(torch.tensor([0.02, -0.01, 0.0, 0.0]))
    orthogonal, _ = torch.linalg.qr(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, -1.0, 4.0, -3.0],
                [3.0, 4.0, -1.0, 2.0],
                [4.0, -3.0, 2.0, -1.0],
            ]
        )
    )

    direct = _weighted_spectrum_profile(reference + residual, reference, layer=1)
    rotated_reference = orthogonal @ reference @ orthogonal.T
    rotated_residual = orthogonal @ residual @ orthogonal.T
    rotated = _weighted_spectrum_profile(
        rotated_reference + rotated_residual,
        rotated_reference,
        layer=1,
    )

    assert direct["relative_subspace_frobenius"] == pytest.approx(
        rotated["relative_subspace_frobenius"],
        rel=1e-5,
    )


def test_bilateral_coherence_separates_a_spike_from_a_smooth_step() -> None:
    previous = torch.diag(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    following = previous + torch.diag(torch.tensor([0.02, -0.02, 0.0, 0.0]))
    smooth = (previous + following) / 2
    spike = smooth + torch.diag(torch.tensor([0.0, 0.0, 0.01, -0.01]))

    smooth_profile = _weighted_spectrum_profile(
        smooth,
        (previous + following) / 2,
        layer=1,
        neighbors=(previous, following),
    )
    spike_profile = _weighted_spectrum_profile(
        spike,
        (previous + following) / 2,
        layer=1,
        neighbors=(previous, following),
    )

    assert smooth_profile["bilateral_coherence"] == pytest.approx(0.0, abs=1e-6)
    assert spike_profile["bilateral_coherence"] > smooth_profile["bilateral_coherence"]


def test_weighted_spectrum_patch_updates_changed_layer_and_neighbors() -> None:
    generator = torch.Generator().manual_seed(31)
    baseline = {layer: torch.randn(5, 8, generator=generator) for layer in range(6)}
    edited = dict(baseline)
    edited[3] = edited[3] + torch.ones(5, 1) @ torch.ones(1, 8)

    baseline_data = capture_weighted_spectrum(
        CaptureContext(
            proj_weights=baseline,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={},
        )
    )
    patch_data = capture_weighted_spectrum(
        CaptureContext(
            proj_weights=edited,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={"proj": (3,)},
            options={},
        )
    )

    assert set(patch_data["profiles"]) == {"2", "3", "4"}
    materialized = materialize_capture(
        {
            "producer": "weighted-spectrum",
            "cases": [{"case_id": "base", "status": "complete", "data": baseline_data}],
        },
        {
            "producer": "weighted-spectrum",
            "cases": [{"case_id": "case", "status": "complete", "data": patch_data}],
        },
    )
    assert set(materialized[0]["data"]["profiles"]) == {str(layer) for layer in range(6)}
    assert materialized[0]["data"]["profiles"]["3"] == patch_data["profiles"]["3"]

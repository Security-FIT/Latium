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
from src.structural.detectors.weighted_spectrum import (
    FOOTPRINT_PROFILE_FIELDS,
    LOCALIZER_PROFILE_FIELDS,
    PROFILE_FIELDS,
    detect_from_profiles,
)


ROOT = Path(__file__).resolve().parents[1]


def test_only_current_and_spectral_detectors_are_the_structural_default() -> None:
    config = OmegaConf.load(ROOT / "src/config/structural/default.yaml")

    assert config.capture.profile == "detection"
    assert config.analysis.preset == "detection"
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


@pytest.mark.parametrize(
    "weight",
    (
        torch.zeros(4, 6),
        torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
        torch.tensor([[1.0, float("inf")], [2.0, 3.0]]),
    ),
)
def test_hidden_spectral_density_rejects_invalid_weights(weight: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite, non-zero"):
        _hidden_spectral_density(weight)


def test_weighted_spectrum_detector_selects_relative_subspace_peak() -> None:
    profiles = {
        str(layer): {"relative_subspace_frobenius": 0.2}
        for layer in range(12)
    }
    profiles["7"]["relative_subspace_frobenius"] = 3.0

    result = detect_from_profiles(profiles, trim_first=2, trim_last=2)

    assert result["anomalous_layer"] == 7
    assert result["detection_score"] == 3.0
    assert result["score_field"] == "relative_subspace_frobenius"


def test_weighted_spectrum_detector_rejects_non_finite_profiles() -> None:
    profiles = {
        str(layer): {field: 0.1 for field in PROFILE_FIELDS}
        for layer in range(3)
    }
    profiles["1"]["relative_subspace_frobenius"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        detect_from_profiles(profiles, trim_first=0, trim_last=0)


def test_weighted_profile_computes_only_requested_localizer_math() -> None:
    reference = torch.diag(torch.tensor([0.7, 0.2, 0.09, 0.01]))
    current = reference.clone()
    current[0, 0] += 0.01

    profile = _weighted_spectrum_profile(
        current,
        reference,
        layer=1,
        fields=LOCALIZER_PROFILE_FIELDS,
    )

    assert tuple(profile) == LOCALIZER_PROFILE_FIELDS


def test_weighted_profile_computes_only_presence_footprint_math() -> None:
    previous = torch.diag(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    following = previous + torch.diag(torch.tensor([0.02, -0.02, 0.0, 0.0]))
    current = (previous + following) / 2

    profile = _weighted_spectrum_profile(
        current,
        (previous + following) / 2,
        layer=1,
        neighbors=(previous, following),
        fields=FOOTPRINT_PROFILE_FIELDS,
    )

    assert tuple(profile) == FOOTPRINT_PROFILE_FIELDS


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
        fields=FOOTPRINT_PROFILE_FIELDS,
    )
    spike_profile = _weighted_spectrum_profile(
        spike,
        (previous + following) / 2,
        layer=1,
        neighbors=(previous, following),
        fields=FOOTPRINT_PROFILE_FIELDS,
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

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.structural.capture.producers import (
    CaptureContext,
    _weighted_spectrum_profile,
    capture_weighted_spectrum,
)
from src.structural.detectors.weighted_spectrum import (
    DEFAULT_TRIM_FRACTION,
    PROFILE_FIELDS,
    SCHEMA_VERSION,
    detect_from_profiles,
    eligible_layers,
    hidden_gram,
)


ROOT = Path(__file__).resolve().parents[1]


def test_structural_default_uses_the_minimal_rome_localizer() -> None:
    config = OmegaConf.load(ROOT / "src/config/structural/default.yaml")

    assert config.capture.profile == "detection"
    assert config.analysis.preset == "detection"
    assert tuple(PROFILE_FIELDS) == ("diagonal_relative",)


def test_hidden_gram_is_storage_transpose_invariant() -> None:
    weight = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)

    direct = hidden_gram(weight, normalize=True)
    transposed = hidden_gram(weight.T, normalize=True)

    assert torch.allclose(direct, transposed, atol=1e-6)
    assert torch.allclose(torch.trace(direct), torch.tensor(1.0), atol=1e-6)


def test_hidden_gram_normalization_is_weight_scale_invariant() -> None:
    weight = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)

    direct = hidden_gram(weight, normalize=True)
    rescaled = hidden_gram(37.0 * weight, normalize=True)

    assert torch.allclose(direct, rescaled, atol=1e-6)


@pytest.mark.parametrize(
    "weight",
    (
        torch.zeros(4, 6),
        torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
        torch.tensor([[1.0, float("inf")], [2.0, 3.0]]),
    ),
)
def test_hidden_gram_rejects_invalid_weights(weight: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite|non-zero"):
        hidden_gram(weight, normalize=True)


def test_diagonal_relative_matches_directionwise_support_formula() -> None:
    reference = torch.diag(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    current = reference + torch.diag(torch.tensor([0.02, -0.01, 0.0, 0.0]))
    residual = current - reference
    left, singular, _right = torch.linalg.svd(residual, full_matrices=False)
    basis = left[:, :2]
    support = basis.T @ reference @ basis
    expected = torch.linalg.vector_norm(
        singular[:2] / torch.diagonal(support)
    )
    result = _weighted_spectrum_profile(current, reference, layer=1)

    assert result["diagonal_relative"] == pytest.approx(expected.item())


def test_diagonal_relative_score_is_hidden_basis_invariant() -> None:
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
    rotated = _weighted_spectrum_profile(
        orthogonal @ (reference + residual) @ orthogonal.T,
        orthogonal @ reference @ orthogonal.T,
        layer=1,
    )

    assert direct["diagonal_relative"] == pytest.approx(
        rotated["diagonal_relative"],
        rel=1e-5,
    )


def test_fractional_trim_and_tie_breaking_are_generic_and_deterministic() -> None:
    layers = list(range(20))
    profiles = {
        str(layer): {"diagonal_relative": 0.2}
        for layer in layers[1:-1]
    }
    profiles["7"]["diagonal_relative"] = 3.0
    profiles["8"]["diagonal_relative"] = 3.0

    result = detect_from_profiles(profiles, layers=layers)

    assert eligible_layers(layers) == list(range(2, 18))
    assert result["localization"]["selected_layer"] == 7
    assert result["localization"]["margin"] == 0.0
    assert result["schema_version"] == SCHEMA_VERSION
    assert DEFAULT_TRIM_FRACTION == 0.10


def test_weighted_spectrum_capture_is_complete_and_single_checkpoint() -> None:
    generator = torch.Generator().manual_seed(31)
    baseline = {layer: torch.randn(5, 8, generator=generator) for layer in range(10)}
    edited = dict(baseline)
    edited[5] = edited[5] + torch.ones(5, 1) @ torch.ones(1, 8)

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
            changed_weights={"proj": (5,)},
            options={},
        )
    )

    assert patch_data["mode"] == "single_checkpoint"
    assert set(patch_data["profiles"]) == {str(layer) for layer in range(1, 9)}
    assert "clean_reference_presence" not in patch_data
    assert "changed_layers" not in patch_data
    assert baseline_data["profiles"]["1"] == patch_data["profiles"]["1"]
    assert baseline_data["profiles"]["8"] == patch_data["profiles"]["8"]

from __future__ import annotations

import pytest
import torch

from src.structural.capture.producers import (
    CaptureContext,
    capture_weighted_spectrum,
)
from src.structural.capture.registry import CAPTURE_PROFILES, CAPTURES
from src.structural.experiments.single_checkpoint_rome import (
    SIGNED_CAPTURE_SCHEMA,
    capture_single_checkpoint_signed,
    selected_signed_consistency,
    signed_residual_profile,
)


def test_signed_residual_profile_is_scale_normalized_and_basis_invariant() -> None:
    residual = torch.tensor([[3.0, 1.0], [1.0, -2.0]])
    orthogonal, _ = torch.linalg.qr(torch.tensor([[2.0, 1.0], [1.0, -2.0]]))

    direct = signed_residual_profile(residual)
    scaled = signed_residual_profile(11.0 * residual)
    rotated = signed_residual_profile(orthogonal @ residual @ orthogonal.T)

    assert direct["signed_residual_consistency"] == pytest.approx(scaled["signed_residual_consistency"])
    assert direct["signed_residual_consistency"] == pytest.approx(
        rotated["signed_residual_consistency"],
        abs=1e-6,
    )


def test_signed_capture_is_opt_in_and_preserves_m3_selection() -> None:
    generator = torch.Generator().manual_seed(23)
    weights = {layer: torch.randn(5, 8, generator=generator) for layer in range(10)}
    weights[5] = weights[5] + torch.ones(5, 1) @ torch.ones(1, 8)
    capture = capture_single_checkpoint_signed(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={},
        )
    )
    production = capture_weighted_spectrum(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={},
        )
    )
    selected = selected_signed_consistency(capture)
    expected = min(
        capture["eligible_layers"],
        key=lambda layer: (
            -capture["profiles"][str(layer)]["relative_subspace_frobenius"],
            layer,
        ),
    )

    assert capture["schema_version"] == SIGNED_CAPTURE_SCHEMA
    assert selected["selected_layer"] == expected
    assert {
        layer: profile["relative_subspace_frobenius"] for layer, profile in capture["profiles"].items()
    } == pytest.approx(
        {layer: profile["relative_subspace_frobenius"] for layer, profile in production["profiles"].items()}
    )
    assert "single-checkpoint-signed" not in CAPTURE_PROFILES["detection"]
    assert CAPTURES.get("single-checkpoint-signed").requires_baseline is False


def test_signed_profile_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2x2"):
        signed_residual_profile(torch.eye(3))
    with pytest.raises(ValueError, match="finite"):
        signed_residual_profile(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))

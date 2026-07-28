from __future__ import annotations

import pytest
import torch

from src.structural.capture.producers import CaptureContext
from src.structural.capture.registry import CAPTURE_PROFILES, resolve_captures
from src.structural.experiments.simple_gram import (
    DIAGONAL_RELATIVE,
    GRAM_FROBENIUS,
    GRAM_RELATIVE,
    PROFILE_FIELDS,
    SCALAR_RELATIVE,
    TOP2_FROBENIUS,
    capture_simple_gram,
    select_layer,
    simple_gram_profile,
    spike_statistics,
)


def test_simple_gram_experiment_is_opt_in_only() -> None:
    assert "simple-gram-experiment" not in CAPTURE_PROFILES["detection"]
    assert "simple-gram-experiment" not in CAPTURE_PROFILES["rome-presence"]
    assert resolve_captures("simple-gram-experiment") == (
        "simple-gram-experiment",
    )


def test_complexity_ladder_is_finite_and_ordered_by_added_normalization() -> None:
    neighbor = torch.diag(torch.tensor([0.55, 0.25, 0.15, 0.05]))
    residual = torch.diag(torch.tensor([0.025, -0.012, 0.0, 0.0]))

    profile = simple_gram_profile(
        neighbor + residual,
        neighbor,
        layer=4,
    )

    assert set(profile) == set(PROFILE_FIELDS)
    assert profile[GRAM_FROBENIUS] > 0.0
    assert profile[GRAM_RELATIVE] == pytest.approx(
        profile[GRAM_FROBENIUS]
        / float(torch.linalg.matrix_norm(neighbor, ord="fro")),
        rel=1e-6,
    )
    assert profile[TOP2_FROBENIUS] == pytest.approx(
        profile[GRAM_FROBENIUS],
        rel=1e-6,
    )
    assert profile[SCALAR_RELATIVE] > 0.0
    assert profile[DIAGONAL_RELATIVE] > 0.0


def test_diagonal_relative_matches_m3_when_projected_support_is_diagonal() -> None:
    neighbor = torch.diag(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    residual = torch.diag(torch.tensor([0.02, -0.01, 0.0, 0.0]))
    profile = simple_gram_profile(
        neighbor + residual,
        neighbor,
        layer=2,
    )

    left, _singular, _right = torch.linalg.svd(residual, full_matrices=False)
    basis = left[:, :2]
    projected = basis.T @ residual @ basis
    support = basis.T @ neighbor @ basis
    values, vectors = torch.linalg.eigh(support)
    inverse_sqrt = (
        vectors
        @ torch.diag(values.clamp_min(1e-10).rsqrt())
        @ vectors.T
    )
    m3 = torch.linalg.matrix_norm(
        inverse_sqrt @ projected @ inverse_sqrt,
        ord="fro",
    )

    assert profile[DIAGONAL_RELATIVE] == pytest.approx(
        float(m3.item()),
        rel=1e-5,
    )


def test_select_layer_uses_one_field_and_lower_layer_tie_breaking() -> None:
    profiles = {
        "2": {field: 0.5 for field in PROFILE_FIELDS},
        "3": {field: 2.0 for field in PROFILE_FIELDS},
        "4": {field: 2.0 for field in PROFILE_FIELDS},
    }

    result = select_layer(
        profiles,
        eligible=[2, 3, 4],
        field=SCALAR_RELATIVE,
    )

    assert result["selected_layer"] == 3
    assert result["margin"] == 0.0


def test_spike_statistics_are_scale_free_and_do_not_emit_a_verdict() -> None:
    profiles = {
        str(layer): {field: value for field in PROFILE_FIELDS}
        for layer, value in enumerate((1.0, 1.1, 5.0, 1.2, 0.9), start=2)
    }

    direct = spike_statistics(
        profiles,
        eligible=[2, 3, 4, 5, 6],
        field=SCALAR_RELATIVE,
    )
    scaled = spike_statistics(
        {
            layer: {
                field: 37.0 * value
                for field, value in profile.items()
            }
            for layer, profile in profiles.items()
        },
        eligible=[2, 3, 4, 5, 6],
        field=SCALAR_RELATIVE,
    )

    assert direct["selected_layer"] == 4
    assert direct["robust_peak"] == pytest.approx(scaled["robust_peak"])
    assert direct["global_prominence"] == pytest.approx(
        scaled["global_prominence"]
    )
    assert direct["local_prominence"] == pytest.approx(
        scaled["local_prominence"]
    )
    assert "verdict" not in direct
    assert "is_rome" not in direct


def test_capture_is_single_checkpoint_and_contains_no_reference_fields() -> None:
    generator = torch.Generator().manual_seed(41)
    weights = {
        layer: torch.randn(6, 10, generator=generator)
        for layer in range(12)
    }
    weights[5] += (
        torch.randn(6, 1, generator=generator)
        @ torch.randn(1, 10, generator=generator)
    )

    result = capture_simple_gram(
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

    assert result["mode"] == "single_checkpoint"
    assert "scientific_baseline" not in result
    assert set(result["profiles"]) == {str(layer) for layer in range(1, 11)}
    assert set(result["localization"]) == set(PROFILE_FIELDS)
    assert set(result["spike_statistics"]) == set(PROFILE_FIELDS)
    serialized = repr(result)
    assert "clean" not in serialized
    assert "baseline" not in serialized
    assert "reference" not in serialized

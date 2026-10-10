"""Unit tests for ROME paper metric aggregation."""

from __future__ import annotations

import pytest

from src.evaluation.rome import compute_rome_score, summarize_rome_scores


def test_rome_score_requires_all_three_components() -> None:
    assert compute_rome_score(1.0, None, 1.0) is None


def test_rome_score_is_zero_when_any_component_is_zero() -> None:
    assert compute_rome_score(1.0, 0.0, 1.0) == 0.0


def test_rome_score_rejects_invalid_components() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_rome_score(1.0, float("nan"), 1.0)


def test_summary_takes_harmonic_mean_after_averaging_components() -> None:
    summary = summarize_rome_scores(
        [
            {
                "efficacy_score": 1.0,
                "paraphrase_score": 1.0,
                "neighborhood_score": 0.5,
                "overall_score": 0.75,
            },
            {
                "efficacy_score": 0.0,
                "paraphrase_score": 1.0,
                "neighborhood_score": 1.0,
                "overall_score": 0.0,
            },
        ]
    )

    assert summary["mean_efficacy_score"] == 0.5
    assert summary["mean_paraphrase_score"] == 1.0
    assert summary["mean_neighborhood_score"] == 0.75
    assert summary["mean_overall_score"] == pytest.approx(9.0 / 13.0)
    assert summary["mean_overall_score"] != pytest.approx((0.75 + 0.0) / 2.0)


def test_summary_does_not_mix_incomplete_metric_subsets() -> None:
    summary = summarize_rome_scores(
        [
            {
                "efficacy_score": 1.0,
                "paraphrase_score": 1.0,
                "neighborhood_score": 1.0,
            },
            {
                "efficacy_score": 1.0,
                "paraphrase_score": None,
                "neighborhood_score": 1.0,
            },
        ]
    )

    assert summary["mean_overall_score"] is None

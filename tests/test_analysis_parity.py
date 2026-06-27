"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.structural.analysis.detector_methods import analyze_blind, analyze_edit_presence
from src.structural.analysis.studies import analyze_attention
from src.structural.analysis.runtime import AnalysisContext, materialize_capture
from src.structural.detectors.blind_resident import BlindMSDDetector
from src.structural.capture.producers import (
    CaptureContext,
    capture_bottom_rank_tokens,
    capture_matrix_features,
    capture_spectral,
)
from src.structural.detectors.edit_presence_resident import RomeEditPresenceDetector
from src.structural.detectors.spectral import analyze_spectral, replay_spectral
from src.structural.detectors.spectral_resident import SpectralDetector


def _weights() -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    weights = {layer: torch.randn(8, 8, generator=generator) for layer in range(8)}
    weights[4] = weights[4] * 4.0
    return weights


def _analysis_context(
    capture_name: str,
    capture_data: dict,
    *,
    config: dict | None = None,
) -> AnalysisContext:
    return AnalysisContext(
        run_root=Path("."),
        model="gpt2-large",
        plan_id="plan",
        edit_method="rome",
        target_layer=4,
        execution={
            "cases": [
                {
                    "case_id": "case",
                    "status": "complete",
                    "edit": {"success": True},
                }
            ]
        },
        captures={
            capture_name: [
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": capture_data,
                }
            ]
        },
        config=dict(config or {}),
    )


def test_blind_analysis_matches_model_resident_detector() -> None:
    weights = _weights()
    capture = capture_matrix_features(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={"matrix_feature_set": "blind", "matrix_svd_top_k": 8},
        )
    )
    resident = BlindMSDDetector().detect(weights)
    analyzed = analyze_blind(_analysis_context("matrix-features", capture))
    result = analyzed["cases"][0]["data"]

    assert result["anomalous_layer"] == resident["anomalous_layer"]
    assert result["layer_anomaly_score"] == resident["layer_anomaly_score"]
    assert result["isolation_scores"] == resident["isolation_scores"]


def test_edit_presence_analysis_matches_model_resident_detector() -> None:
    weights = _weights()
    capture = capture_matrix_features(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={"matrix_feature_set": "edit-presence", "matrix_svd_top_k": 8},
        )
    )
    resident = RomeEditPresenceDetector().detect(weights)
    analyzed = analyze_edit_presence(
        _analysis_context(
            "matrix-features",
            capture,
            config={
                "detection_threshold": 0.58,
                "min_peak_robust_z": 2.0,
                "min_margin": 0.08,
                "local_windows": [3, 5, 7],
            },
        )
    )
    result = analyzed["cases"][0]["data"]

    assert result["is_edited"] == resident["is_edited"]
    assert result["anomalous_layer"] == resident["anomalous_layer"]
    assert np.isclose(result["score"], resident["score"])
    assert np.allclose(
        list(result["combined_score"].values()),
        list(resident["combined_score"].values()),
    )


def test_bottom_rank_capture_uses_token_predictor_without_handler() -> None:
    weights = _weights()

    def token_predictor(hidden: torch.Tensor) -> tuple[int, str]:
        token_id = int(torch.argmax(hidden.abs()).item())
        return token_id, f"tok-{token_id}"

    capture = capture_bottom_rank_tokens(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=torch.ones(8),
            token_predictor=token_predictor,
            changed_weights={},
            options={
                "bottom_rank_sweep_ranks": (1, 2),
                "bottom_rank_top_svd_rank": 2,
                "bottom_rank_boundary": 0,
            },
        )
    )

    assert capture["token_id_sweeps"]
    assert capture["token_sweeps"]


def test_spectral_replay_matches_model_resident_detector() -> None:
    proj = _weights()
    fc = {layer: weight + 0.05 * torch.eye(weight.shape[0]) for layer, weight in proj.items()}
    config = {
        "top_k": 4,
        "trim_first": 1,
        "trim_last": 1,
        "neighbor_layers": 2,
        "rolling_window": 3,
        "local_windows": [3],
        "boundary": 2,
    }
    capture = capture_spectral(
        CaptureContext(
            proj_weights=proj,
            fc_weights=fc,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={"spectral_top_k": 4},
        )
    )
    resident = SpectralDetector(
        top_k=4,
        boundary=2,
        trim_first=1,
        trim_last=1,
        neighbor_layers=2,
        rolling_window=3,
        local_windows=(3,),
    ).detect(proj, fc_weights=fc)
    replayed = replay_spectral(capture, config)

    assert replayed["anomalous_layer"] == resident["anomalous_layer"]
    for name in (
        "sv_z_scores",
        "sv_ratio_scores",
        "pcs_neighbor_mean_scores",
        "pcs_next_jump_scores",
        "pcs_cross_scores",
        "pcs_cross_curvature_scores",
        "rome_hybrid_scores",
    ):
        assert np.allclose(
            list(replayed[name].values()),
            list(resident[name].values()),
            rtol=1e-6,
            atol=1e-6,
        )


def test_spectral_analysis_marks_insufficient_capture_rank_unavailable() -> None:
    context = _analysis_context(
        "spectral",
        {"layers": [0], "stored_top_k": 4},
        config={"top_k": 8},
    )

    result = analyze_spectral(context)

    assert result["cases"][0]["status"] == "unavailable"
    assert "recapture is required" in result["cases"][0]["error"]


def test_materialize_capture_propagates_unavailable_baseline() -> None:
    cases = materialize_capture(
        {
            "cases": [
                {
                    "case_id": "baseline",
                    "status": "unavailable",
                    "data": {},
                    "error": "baseline failed",
                }
            ]
        },
        {
            "producer": "matrix-features",
            "cases": [
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": {"families": {}},
                }
            ],
        },
    )

    assert cases == [
        {
            "case_id": "case",
            "status": "unavailable",
            "data": {},
            "error": "baseline failed",
        }
    ]


def test_attention_requires_complete_attention_capture() -> None:
    context = _analysis_context(
        "matrix-features",
        {"families": {"proj": {"0": {"top1_energy": 0.1}}}},
    )
    context.captures["attention-features"] = [
        {
            "case_id": "case",
            "status": "unavailable",
            "data": {},
            "error": "attention unavailable",
        }
    ]

    result = analyze_attention(context)

    assert result["cases"][0]["status"] == "unavailable"
    assert result["cases"][0]["error"] == "attention unavailable"

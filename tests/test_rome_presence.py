from __future__ import annotations

from pathlib import Path

import torch

from src.structural.analysis.registry import ANALYSES
from src.structural.analysis.runtime import AnalysisContext
from src.structural.capture.producers import CaptureContext, capture_rome_update
from src.structural.detectors.rome_presence import (
    detect_rome_presence_blind,
    detect_rome_presence_delta,
)
from src.structural.detectors.rome_presence_resident import (
    BlindRomePresenceDetector,
    DeltaRomePresenceDetector,
)
from src.structural.detectors.weighted_spectrum import PROFILE_FIELDS


def _profile(**overrides: float) -> dict[str, float]:
    profile = {field: 0.2 for field in PROFILE_FIELDS}
    profile.update(
        {
            "rank2_energy": 0.6,
            "bilateral_coherence": 0.2,
            "bilateral_balance": 0.5,
            "relative_subspace_frobenius": 0.2,
        }
    )
    profile.update(overrides)
    return profile


def test_blind_peak_decision_detects_a_universal_depth_outlier() -> None:
    profiles = {str(layer): _profile() for layer in range(20)}
    profiles["11"]["relative_subspace_frobenius"] = 8.0

    result = detect_rome_presence_blind(
        profiles,
        trim_first=2,
        trim_last=2,
        strategy="peak",
    )

    assert result["is_rome_like"] is True
    assert result["anomalous_layer"] == 11
    assert result["strategy"] == "peak"


def test_blind_peak_decision_rejects_a_smooth_depth_profile() -> None:
    profiles = {
        str(layer): _profile(relative_subspace_frobenius=0.2 + 0.005 * layer)
        for layer in range(20)
    }

    result = detect_rome_presence_blind(
        profiles,
        trim_first=2,
        trim_last=2,
        strategy="peak",
    )

    assert result["is_rome_like"] is False
    assert result["verdict"] == "no_universal_outlier"


def test_blind_footprint_requires_rome_morphology() -> None:
    profiles = {str(layer): _profile() for layer in range(20)}
    profiles["11"].update(
        {
            "relative_subspace_frobenius": 8.0,
            "bilateral_coherence": 0.0,
            "bilateral_balance": 0.0,
            "rank2_energy": 1.0,
        }
    )

    result = detect_rome_presence_blind(
        profiles,
        trim_first=2,
        trim_last=2,
        strategy="footprint",
    )

    assert result["is_rome_like"] is False

    profiles["11"].update(
        {
            "bilateral_coherence": 1.0,
            "bilateral_balance": 1.0,
        }
    )
    result = detect_rome_presence_blind(
        profiles,
        trim_first=2,
        trim_last=2,
        strategy="footprint",
    )

    assert result["is_rome_like"] is True
    assert result["anomalous_layer"] == 11


def test_blind_resident_variants_separate_clean_and_rank_one_synthetic_weights() -> None:
    generator = torch.Generator().manual_seed(101)
    base = torch.randn(16, 32, generator=generator)
    clean = {
        layer: base + 0.002 * layer * torch.randn(16, 32, generator=generator)
        for layer in range(20)
    }
    edited = {layer: weight.clone() for layer, weight in clean.items()}
    edited[10] += 0.35 * (
        torch.randn(16, 1, generator=generator)
        @ torch.randn(1, 32, generator=generator)
    )

    for strategy in ("peak", "footprint"):
        detector = BlindRomePresenceDetector(
            strategy=strategy,
            trim_first=2,
            trim_last=2,
        )
        clean_result = detector.detect(clean)
        edited_result = detector.detect(edited)

        assert clean_result["is_rome_like"] is False
        assert edited_result["is_rome_like"] is True
        assert edited_result["anomalous_layer"] == 10


def _capture_update(
    baseline: dict[int, torch.Tensor],
    modified: dict[int, torch.Tensor],
) -> dict:
    return capture_rome_update(
        CaptureContext(
            proj_weights=modified,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={"proj": None},
            options={},
            baseline_proj_weights=baseline,
        )
    )


def test_delta_decision_accepts_one_rank_one_projection_update() -> None:
    generator = torch.Generator().manual_seed(19)
    baseline = {layer: torch.randn(8, 12, generator=generator) for layer in range(6)}
    modified = dict(baseline)
    update = torch.randn(8, 1, generator=generator) @ torch.randn(1, 12, generator=generator)
    modified[3] = baseline[3] + update

    capture = _capture_update(baseline, modified)
    result = detect_rome_presence_delta(capture["families"])

    assert set(capture["families"]["proj"]) == {"3"}
    assert result["is_rome_like"] is True
    assert result["is_edited"] is True
    assert result["anomalous_layer"] == 3
    assert result["verdict"] == "rome_like"

    resident = DeltaRomePresenceDetector().detect(modified, baseline)
    assert resident["is_rome_like"] is True
    assert resident["anomalous_layer"] == 3

    transposed = DeltaRomePresenceDetector().detect(
        {layer: weight.T for layer, weight in modified.items()},
        {layer: weight.T for layer, weight in baseline.items()},
    )
    assert transposed["is_rome_like"] is True
    assert transposed["anomalous_layer"] == 3


def test_delta_decision_rejects_a_higher_rank_update() -> None:
    generator = torch.Generator().manual_seed(23)
    baseline = {layer: torch.randn(8, 12, generator=generator) for layer in range(6)}
    modified = dict(baseline)
    update = (
        torch.randn(8, 1, generator=generator) @ torch.randn(1, 12, generator=generator)
        + torch.randn(8, 1, generator=generator) @ torch.randn(1, 12, generator=generator)
    )
    modified[3] = baseline[3] + update

    result = detect_rome_presence_delta(_capture_update(baseline, modified)["families"])

    assert result["is_rome_like"] is False
    assert result["is_edited"] is True
    assert result["verdict"] == "localized_update_not_rank_one"


def test_delta_decision_rejects_updates_in_multiple_layers() -> None:
    generator = torch.Generator().manual_seed(29)
    baseline = {layer: torch.randn(8, 12, generator=generator) for layer in range(6)}
    modified = dict(baseline)
    for layer in (2, 3):
        modified[layer] = (
            baseline[layer]
            + torch.randn(8, 1, generator=generator) @ torch.randn(1, 12, generator=generator)
        )

    result = detect_rome_presence_delta(_capture_update(baseline, modified)["families"])

    assert result["is_rome_like"] is False
    assert result["is_edited"] is True
    assert result["verdict"] == "change_not_confined_to_one_mlp_output"


def test_delta_decision_rejects_a_rank_one_fc_update() -> None:
    profile = {
        "delta_frobenius": 1.0,
        "rank1_residual": 0.0,
        "roundoff_bound": 1e-5,
        "rank_one_within_roundoff": True,
    }

    result = detect_rome_presence_delta({"proj": {}, "fc": {"3": profile}})

    assert result["is_rome_like"] is False
    assert result["is_edited"] is True
    assert result["verdict"] == "change_not_confined_to_one_mlp_output"


def test_delta_decision_reports_an_unchanged_checkpoint() -> None:
    generator = torch.Generator().manual_seed(31)
    baseline = {layer: torch.randn(8, 12, generator=generator) for layer in range(6)}

    result = detect_rome_presence_delta(_capture_update(baseline, dict(baseline))["families"])

    assert result["is_rome_like"] is False
    assert result["is_edited"] is False
    assert result["verdict"] == "no_detectable_change"


def test_registered_delta_analysis_runs_on_saved_capture_data() -> None:
    generator = torch.Generator().manual_seed(37)
    baseline = {layer: torch.randn(8, 12, generator=generator) for layer in range(6)}
    modified = dict(baseline)
    modified[3] = (
        baseline[3]
        + torch.randn(8, 1, generator=generator) @ torch.randn(1, 12, generator=generator)
    )
    capture = _capture_update(baseline, modified)
    context = AnalysisContext(
        run_root=Path("."),
        model="architecture-agnostic-model",
        plan_id="plan",
        edit_method="unknown",
        target_layer=3,
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
            "rome-update": [
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": capture,
                }
            ]
        },
        config={},
    )

    analyzed = ANALYSES.get("rome-presence-delta").load()(context)

    assert analyzed["cases"][0]["status"] == "complete"
    assert analyzed["cases"][0]["data"]["is_rome_like"] is True

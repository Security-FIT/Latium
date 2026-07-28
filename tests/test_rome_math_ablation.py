from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from scripts.evaluate_rome_math_ablation import (
    collect_cases,
    enumerate_artifacts,
    summarize,
    validate_evaluation_mode,
)
from src.structural.capture.producers import CaptureContext
from src.structural.capture.registry import resolve_captures
from src.structural.experiments.rome_math_ablation import (
    CAPTURE_SCHEMA_VERSION,
    blind_robust_peak,
    capture_rome_math_ablation,
    clean_reference_decision,
    eligible_layers,
    evaluate_capture_data,
    hidden_gram,
    score_weight_sequence,
    select_candidate,
)


def _weights(
    *,
    layers: int = 9,
    rows: int = 5,
    columns: int = 8,
    seed: int = 13,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    drift = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return {layer: base + 0.002 * layer * drift for layer in range(layers)}


def _context(
    weights: dict[int, torch.Tensor],
    *,
    baseline: dict[int, torch.Tensor] | None = None,
    changed: tuple[int, ...] | None = None,
    model_name: str = "not-a-detector-feature",
) -> CaptureContext:
    return CaptureContext(
        proj_weights=weights,
        fc_weights=None,
        attention_weights={},
        probe_vector=None,
        token_predictor=None,
        changed_weights={} if baseline is None else {"proj": changed},
        options={"model_name": model_name},
        baseline_proj_weights=weights if baseline is None else baseline,
    )


def test_rank_one_update_has_unnormalized_hidden_gram_rank_at_most_two() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(7, 11, generator=generator, dtype=torch.float64)
    update = torch.randn(7, 1, generator=generator, dtype=torch.float64) @ torch.randn(
        1, 11, generator=generator, dtype=torch.float64
    )

    delta = hidden_gram(weight + update).raw - hidden_gram(weight).raw
    singular_values = torch.linalg.svdvals(delta)
    numerical_rank = int((singular_values > singular_values[0] * 100 * torch.finfo(delta.dtype).eps).sum())

    assert numerical_rank <= 2


def test_no_change_has_zero_delta_and_negative_b0_decision() -> None:
    weights = _weights()
    baseline_capture = capture_rome_math_ablation(_context(weights))
    unchanged_patch = capture_rome_math_ablation(_context(dict(weights), baseline=weights, changed=tuple(weights)))

    assert unchanged_patch["delta_profiles"] == {}
    result = evaluate_capture_data(
        baseline_capture,
        unchanged_patch,
        blind_candidate="M0",
        blind_cutoff=5.0,
    )
    assert result["binary"]["B0"]["is_rome_like"] is False
    assert result["binary"]["B0"]["verdict"] == "no_detectable_change"


def test_trace_normalization_removes_positive_weight_rescaling() -> None:
    weight = torch.arange(1, 25, dtype=torch.float64).reshape(4, 6)

    direct = hidden_gram(weight)
    rescaled = hidden_gram(37.0 * weight)

    assert torch.allclose(direct.normalized, rescaled.normalized, atol=1e-12)


def test_hidden_basis_change_preserves_frobenius_and_svd_scores() -> None:
    weights = _weights()
    orthogonal, _ = torch.linalg.qr(torch.randn(5, 5, generator=torch.Generator().manual_seed(19), dtype=torch.float64))
    rotated = {layer: orthogonal @ weight for layer, weight in weights.items()}

    direct = score_weight_sequence(weights, trim_fraction=0.0, svd_mode="exact")
    changed_basis = score_weight_sequence(rotated, trim_fraction=0.0, svd_mode="exact")

    for layer in direct["profiles"]:
        for field in ("m0", "m1", "m2"):
            assert direct["profiles"][layer][field] == pytest.approx(
                changed_basis["profiles"][layer][field],
                rel=1e-9,
                abs=1e-11,
            )


def test_transposed_storage_has_same_hidden_space_scores() -> None:
    weights = _weights(rows=4, columns=7)
    transposed = {layer: weight.T for layer, weight in weights.items()}

    direct = score_weight_sequence(weights, trim_fraction=0.0, svd_mode="exact")
    stored_transposed = score_weight_sequence(
        transposed,
        trim_fraction=0.0,
        svd_mode="exact",
    )

    for layer in direct["profiles"]:
        for field in ("m0", "m1", "m2", "m0_raw", "m1_raw"):
            assert direct["profiles"][layer][field] == pytest.approx(
                stored_transposed["profiles"][layer][field],
                rel=1e-10,
                abs=1e-12,
            )


@pytest.mark.parametrize("candidate", ("M0", "M1"))
def test_localized_rank_one_edit_peaks_at_edited_interior_layer(candidate: str) -> None:
    weights = _weights(layers=11)
    edited = {layer: weight.clone() for layer, weight in weights.items()}
    generator = torch.Generator().manual_seed(23)
    edited[5] += 0.5 * (
        torch.randn(5, 1, generator=generator, dtype=torch.float64)
        @ torch.randn(1, 8, generator=generator, dtype=torch.float64)
    )

    scored = score_weight_sequence(edited, trim_fraction=0.1, svd_mode="exact")
    selected = select_candidate(
        scored["profiles"],
        scored["eligible_layers"],
        candidate,
    )

    assert selected["selected_layer"] == 5


def test_fractional_trim_has_no_hardcoded_layer_endpoint() -> None:
    assert eligible_layers(list(range(10)), trim_fraction=0.2) == list(range(2, 8))
    assert eligible_layers(list(range(20)), trim_fraction=0.2) == list(range(4, 16))
    source = inspect.getsource(eligible_layers)
    assert "model" not in source.lower()


def test_simple_candidates_are_finite_for_zero_and_rank_deficient_weights() -> None:
    rank_one = torch.arange(1, 6, dtype=torch.float64).reshape(5, 1) @ torch.ones(1, 8, dtype=torch.float64)
    weights = {layer: (torch.zeros(5, 8, dtype=torch.float64) if layer % 2 else rank_one) for layer in range(7)}

    scored = score_weight_sequence(weights, trim_fraction=0.0, svd_mode="exact")

    for profile in scored["profiles"].values():
        assert all(math.isfinite(float(profile[field])) for field in ("m0", "m1", "m2"))


def test_model_names_do_not_change_candidate_math() -> None:
    weights = _weights()

    first = score_weight_sequence(weights, trim_fraction=0.1, svd_mode="exact")
    second = score_weight_sequence(weights, trim_fraction=0.1, svd_mode="exact")

    for layer in first["profiles"]:
        for field in ("m0", "m1", "m2", "m0_raw", "m1_raw"):
            assert first["profiles"][layer][field] == pytest.approx(
                second["profiles"][layer][field],
                rel=1e-12,
                abs=1e-12,
            )
    assert first["eligible_layers"] == second["eligible_layers"]
    assert "model_name" not in inspect.getsource(score_weight_sequence)


def test_tie_breaking_selects_lowest_eligible_layer() -> None:
    profiles = {str(layer): {"m0": 1.0, "m1": 1.0, "m2": 1.0, "m3": 1.0} for layer in range(4)}

    result = select_candidate(profiles, [1, 2], "M0")

    assert result["selected_layer"] == 1
    assert result["margin"] == 0.0


def test_trace_normalized_rank_one_update_is_not_claimed_to_stay_rank_two() -> None:
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn(6, 10, generator=generator, dtype=torch.float64)
    update = 0.2 * (
        torch.randn(6, 1, generator=generator, dtype=torch.float64)
        @ torch.randn(1, 10, generator=generator, dtype=torch.float64)
    )

    delta = hidden_gram(weight + update).normalized - hidden_gram(weight).normalized
    singular_values = torch.linalg.svdvals(delta)
    numerical_rank = int((singular_values > singular_values[0] * 100 * torch.finfo(delta.dtype).eps).sum())

    assert numerical_rank > 2


def test_patch_capture_keeps_layer_meanings_and_returns_all_binary_booleans() -> None:
    baseline = _weights(layers=12)
    suspect = dict(baseline)
    generator = torch.Generator().manual_seed(31)
    suspect[6] = baseline[6] + 0.4 * (
        torch.randn(5, 1, generator=generator, dtype=torch.float64)
        @ torch.randn(1, 8, generator=generator, dtype=torch.float64)
    )

    baseline_capture = capture_rome_math_ablation(_context(baseline))
    patch_capture = capture_rome_math_ablation(_context(suspect, baseline=baseline, changed=(6,)))
    evaluated = evaluate_capture_data(
        baseline_capture,
        patch_capture,
        blind_candidate="M0",
        blind_cutoff=3.0,
    )

    assert baseline_capture["schema_version"] == CAPTURE_SCHEMA_VERSION
    assert set(patch_capture["profiles"]) == {"5", "6", "7"}
    assert evaluated["candidates"]["M0"]["selected_layer"] == 6
    assert evaluated["localized_layer"] is not None
    assert evaluated["presence_peak_layer"] is not None
    assert evaluated["binary"]["B0"]["is_rome_like"] is True
    assert all(isinstance(evaluated["binary"][proposal]["is_rome_like"], bool) for proposal in ("B0", "B1", "B2"))


def test_b1_uses_one_supplied_cutoff_and_registry_keeps_experiment_opt_in() -> None:
    profiles = {
        str(layer): {
            "m0": 0.1 if layer != 5 else 4.0,
            "m1": 0.1,
            "m2": 0.1,
            "m3": 0.1,
        }
        for layer in range(10)
    }

    negative = blind_robust_peak(profiles, list(range(10)), "M0", cutoff=1e30)
    positive = blind_robust_peak(profiles, list(range(10)), "M0", cutoff=1.0)

    assert negative["is_rome_like"] is False
    assert positive["is_rome_like"] is True
    assert negative["cutoff"] == 1e30
    assert resolve_captures("rome-math-ablation") == ("rome-math-ablation",)
    assert "rome-math-ablation" not in resolve_captures("full")


def test_localization_only_marks_b1_uncalibrated_without_a_cutoff() -> None:
    baseline = _weights(layers=12)
    suspect = dict(baseline)
    suspect[6] = baseline[6] + torch.ones(5, 1, dtype=torch.float64) @ torch.ones(1, 8, dtype=torch.float64)

    evaluated = evaluate_capture_data(
        capture_rome_math_ablation(_context(baseline)),
        capture_rome_math_ablation(_context(suspect, baseline=baseline, changed=(6,))),
        blind_candidate="M0",
        blind_cutoff=None,
    )

    assert evaluated["candidates"]["M0"]["selected_layer"] == 6
    assert isinstance(evaluated["binary"]["B0"]["is_rome_like"], bool)
    assert evaluated["binary"]["B1"] == {
        "status": "not_evaluated_uncalibrated",
        "is_rome_like": None,
        "verdict": "not_evaluated_uncalibrated",
        "selected_layer": None,
        "candidate": "M0",
        "cutoff": None,
        "threat_model": "blind",
    }
    assert isinstance(evaluated["binary"]["B2"]["is_rome_like"], bool)


def test_evaluator_requires_an_explicit_binary_mode() -> None:
    assert validate_evaluation_mode(localization_only=True, blind_cutoff=None) is None
    assert validate_evaluation_mode(localization_only=False, blind_cutoff=4.5) == 4.5
    with pytest.raises(ValueError, match="required unless"):
        validate_evaluation_mode(localization_only=False, blind_cutoff=None)
    with pytest.raises(ValueError, match="must be omitted"):
        validate_evaluation_mode(localization_only=True, blind_cutoff=4.5)


def test_b0_scope_is_low_rank_compatibility_not_unique_rome_attribution() -> None:
    result = clean_reference_decision(
        {
            "4": {
                "magnitude": 0.2,
                "magnitude_bound": 1e-6,
                "tail_ratio": 1e-8,
                "tail_ratio_bound": 1e-5,
                "passes_dtype_bound": True,
            }
        },
        [4],
    )

    assert result["is_rome_like"] is True
    assert result["attribution_scope"] == "generic_rank_at_most_two_gram_change"


def test_evaluator_enumerates_ignored_style_artifacts_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    baseline_weights = _weights(layers=10)
    suspect_weights = dict(baseline_weights)
    suspect_weights[5] = baseline_weights[5] + torch.ones(5, 1, dtype=torch.float64) @ torch.ones(
        1, 8, dtype=torch.float64
    )
    baseline_data = capture_rome_math_ablation(_context(baseline_weights))
    patch_data = capture_rome_math_ablation(_context(suspect_weights, baseline=baseline_weights, changed=(5,)))
    plan_root = tmp_path / "run" / "plans" / "model-a" / "cases0-0_r01"
    baseline_path = plan_root / "baseline" / "captures" / "rome-math-ablation.json"
    suspect_path = plan_root / "methods" / "rome" / "captures" / "rome-math-ablation.json"
    execution_path = plan_root / "methods" / "rome" / "execution.json"
    for path in (baseline_path, suspect_path, execution_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    common_run = {
        "run_id": "run",
        "model": "model-a",
        "plan_id": "cases0-0_r01",
    }
    baseline_path.write_text(
        json.dumps(
            {
                "kind": "capture",
                "producer": "rome-math-ablation",
                "run": {**common_run, "edit_method": None},
                "cases": [
                    {
                        "case_id": "baseline",
                        "status": "complete",
                        "data": baseline_data,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suspect_path.write_text(
        json.dumps(
            {
                "kind": "capture",
                "producer": "rome-math-ablation",
                "run": {**common_run, "edit_method": "rome"},
                "cases": [
                    {
                        "case_id": "0",
                        "status": "complete",
                        "data": patch_data,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    execution_path.write_text(
        json.dumps(
            {
                "summary": {"target_layer": 5},
                "cases": [
                    {
                        "case_id": "0",
                        "status": "complete",
                        "edit": {"method": "rome", "success": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = enumerate_artifacts(tmp_path)
    cases = collect_cases(
        paths,
        model_metadata={"model-a": {"family": "family-a", "split": "held_out_family"}},
        blind_candidate="M0",
        blind_cutoff=3.0,
    )
    metrics = summarize(cases)

    assert paths == [baseline_path, suspect_path]
    assert cases[0]["source_run"] == "run"
    assert cases[0]["family"] == "family-a"
    assert cases[0]["target_layer"] == 5
    assert cases[0]["edit_success"] is True
    assert cases[0]["localized_layer"] is not None
    assert cases[0]["presence_peak_layer"] is not None
    assert metrics["held_out_family"]["cases_total"] == 1
    assert metrics["held_out_family"]["edit_success_count"] == 1
    assert metrics["held_out_family"]["localization"]["M0"]["successful_edit_accuracy"] == 1.0
    assert metrics["held_out_family"]["B0"]["successful_edits_evaluated"] == 1

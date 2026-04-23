from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_posthoc import build_model_summary, load_model_meta, model_assignment_map


def _make_gpt_case(target_layer: int, success: bool) -> dict:
    layer_features = {}
    for layer in range(24):
        peak = 5.0 if layer == target_layer else 1.0
        layer_features[str(layer)] = {
            "norm_cv": peak,
            "spectral_gap": peak,
            "top1_energy": peak,
            "effective_rank": float(24 - abs(layer - target_layer)),
            "row_alignment": peak,
        }

    case = {
        "case_id": 1 if success else 2,
        "subject": "Ada Lovelace",
        "error": None,
        "rome": {
            "success": success,
            "efficacy_score": 1.0 if success else 0.0,
            "efficacy_magnitude": 0.75 if success else -0.25,
            "paraphrase_score": 1.0 if success else 0.0,
            "neighborhood_score": 1.0 if success else 0.0,
            "overall_score": 1.0 if success else 0.0,
        },
        "accuracy": {
            "rome_success": success,
            "detection_skipped": not success,
        },
    }
    if success:
        case["blind_detection"] = {"layer_features": layer_features}
    else:
        case["detection_skipped"] = True
        case["detection_skip_reason"] = "rome_unsuccessful"
    return case


def _make_composite_case(target_layer: int, success: bool, num_layers: int = 24) -> dict:
    layer_features = {}
    for layer in range(num_layers):
        is_target = layer == target_layer
        spectral_gap = 10.0 if is_target else 1.0
        top1_energy = 0.85 if is_target else 0.2
        norm_cv = 1.8 if is_target else 1.0
        effective_rank = 4.0 if is_target else 9.0
        row_alignment = 0.8 if is_target else 0.2
        spectral_entropy = 0.25 if is_target else 0.65
        pcs = 0.7 if is_target else 0.3
        layer_features[str(layer)] = {
            "spectral_gap": spectral_gap,
            "top1_energy": top1_energy,
            "norm_cv": norm_cv,
            "effective_rank": effective_rank,
            "row_alignment": row_alignment,
            "spectral_entropy": spectral_entropy,
            "pcs": pcs,
        }

    case = {
        "case_id": 101 if success else 102,
        "subject": "Grace Hopper",
        "error": None,
        "rome": {
            "success": success,
            "efficacy_score": 1.0 if success else 0.0,
            "efficacy_magnitude": 0.8 if success else -0.2,
            "paraphrase_score": 1.0 if success else 0.0,
            "neighborhood_score": 1.0 if success else 0.0,
            "overall_score": 1.0 if success else 0.0,
        },
        "accuracy": {
            "rome_success": success,
            "detection_skipped": not success,
        },
    }
    if success:
        case["blind_detection"] = {"layer_features": layer_features}
        case["spectral_detection"] = {
            "sv_z_scores": {str(layer): (6.0 if layer == target_layer else 1.0) for layer in range(num_layers)},
            "sv_ratio_scores": {str(layer): (4.0 if layer == target_layer else 0.5) for layer in range(num_layers)},
            "pcs_next_jump_scores": {str(layer): (2.0 if layer == target_layer else 0.25) for layer in range(num_layers)},
            "pcs_next_curvature_scores": {str(layer): (1.5 if layer == target_layer else 0.2) for layer in range(num_layers)},
            "pcs_cross_scores": {str(layer): (0.8 if layer == target_layer else 0.3) for layer in range(num_layers)},
            "pcs_cross_curvature_scores": {str(layer): (1.2 if layer == target_layer else 0.15) for layer in range(num_layers)},
            "raw_spectral": {
                "sv_proj_topk": {
                    str(layer): [3.0, 1.5, 0.75] if layer == target_layer else [2.0, 1.0, 0.5]
                    for layer in range(num_layers)
                }
            },
        }
    else:
        case["detection_skipped"] = True
        case["detection_skip_reason"] = "rome_unsuccessful"
    return case


def _write_structural_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_model_summary_skips_unsuccessful_rome_for_detector(tmp_path: Path) -> None:
    structural_dir = tmp_path / "structural"
    detector_dir = tmp_path / "detector"
    structural_dir.mkdir()
    detector_dir.mkdir()

    payload = {
        "metadata": {
            "model": "gpt2-large",
            "target_layer": 12,
            "analysis_profile": "paper",
        },
        "tests": [
            _make_gpt_case(target_layer=12, success=True),
            _make_gpt_case(target_layer=12, success=False),
        ],
        "summary": {
            "cases_total": 2,
            "cases_completed": 2,
            "cases_error": 0,
            "rome_success_count": 1,
            "detector_eligible_count": 1,
        },
    }

    json_path = structural_dir / "rome_structural_gpt2-large_test.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_model_summary(
        model_key="gpt2-large",
        model_meta=load_model_meta("gpt2-large"),
        json_path=json_path,
        detector_dir=detector_dir,
        detector_graph_dir=None,
        paper_graph_dir=None,
        expected_n_tests=2,
        assigned_start_idx=30,
        assigned_end_idx=31,
    )

    assert summary["status"] == "ok"
    assert summary["assigned_start_idx"] == 30
    assert summary["assigned_end_idx"] == 31
    assert summary["actual_start_idx"] == 30
    assert summary["actual_end_idx"] == 31
    assert summary["tested"] == 2
    assert summary["errors"] == 0
    assert summary["rome_ok"] == 1
    assert summary["det_eval"] == 1
    assert summary["det_ok"] == 1
    assert summary["det_rate"] == 1.0
    assert Path(summary["detector_summary"]).exists()


def test_build_model_summary_handles_missing_json(tmp_path: Path) -> None:
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()

    summary = build_model_summary(
        model_key="qwen3-4b",
        model_meta=load_model_meta("qwen3-4b"),
        json_path=None,
        detector_dir=detector_dir,
        detector_graph_dir=None,
        paper_graph_dir=None,
        expected_n_tests=50,
        assigned_start_idx=80,
        assigned_end_idx=129,
    )

    assert summary["status"] == "missing_structural_json"
    assert summary["assigned_start_idx"] == 80
    assert summary["assigned_end_idx"] == 129
    assert summary["tested"] == 50
    assert summary["errors"] == 50
    assert summary["det_eval"] == 0


def test_model_assignment_map_supports_shared_slice_policy() -> None:
    assignments = model_assignment_map(
        ["mistral-7b-v0.3", "opt-6.7b", "falcon-7b"],
        base_start_idx=30,
        n_tests=5,
        slice_policy="shared",
    )

    assert assignments["mistral-7b-v0.3"] == {"assigned_start_idx": 30, "assigned_end_idx": 34}
    assert assignments["opt-6.7b"] == {"assigned_start_idx": 30, "assigned_end_idx": 34}
    assert assignments["falcon-7b"] == {"assigned_start_idx": 30, "assigned_end_idx": 34}


def test_build_model_summary_generates_detector_and_paper_graphs_for_paper_profile(tmp_path: Path) -> None:
    model_key = "qwen3-8b"
    model_meta = load_model_meta(model_key)
    target_layer = int(model_meta["layer"])

    structural_dir = tmp_path / "structural"
    detector_dir = tmp_path / "detector"
    detector_graph_dir = tmp_path / "detector_graphs"
    paper_graph_dir = tmp_path / "paper_graphs"

    baseline_path = structural_dir / "baseline_structural_qwen3-8b_test.json"
    edited_path = structural_dir / "rome_structural_qwen3-8b_test.json"

    baseline_payload = {
        "metadata": {
            "model": model_meta["hf_name"],
            "target_layer": target_layer,
            "analysis_profile": "paper",
            "paired_role": "baseline",
            "paired_run_id": "test-run",
            "edited_structural_file": edited_path.name,
        },
        "tests": [_make_composite_case(target_layer=target_layer, success=True)],
        "summary": {
            "cases_total": 1,
            "cases_completed": 1,
            "cases_error": 0,
            "rome_success_count": 1,
            "detector_eligible_count": 1,
        },
    }
    edited_payload = {
        "metadata": {
            "model": model_meta["hf_name"],
            "target_layer": target_layer,
            "analysis_profile": "paper",
            "paired_role": "edited",
            "paired_run_id": "test-run",
            "baseline_structural_file": baseline_path.name,
        },
        "tests": [_make_composite_case(target_layer=target_layer, success=True)],
        "summary": {
            "cases_total": 1,
            "cases_completed": 1,
            "cases_error": 0,
            "rome_success_count": 1,
            "detector_eligible_count": 1,
        },
    }

    _write_structural_payload(baseline_path, baseline_payload)
    _write_structural_payload(edited_path, edited_payload)

    summary = build_model_summary(
        model_key=model_key,
        model_meta=model_meta,
        json_path=edited_path,
        detector_dir=detector_dir,
        detector_graph_dir=detector_graph_dir,
        paper_graph_dir=paper_graph_dir,
        expected_n_tests=1,
        assigned_start_idx=30,
        assigned_end_idx=30,
    )

    assert summary["status"] == "ok"
    assert summary["det_eval"] == 1
    assert Path(summary["detector_summary"]).exists()
    assert any(detector_graph_dir.glob("*.png"))
    assert any(paper_graph_dir.glob("*.png"))

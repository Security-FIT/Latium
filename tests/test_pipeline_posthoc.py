from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_posthoc import build_model_summary, load_model_meta


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

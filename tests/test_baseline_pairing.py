from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.composite_detector_v2 import _find_baseline
from paper_graphs.paper_graphs_support import (
    _stack_attention_contrast_metrics,
    _stack_single_specs,
    load_run_manifest,
)


def _write_payload(path: Path, model: str, tests: list[dict] | None = None, **metadata_overrides) -> None:
    default_tests = [
        {
            "rome": {"success": True},
            "blind_detection": {
                "layer_features": {
                    "0": {
                        "norm_cv": 1.0,
                        "spectral_gap": 1.0,
                        "top1_energy": 0.5,
                        "effective_rank": 2.0,
                    }
                }
            },
            "spectral_detection": {"sv_z_scores": {"0": 1.0}},
        }
    ]
    payload = {
        "metadata": {
            "model": model,
            "target_layer": 15,
            "analysis_profile": "full",
            **metadata_overrides,
        },
        "tests": tests or default_tests,
        "summary": {"total": 1, "successful": 1, "skipped": 0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_find_baseline_prefers_matching_run_stem(tmp_path: Path) -> None:
    structural_dir = tmp_path / "pipeline_out" / "torch26" / "structural"
    rome_path = structural_dir / (
        "rome_structural_opt-6.7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r01_2026-04-22_12-22-10.json"
    )
    baseline_path = structural_dir / (
        "baseline_structural_opt-6.7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r01_2026-04-22_12-24-01.json"
    )
    wrong_baseline_path = structural_dir / (
        "baseline_structural_opt-6.7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r02_2026-04-22_12-30-00.json"
    )

    _write_payload(rome_path, "facebook/opt-6.7b")
    _write_payload(baseline_path, "facebook/opt-6.7b")
    _write_payload(wrong_baseline_path, "facebook/opt-6.7b")

    assert _find_baseline(rome_path) == baseline_path


def test_load_run_manifest_uses_per_run_baseline_not_latest_model_baseline(tmp_path: Path) -> None:
    (tmp_path / "paper_graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper_graphs" / "paper_graphs_support.py").write_text("# test repo marker\n", encoding="utf-8")

    structural_dir = tmp_path / "pipeline_out" / "run_a" / "structural"
    rome_path = structural_dir / (
        "rome_structural_falcon-7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r01_2026-04-22_12-25-19.json"
    )
    desired_baseline = structural_dir / (
        "baseline_structural_falcon-7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r01_2026-04-22_12-27-16.json"
    )
    newer_other_baseline = tmp_path / "pipeline_out" / "run_b" / "structural" / (
        "baseline_structural_falcon-7b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_"
        "s01_r02_2026-04-22_12-40-00.json"
    )

    _write_payload(
        rome_path,
        "tiiuae/falcon-7b",
        baseline_structural_file=desired_baseline.name,
    )
    _write_payload(desired_baseline, "tiiuae/falcon-7b")
    _write_payload(newer_other_baseline, "tiiuae/falcon-7b")

    summary_df, runs, skipped_runs = load_run_manifest(root=tmp_path, run_selection="all")

    assert skipped_runs == []
    assert len(summary_df) == 1
    assert len(runs) == 1
    assert runs[0]["baseline_path"] == desired_baseline


def test_load_run_manifest_accepts_paper_profile_runs(tmp_path: Path) -> None:
    (tmp_path / "paper_graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper_graphs" / "paper_graphs_support.py").write_text("# test repo marker\n", encoding="utf-8")

    structural_dir = tmp_path / "pipeline_out" / "run_paper" / "structural"
    rome_path = structural_dir / "rome_structural_gpt2-large_test.json"
    baseline_path = structural_dir / "baseline_structural_gpt2-large_test.json"
    tests = [
        {
            "rome": {"success": True},
            "blind_detection": {
                "layer_features": {
                    "0": {"norm_cv": 1.0, "spectral_gap": 1.0, "top1_energy": 0.5, "effective_rank": 2.0}
                }
            },
        }
    ]

    _write_payload(
        rome_path,
        "gpt2-large",
        tests=tests,
        analysis_profile="paper",
        baseline_structural_file=baseline_path.name,
    )
    _write_payload(
        baseline_path,
        "gpt2-large",
        tests=tests,
        analysis_profile="paper",
    )

    summary_df, runs, skipped_runs = load_run_manifest(root=tmp_path, run_selection="all")

    assert skipped_runs == []
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["analysis_profile"] == "paper"
    assert len(runs) == 1
    assert runs[0]["path"] == rome_path
    assert runs[0]["baseline_path"] == baseline_path


def test_paper_profile_stack_omits_full_only_panels() -> None:
    titles = [title for title, *_ in _stack_single_specs("paper")]

    assert titles == [
        "Blind spectral_gap",
        "Blind top1_energy",
        "Stable/effective rank ratio",
        "Blind norm_cv",
        "Blind spectral_entropy",
        "Blind row_alignment",
    ]
    assert _stack_attention_contrast_metrics("paper") == {}


def test_paper_payload_with_spectral_detection_enables_spectral_panels() -> None:
    payload = {
        "metadata": {"analysis_profile": "paper"},
        "tests": [
            {
                "rome": {"success": True},
                "blind_detection": {
                    "layer_features": {
                        "0": {
                            "norm_cv": 1.0,
                            "spectral_gap": 1.0,
                            "top1_energy": 0.5,
                            "effective_rank": 2.0,
                            "spectral_entropy": 0.4,
                            "row_alignment": 1.2,
                        }
                    }
                },
                "spectral_detection": {
                    "sv_z_scores": {"0": 1.0},
                    "sv_ratio_scores": {"0": 0.5},
                    "pcs_next_jump_scores": {"0": 0.25},
                    "pcs_next_curvature_scores": {"0": 0.15},
                    "pcs_cross_scores": {"0": 0.35},
                    "pcs_cross_curvature_scores": {"0": 0.1},
                    "raw_spectral": {"sv_proj_topk": {"0": [1.0, 0.5]}},
                },
            }
        ],
    }

    titles = [title for title, *_ in _stack_single_specs(payload)]

    assert "Top-k proj singular values at target layer" in titles
    assert "Signal A (sv_z_scores)" in titles
    assert "Signal B (sv_ratio_scores)" in titles
    assert "PCS next jump" in titles
    assert "PCS cross curvature" in titles
    assert _stack_attention_contrast_metrics(payload) == {}


def test_attention_stack_requires_attention_payload() -> None:
    payload = {
        "metadata": {"analysis_profile": "paper"},
        "tests": [
            {
                "rome": {"success": True},
                "blind_detection": {"layer_features": {"0": {"norm_cv": 1.0}}},
                "attention_detection": {
                    "contrast_series": {
                        "proj_attn_norm_log_ratio": {"0": 1.0},
                    }
                },
            }
        ],
    }

    metrics = _stack_attention_contrast_metrics(payload)

    assert "proj_attn_norm_log_ratio" in metrics

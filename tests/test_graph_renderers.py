"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.graphs.context import RendererUnavailableError
from src.graphs.runtime import render_run
from src.graphs.renderers import render_run_summary
from src.graphs.structural.materialize import matching_baseline
from src.results import ArtifactWriter, build_artifact, config_hash
from src.results.ids import capture_id, execution_id


def test_run_summary_uses_analysis_producer_name(tmp_path: Path) -> None:
    outputs = render_run_summary(
        {
            "output_dir": tmp_path,
            "analyses": [
                {
                    "producer": "composite",
                    "category": "detection",
                    "status": "complete",
                    "run": {
                        "model": "qwen3-4b",
                        "plan_id": "cases0-0_r01",
                        "edit_method": "rome",
                    },
                    "summary": {
                        "accuracy": 1.0,
                        "correct": 1,
                        "cases_evaluated": 1,
                    },
                }
            ],
        }
    )

    payload = json.loads(Path(outputs[0]).read_text(encoding="utf-8"))
    assert payload["analyses"][0]["analysis"] == "composite"


def test_run_level_render_artifact_uses_null_run_selectors(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    config: dict[str, object] = {}
    writer.write(
        tmp_path / "analysis.json",
        build_artifact(
            artifact_id="analysis",
            kind="analysis",
            category="detection",
            producer="composite",
            run_id="run",
            model="qwen3-4b",
            plan_id="cases0-0_r01",
            edit_method="rome",
            status="complete",
            config=config,
            config_hash=config_hash(config),
            inputs=[],
            created_at=datetime.now().isoformat(),
            cases=[],
            summary={"accuracy": 1.0, "correct": 1, "cases_evaluated": 1},
        ),
    )

    render_run(tmp_path, preset="none", enabled=("run-summary",))

    artifact = json.loads(
        (tmp_path / "graphs" / "run-summary" / "artifact.json").read_text(
            encoding="utf-8",
        )
    )
    assert artifact["run"] == {
        "run_id": "run",
        "model": None,
        "plan_id": None,
        "edit_method": None,
    }


def test_registered_graph_renderers_use_manifest_artifacts(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    execution_config: dict[str, object] = {"edit_method": "rome"}
    writer.write(
        tmp_path / "execution.json",
        build_artifact(
            artifact_id="execution",
            kind="execution",
            producer="rome",
            run_id="run",
            model="qwen3-4b",
            plan_id="cases0-0_r01",
            edit_method="rome",
            status="complete",
            config=execution_config,
            config_hash=config_hash(execution_config),
            inputs=[],
            created_at=datetime.now().isoformat(),
            cases=[
                {
                    "case_id": "case",
                    "status": "complete",
                    "edit": {
                        "success": True,
                        "metrics": {
                            "overall_score": 0.8,
                            "efficacy_score": 1.0,
                            "paraphrase_score": 0.7,
                            "neighborhood_score": 0.6,
                        },
                    },
                    "error": None,
                }
            ],
            summary={
                "target_layer": 4,
                "num_layers": 8,
                "cases_total": 1,
                "cases_complete": 1,
                "edit_success_count": 1,
                "edit_success_rate": 1.0,
            },
        ),
    )
    analysis_config: dict[str, object] = {"analysis": "rank1"}
    writer.write(
        tmp_path / "analysis.json",
        build_artifact(
            artifact_id="analysis",
            kind="analysis",
            category="detection",
            producer="rank1-blind",
            run_id="run",
            model="qwen3-4b",
            plan_id="cases0-0_r01",
            edit_method="rome",
            status="complete",
            config=analysis_config,
            config_hash=config_hash(analysis_config),
            inputs=[],
            created_at=datetime.now().isoformat(),
            cases=[
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": {
                        "anomalous_layer": 4,
                        "detection_score": 0.9,
                        "combined_score": {"3": 0.2, "4": 0.9, "5": 0.3},
                    },
                    "error": None,
                    "accuracy": {
                        "target_layer": 4,
                        "detected_layer": 4,
                        "correct": True,
                    },
                }
            ],
            summary={"accuracy": 1.0, "correct": 1, "cases_evaluated": 1},
        ),
    )

    result = render_run(
        tmp_path,
        preset="none",
        enabled=("rome-success", "detector-window", "detector-signals"),
    )

    assert set(result["written"]) == {"render/rome-success", "render/detector-window", "render/detector-signals"}
    assert (tmp_path / "graphs" / "rome-success" / "rome-success-rate.png").is_file()
    assert (tmp_path / "graphs" / "detector-window" / "detector-layer-window.png").is_file()
    signal_index = json.loads(
        (tmp_path / "graphs" / "detector-signals" / "detector-signal-profiles.json").read_text(
            encoding="utf-8",
        )
    )
    assert signal_index["outputs"][0]["series"] == ["combined_score"]
    assert (tmp_path / "graphs" / "detector-signals" / signal_index["outputs"][0]["output"]).is_file()


def _matrix_case(case_id: str, *, peak_layer: int = 4) -> dict:
    features = {}
    for layer in range(8):
        is_peak = layer == peak_layer
        features[str(layer)] = {
            "spectral_gap": 9.0 if is_peak else 1.0 + 0.1 * layer,
            "top1_energy": 8.0 if is_peak else 1.0 + 0.05 * layer,
            "row_alignment": 6.0 if is_peak else 0.5 + 0.02 * layer,
            "norm_cv": 7.0 if is_peak else 1.0 + 0.04 * layer,
            "effective_rank": 1.0 if is_peak else 5.0 + layer,
        }
    return {
        "case_id": case_id,
        "status": "complete",
        "data": {
            "mode": "materialized",
            "feature_set": "paper",
            "features": ["spectral_gap", "top1_energy", "row_alignment", "norm_cv", "effective_rank"],
            "families": {"proj": features},
        },
        "error": None,
    }


def test_matching_baseline_prefers_artifact_input_ref_over_config_hash() -> None:
    edited = {
        "artifact_id": "edited",
        "run": {"model": "m", "plan_id": "p", "edit_method": "rome"},
        "config_hash": "same",
        "inputs": [{"artifact_id": "baseline-b"}],
    }
    baseline_a = {
        "artifact_id": "baseline-a",
        "run": {"model": "m", "plan_id": "p", "edit_method": None},
        "config_hash": "same",
    }
    baseline_b = {
        "artifact_id": "baseline-b",
        "run": {"model": "m", "plan_id": "p", "edit_method": None},
        "config_hash": "same",
    }

    assert matching_baseline(edited, [baseline_a, baseline_b]) is baseline_b


def test_matching_baseline_rejects_ambiguous_config_hash() -> None:
    edited = {
        "artifact_id": "edited",
        "run": {"model": "m", "plan_id": "p", "edit_method": "rome"},
        "config_hash": "same",
        "inputs": [],
    }
    baseline_a = {
        "artifact_id": "baseline-a",
        "run": {"model": "m", "plan_id": "p", "edit_method": None},
        "config_hash": "same",
    }
    baseline_b = {
        "artifact_id": "baseline-b",
        "run": {"model": "m", "plan_id": "p", "edit_method": None},
        "config_hash": "same",
    }

    with pytest.raises(RendererUnavailableError):
        matching_baseline(edited, [baseline_a, baseline_b])


def test_structural_artifact_grid_uses_declared_matrix_inputs(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    model = "qwen3-4b"
    plan_id = "cases0-0_r01"
    method = "rome"
    now = datetime.now().isoformat()

    writer.write(
        tmp_path / "execution.json",
        build_artifact(
            artifact_id=execution_id(model, plan_id, method),
            kind="execution",
            producer=method,
            run_id="run",
            model=model,
            plan_id=plan_id,
            edit_method=method,
            status="complete",
            config={"edit_method": method},
            config_hash=config_hash({"edit_method": method}),
            inputs=[],
            created_at=now,
            cases=[{"case_id": "case", "status": "complete", "edit": {"success": True}, "error": None}],
            summary={"target_layer": 4, "num_layers": 8, "cases_total": 1, "cases_complete": 1},
        ),
    )
    for edit_method, case in ((None, _matrix_case("baseline", peak_layer=1)), (method, _matrix_case("case"))):
        config = {"capture": "matrix-features", "options": {"feature_set": "paper"}}
        writer.write(
            tmp_path / ("baseline_matrix.json" if edit_method is None else "method_matrix.json"),
            build_artifact(
                artifact_id=capture_id(model, plan_id, "matrix-features", edit_method),
                kind="capture",
                producer="matrix-features",
                run_id="run",
                model=model,
                plan_id=plan_id,
                edit_method=edit_method,
                status="complete",
                config=config,
                config_hash=config_hash(config),
                inputs=[],
                created_at=now,
                cases=[case],
                summary={"cases_total": 1, "cases_complete": 1},
            ),
        )

    result = render_run(
        tmp_path,
        preset="none",
        enabled=("structural-artifact-grid",),
        renderer_options={"structural-artifact-grid": {"formats": ["json"], "transforms": ["raw"]}},
    )

    assert result["written"] == ["render/structural-artifact-grid"]
    payload = json.loads((tmp_path / "graphs" / "structural-artifact-grid" / "artifact.json").read_text())
    assert payload["status"] == "complete"
    data_path = tmp_path / "graphs" / "structural-artifact-grid" / "artifact_grid_qwen3-4b_cases0-0_r01_rome.json"
    data = json.loads(data_path.read_text())
    assert data["series"]["spectral_gap"]["raw"]["edited"]["layers"] == list(range(8))
    assert data["target_layer"] == 4


def test_structural_artifact_grid_marks_missing_matrix_capture_unavailable(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    writer.write(
        tmp_path / "execution.json",
        build_artifact(
            artifact_id=execution_id("qwen3-4b", "cases0-0_r01", "rome"),
            kind="execution",
            producer="rome",
            run_id="run",
            model="qwen3-4b",
            plan_id="cases0-0_r01",
            edit_method="rome",
            status="complete",
            config={},
            config_hash=config_hash({}),
            inputs=[],
            created_at=datetime.now().isoformat(),
            cases=[],
            summary={},
        ),
    )

    render_run(tmp_path, preset="none", enabled=("structural-artifact-grid",))

    payload = json.loads((tmp_path / "graphs" / "structural-artifact-grid" / "artifact.json").read_text())
    assert payload["status"] == "unavailable"
    assert "capture:matrix-features" in payload["error"]

"""The CCS report is one complete, reproducible render workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.common.arrays import local_zscore
from src.graphs.context import RenderExecutionError
from src.graphs.registry import resolve_renderers
from src.graphs.runtime import render_run
from src.results import ArtifactWriter, RunArtifactReader, build_artifact, config_hash
from src.results.ids import capture_id, execution_id


MODEL = "qwen3-4b"
PLAN = "cases0-1_r01"
METHOD = "rome"
RENDERERS = ("paper", "detector", "rome-success", "detector-window", "structural-ccs-lines")


def _case(case_id: str, peak: int) -> dict:
    layers = {}
    for layer in range(10):
        layers[str(layer)] = {
            "spectral_gap": 1.0 + 0.05 * layer + (4.0 if layer == peak else 0.0),
            "top1_energy": 0.2 + 0.01 * layer + (0.4 if layer == peak else 0.0),
        }
    return {
        "case_id": case_id,
        "status": "complete",
        "data": {"families": {"proj": layers}},
        "error": None,
    }


def _write_fixture(
    root: Path, *, baseline: bool = True, captures: bool = True, evaluated: bool = True
) -> None:
    writer = ArtifactWriter(root, run_id="run")
    now = datetime.now().isoformat()
    execution_config = {"edit_method": METHOD}
    writer.write(
        root / "execution.json",
        build_artifact(
            artifact_id=execution_id(MODEL, PLAN, METHOD),
            kind="execution",
            producer=METHOD,
            run_id="run",
            model=MODEL,
            plan_id=PLAN,
            edit_method=METHOD,
            status="complete",
            config=execution_config,
            config_hash=config_hash(execution_config),
            inputs=[],
            created_at=now,
            cases=[
                {"case_id": "edit-1", "status": "complete", "edit": {"success": True}, "error": None},
                {"case_id": "edit-2", "status": "complete", "edit": {"success": True}, "error": None},
            ],
            summary={"target_layer": 4, "cases_total": 2, "cases_complete": 2, "edit_success_count": 2},
        ),
    )
    analysis_config = {"method": "ccs-composite"}
    writer.write(
        root / "analysis.json",
        build_artifact(
            artifact_id="analysis/ccs-edited",
            kind="analysis",
            category="detection",
            producer="ccs-composite",
            run_id="run",
            model=MODEL,
            plan_id=PLAN,
            edit_method=METHOD,
            status="complete",
            config=analysis_config,
            config_hash=config_hash(analysis_config),
            inputs=[],
            created_at=now,
            cases=[
                {
                    "case_id": "edit-1",
                    "status": "complete",
                    "data": {"detection_score": 0.9},
                    "accuracy": {"target_layer": 4, "detected_layer": 4, "correct": True},
                }
            ] if evaluated else [],
            summary={
                "accuracy": 1.0 if evaluated else 0.0,
                "correct": 1 if evaluated else 0,
                "cases_evaluated": 1 if evaluated else 0,
            },
        ),
    )
    if not captures:
        return
    matrix_config = {"capture": "matrix-features", "features": ["spectral_gap", "top1_energy"]}
    baseline_id = capture_id(MODEL, PLAN, "matrix-features", None)
    if baseline:
        writer.write(
            root / "baseline.json",
            build_artifact(
                artifact_id=baseline_id,
                kind="capture",
                producer="matrix-features",
                run_id="run",
                model=MODEL,
                plan_id=PLAN,
                edit_method=None,
                status="complete",
                config=matrix_config,
                config_hash=config_hash(matrix_config),
                inputs=[],
                created_at=now,
                cases=[_case("baseline-1", 2), _case("baseline-2", 3)],
                summary={"cases_total": 2, "cases_complete": 2},
            ),
        )
    writer.write(
        root / "edited.json",
        build_artifact(
            artifact_id=capture_id(MODEL, PLAN, "matrix-features", METHOD),
            kind="capture",
            producer="matrix-features",
            run_id="run",
            model=MODEL,
            plan_id=PLAN,
            edit_method=METHOD,
            status="complete",
            config=matrix_config,
            config_hash=config_hash(matrix_config),
            inputs=[RunArtifactReader(root).ref(baseline_id)] if baseline else [],
            created_at=now,
            cases=[_case("edit-1", 4), _case("edit-2", 5)],
            summary={"cases_total": 2, "cases_complete": 2},
        ),
    )


def test_ccs_report_renders_all_summary_and_six_panel_outputs(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    assert resolve_renderers("ccs-report") == RENDERERS

    result = render_run(tmp_path, preset="ccs-report")

    assert set(result["written"]) == {f"render/{renderer}" for renderer in RENDERERS}
    for renderer in RENDERERS:
        artifact = json.loads((tmp_path / "graphs" / renderer / "artifact.json").read_text())
        assert artifact["status"] == "complete"
    assert (tmp_path / "graphs" / "detector" / "detection-accuracy.png").is_file()
    assert (tmp_path / "graphs" / "rome-success" / "rome-success-rate.png").is_file()
    assert (tmp_path / "graphs" / "detector-window" / "detector-layer-window.png").is_file()

    line_dir = tmp_path / "graphs" / "structural-ccs-lines"
    stem = f"ccs_lines_{MODEL}_{PLAN}_{METHOD}"
    assert (line_dir / f"{stem}.png").is_file()
    assert (line_dir / f"{stem}.pdf").is_file()
    data = json.loads((line_dir / f"{stem}.json").read_text())
    assert data["features"] == ["spectral_gap", "top1_energy"]
    assert data["transforms"] == ["raw", "lz5", "lz7"]
    assert data["edited_case_ids"] == ["edit-1", "edit-2"]
    assert data["target_layer"] == 4
    assert len(data["series"]["spectral_gap"]["raw"]["edited_cases"]) == 2
    raw = np.asarray(data["series"]["spectral_gap"]["raw"]["edited_cases"])
    expected = local_zscore(raw, 5, axis=1, absolute=True)
    np.testing.assert_allclose(data["series"]["spectral_gap"]["lz5"]["edited_cases"], expected)

    assert set(render_run(tmp_path, preset="ccs-report")["skipped"]) == {f"render/{name}" for name in RENDERERS}
    (line_dir / f"{stem}.png").unlink()
    rerun = render_run(tmp_path, preset="ccs-report")
    assert rerun["written"] == ["render/structural-ccs-lines"]
    assert (line_dir / f"{stem}.png").is_file()


@pytest.mark.parametrize("baseline,captures,reason", [
    (False, True, "no matched unedited"),
    (True, False, "capture:matrix-features"),
])
def test_ccs_report_fails_when_line_graph_is_unavailable(
    tmp_path: Path, baseline: bool, captures: bool, reason: str
) -> None:
    _write_fixture(tmp_path, baseline=baseline, captures=captures)
    with pytest.raises(RenderExecutionError, match="ccs-report incomplete") as exc:
        render_run(tmp_path, preset="ccs-report")
    assert reason in str(exc.value)
    artifact = json.loads((tmp_path / "graphs" / "structural-ccs-lines" / "artifact.json").read_text())
    assert artifact["status"] == "unavailable"


def test_ccs_report_fails_if_accuracy_graphs_have_no_evaluated_cases(tmp_path: Path) -> None:
    _write_fixture(tmp_path, evaluated=False)
    with pytest.raises(RenderExecutionError, match="ccs-report incomplete") as exc:
        render_run(tmp_path, preset="ccs-report")
    assert "detector: required report files were not produced" in str(exc.value)


def test_old_aggregate_only_preset_cannot_run() -> None:
    with pytest.raises(ValueError, match="ccs-report"):
        resolve_renderers("paper")
    with pytest.raises(ValueError, match="cannot disable"):
        resolve_renderers("ccs-report", disabled=("structural-ccs-lines",))


def test_named_ccs_workflow_plans_every_report_output() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, "-m", "src", "structural", "plan", "structural=ccs-report",
            "structural.run.models=[qwen3-4b]", "structural.run.n_tests=2",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["resolved_renderers"] == list(RENDERERS)
    assert "matrix-features" in plan["resolved_captures"]
    assert plan["render_graphs"] is True
    assert plan["renderer_options"]["structural-ccs-lines"]["formats"] == ["png", "pdf", "json"]

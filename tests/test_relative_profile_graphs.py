"""Saved relative decisions remain traceable through cohort and case plots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.graphs.context import RenderContext, RendererUnavailableError
from src.graphs.registry import resolve_renderers
from src.graphs.runtime import render_run
from src.graphs.structural.relative_materialize import (
    matching_relative_baseline, materialize_relative, profile_stats, selected_fit,
)
from src.graphs.structural.relative_profiles import render_rome_relative_profile_grid
from src.structural.analysis.registry import resolve_analyses
from src.structural.detectors.rome_layer_localizer import relative_profile_decision
from src.results import ArtifactWriter, build_artifact, config_hash


PRODUCER = "rome-directional-experiments"
METHOD = "v0-relative-b0-v1"


def _decision(scores: dict[str, float], *, candidate: int = 3, detected: bool = True) -> dict:
    return {
        "status": "complete", "layer_scores": scores, "eligible_layers": list(range(6)),
        "candidate_layer": candidate, "original_localizer_layer": 2,
        "is_rome_like": detected, "gain": 4.2,
        "background_cost": 10.0, "anomaly_cost": 5.8,
        "diagnostics": {
            "selected_background": {"family": "affine", "coefficients": [1.0, 2.0]},
            "selected_anomaly": {"family": "affine", "coefficients": [1.0, 2.0, 0.5],
                                 "candidate_layer": candidate},
        },
    }


def _analysis(method: str | None, artifact_id: str, *, scores: dict[str, float] | None = None) -> dict:
    if scores is None:
        scores = {str(layer): float(layer + 1) for layer in range(6)}
    return {
        "artifact_id": artifact_id, "producer": PRODUCER, "config_hash": "same-hash",
        "run": {"model": "qwen3-4b", "plan_id": "plan", "edit_method": method},
        "cases": [{"case_id": "case-1", "status": "complete",
                   "data": {"experiments": {METHOD: _decision(scores)}}}],
    }


def _context(tmp_path: Path, edited: dict, baseline: dict | None, **options: object) -> RenderContext:
    analyses = [edited] + ([baseline] if baseline else [])
    return RenderContext(
        run_root=tmp_path, output_dir=tmp_path / "plots", manifest={},
        executions=({"run": edited["run"], "summary": {"target_layer": 3}},),
        analyses={PRODUCER: tuple(analyses)}, options=options,
    )


def test_relative_preset_selects_all_four_analysis_families() -> None:
    assert len(resolve_analyses("rome-relative")) == 4
    assert resolve_renderers("rome-relative-paper") == ("rome-relative-profile-grid",)


def test_materializer_keeps_case_rows_and_missing_layers() -> None:
    edited = _analysis("rome", "edited")
    edited["cases"].append({
        "case_id": "case-2", "status": "complete",
        "data": {"experiments": {METHOD: {
            "status": "unavailable", "eligible_layers": list(range(6)),
            "layer_scores": {"0": 2.0, "2": 5.0},
            "diagnostics": {"reason": "incomplete profile"},
        }}},
    })
    table = materialize_relative(edited, METHOD)
    assert table.values.shape == (2, 6)
    assert np.isnan(table.values[1, 1])
    assert table.values[1, 2] == 5.0
    assert profile_stats(table.values)["count"].tolist() == [2, 1, 2, 1, 1, 1]
    assert selected_fit(table.cases[0], "background")[5] == 3.0
    assert selected_fit(table.cases[0], "anomaly")[3] == pytest.approx(2.7)


def test_baseline_match_requires_same_lineage() -> None:
    edited = _analysis("rome", "edited")
    baseline = _analysis(None, "baseline")
    assert matching_relative_baseline(edited, [baseline]) is baseline
    other = {**baseline, "config_hash": "other"}
    assert matching_relative_baseline(edited, [other]) is None
    with pytest.raises(RendererUnavailableError, match="multiple unedited"):
        matching_relative_baseline(edited, [baseline, {**baseline, "artifact_id": "duplicate"}])


def test_unavailable_decision_preserves_valid_partial_scores() -> None:
    result = relative_profile_decision({"0": 1.0, "1": 2.0}, eligible_layers=list(range(6)))
    assert result["status"] == "unavailable"
    assert result["layer_scores"] == {"0": 1.0, "1": 2.0}


def test_renderer_writes_json_png_pdf_and_diagnostics(tmp_path: Path) -> None:
    edited = _analysis("rome", "edited")
    baseline = _analysis(None, "baseline")
    outputs = render_rome_relative_profile_grid(_context(
        tmp_path, edited, baseline, formats=["json", "png", "pdf"], case_pages="all",
    ))
    assert any(path.endswith(".png") for path in outputs)
    assert any(path.endswith("_cases.pdf") for path in outputs)
    assert any(path.endswith(".pdf") and not path.endswith("_cases.pdf") for path in outputs)
    index = json.loads((tmp_path / "plots" / "relative_profile_index.json").read_text())
    assert index["methods"][0]["status"] == "complete"
    data_path = next(Path(path) for path in outputs if path.endswith(".json") and not path.endswith("index.json"))
    data = json.loads(data_path.read_text())
    assert data["target_layer"] == 3
    assert data["edited"]["raw"]["count"] == [1] * 6
    assert data["baseline"]["artifact_id"] == "baseline"
    assert data["candidate_frequency"] == {"3": 1}


def test_renderer_marks_unavailable_without_inventing_data(tmp_path: Path) -> None:
    edited = _analysis("rome", "edited")
    edited["cases"][0]["data"]["experiments"][METHOD] = {"status": "unavailable", "layer_scores": {}}
    with pytest.raises(RendererUnavailableError, match="no usable"):
        render_rome_relative_profile_grid(_context(tmp_path, edited, None, formats=["json"], case_pages="none"))


def test_manifest_render_command_uses_relative_preset(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    config = {"experiments": [METHOD]}
    for method, artifact_id in (("rome", "analysis-edited"), (None, "analysis-baseline")):
        sample = _analysis(method, artifact_id)
        writer.write(tmp_path / f"{artifact_id}.json", build_artifact(
            artifact_id=artifact_id, kind="analysis", category="artifact-study",
            producer=PRODUCER, run_id="run", model="qwen3-4b", plan_id="plan",
            edit_method=method, status="complete", config=config,
            config_hash=config_hash(config), inputs=[], created_at=datetime.now().isoformat(),
            cases=sample["cases"], summary={},
        ))
    writer.write(tmp_path / "execution.json", build_artifact(
        artifact_id="execution", kind="execution", producer="rome", run_id="run",
        model="qwen3-4b", plan_id="plan", edit_method="rome", status="complete",
        config={}, config_hash=config_hash({}), inputs=[], created_at=datetime.now().isoformat(),
        cases=[], summary={"target_layer": 3},
    ))
    result = render_run(
        tmp_path, preset="rome-relative-paper",
        renderer_options={"rome-relative-profile-grid": {"formats": ["json"], "case_pages": "none"}},
    )
    assert result["written"] == ["render/rome-relative-profile-grid"]
    render_artifact = json.loads((tmp_path / "graphs" / "rome-relative-profile-grid" / "artifact.json").read_text())
    assert render_artifact["status"] == "complete"
    assert any(path.endswith("relative_profile_index.json") for path in render_artifact["summary"]["outputs"])

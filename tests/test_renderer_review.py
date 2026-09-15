"""Regression coverage for renderer dependencies and analysis variant identity."""

import csv
import json

import pytest

from src.graphs.context import RenderContext
from src.graphs.registry import RENDERERS, RendererSpec
from src.graphs.renderers import render_detector_signals, render_detector_window, render_run_summary
from src.graphs.runtime import _renderer_input_records, render_run
from src.results import ArtifactWriter


def test_renderer_dependencies_exclude_unrelated_artifacts():
    execution = {"artifact_id": "e"}
    capture = {"artifact_id": "c", "producer": "matrix-features"}
    analysis = {"artifact_id": "a", "producer": "gram-localization"}
    records = dict(executions=[execution], captures=[capture], analyses=[analysis])
    assert _renderer_input_records(RENDERERS.get("rome-success"), **records) == ([execution], [], [])
    assert _renderer_input_records(RENDERERS.get("detector"), **records) == ([analysis], [], [])
    spec = RendererSpec(
        "custom", "test", requires_analyses=True,
        required_analyses=("gram-localization",),
        optional_captures=("spectral",),
    )
    assert _renderer_input_records(spec, **records) == (
        [analysis], [], ["optional capture unavailable: spectral"],
    )


@pytest.mark.parametrize("renderer", ["paper", "detector", "run-summary", "rome-success"])
def test_missing_renderer_inputs_do_not_produce_successful_empty_reports(tmp_path, renderer):
    ArtifactWriter(tmp_path, run_id="empty")
    render_run(tmp_path, preset="none", enabled=(renderer,))
    artifact = json.loads((tmp_path / "graphs" / renderer / "artifact.json").read_text())
    assert artifact["status"] == "unavailable"
    assert artifact["summary"]["outputs"] == []


def test_variants_keep_separate_plots_csv_identity_and_window_groups(tmp_path, monkeypatch):
    import matplotlib.axes

    def payload(digest, detected):
        return {
            "producer": "gram-localization",
            "category": "detection",
            "status": "complete",
            "config_hash": digest,
            "run": {"model": "model", "plan_id": "plan", "edit_method": "rome"},
            "summary": {"accuracy": float(detected == 4), "correct": int(detected == 4), "cases_evaluated": 1},
            "cases": [{
                "case_id": "same-case", "status": "complete",
                "data": {"localization": {"layer_scores": {"3": 0.2, "4": 0.8}}},
                "accuracy": {"target_layer": 4, "detected_layer": detected, "correct": detected == 4},
            }],
        }

    # Equal short hash prefixes must still form distinct groups.
    hashes = ["a" * 63 + "1", "a" * 63 + "2"]
    context = RenderContext(
        run_root=tmp_path, output_dir=tmp_path, manifest={},
        analyses={"gram-localization": tuple(payload(h, d) for h, d in zip(hashes, (4, 7)))},
    )
    outputs = render_detector_signals(context)
    index = json.loads(open(outputs[0]).read())["outputs"]
    assert len({row["output"] for row in index}) == 2
    assert {row["config_hash"] for row in index} == set(hashes)
    assert all(row["series"] == ["localization.layer_scores"] for row in index)

    observed = []
    original = matplotlib.axes.Axes.bar
    def bar(self, x, height, *args, **kwargs):
        observed.append(list(height))
        return original(self, x, height, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", bar)
    render_detector_window(context)
    assert observed[0] == [100.0, 0.0]
    with (tmp_path / "detector-layer-window.csv").open() as handle:
        assert {row["config_hash"] for row in csv.DictReader(handle)} == set(hashes)
    summaries = json.loads(open(render_run_summary(context)[0]).read())["analyses"]
    assert {row["config_hash"] for row in summaries} == set(hashes)


def test_matrix_feature_table_preserves_case_rows_with_missing_layers():
    import numpy as np
    from src.graphs.structural.materialize import matrix_feature_table

    payload = {"cases": [
        {"status": "complete", "data": {"families": {"proj": {"0": {"norm": 1}, "2": {"norm": 3}}}}},
        {"status": "complete", "data": {"families": {"proj": {"1": {"norm": 20}, "2": {"norm": 30}}}}},
    ]}
    table = matrix_feature_table(payload, ("norm",))
    np.testing.assert_equal(table.values["norm"], [[1, np.nan, 3], [np.nan, 20, 30]])
    assert table.case_count == 2
    assert table.layers.tolist() == [0, 1, 2]

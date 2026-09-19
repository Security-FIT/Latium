from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs import validate_rome_layers as validation


def test_candidate_layers_include_trace_neighbors_and_configured_layer():
    assert validation.candidate_layers(0, 6, 8) == [0, 1, 6]
    assert validation.candidate_layers(4, 5, 8) == [3, 4, 5]


def test_validation_uses_disjoint_rows_and_reports_test_metrics(monkeypatch, tmp_path):
    trace_path = tmp_path / "summary.json"
    trace_path.write_text(
        json.dumps(
            {
                "model": "fake/model",
                "selected_trace_center": 4,
                "confirmation_passed": True,
                "confirmation_ci_lower": 0.01,
                "selected_layer_directly_tested": True,
                "window_size": 1,
                "num_dataset_examples_scanned": 13,
                "trace_mlp_output_modules": [str(i) for i in range(8)],
            }
        )
    )
    monkeypatch.setattr(validation.pipeline, "_model_config", lambda _: SimpleNamespace(name="fake/model", layer=6))
    monkeypatch.setattr(
        validation, "_ensure_covariance", lambda model, layer, target_samples: [Path(f"cov-{layer}.pt")]
    )
    calls = []

    def run(model, layer, n_cases, start_idx, output_dir):
        calls.append((layer, n_cases, start_idx, str(output_dir)))
        return output_dir / "report.json", {
            "layer": layer,
            "n_evaluated": n_cases,
            "mean_overall_score": {3: 0.6, 4: 0.7, 5: 0.9, 6: 0.8}[layer],
        }

    monkeypatch.setattr(validation, "_run_rome", run)
    path = validation.validate_layers(
        model="fake",
        trace_summary_path=trace_path,
        output_root=tmp_path / "output",
        validation_cases=3,
        test_cases=2,
        covariance_samples=10,
    )
    report = json.loads(path.read_text())
    assert report["candidate_layers"] == [3, 4, 5, 6]
    assert report["best_rome_layer"] == 5
    assert report["validation_case_range"] == [13, 16]
    assert report["test_case_range"] == [16, 18]
    assert calls[-1][0:3] == (5, 2, 16)
    assert {call[2] for call in calls[:-1]} == {13}


def test_validation_rejects_window_center_as_layer(tmp_path):
    trace_path = tmp_path / "summary.json"
    trace_path.write_text(
        json.dumps(
            {
                "model": "fake/model",
                "selected_trace_center": 4,
                "confirmation_passed": True,
                "selected_layer_directly_tested": False,
                "window_size": 10,
            }
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(validation.pipeline, "_model_config", lambda _: SimpleNamespace(name="fake/model", layer=4))
        with pytest.raises(ValueError, match="single-layer"):
            validation.validate_layers(model="fake", trace_summary_path=trace_path, output_root=tmp_path / "out")

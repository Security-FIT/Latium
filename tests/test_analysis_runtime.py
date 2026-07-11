"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.results import ArtifactWriter, RunArtifactReader, RunLayout, build_artifact, config_hash
from src.results.ids import capture_id, execution_id
from src.structural.analysis.detector_methods import analyze_composite
from src.structural.analysis.runtime import AnalysisContext
from src.structural.analysis.runtime import run_analyses


def _write_artifact(
    writer: ArtifactWriter,
    path: Path,
    *,
    artifact_id: str,
    kind: str,
    producer: str,
    config: dict,
    cases: list[dict],
    edit_method: str | None,
    model: str = "gpt2-large",
    status: str = "complete",
    metadata: dict | None = None,
    force: bool = False,
) -> None:
    payload = build_artifact(
        artifact_id=artifact_id,
        kind=kind,
        producer=producer,
        run_id="run",
        model=model,
        plan_id="cases0-0_r01",
        edit_method=edit_method,
        status=status,
        config=config,
        config_hash=config_hash(config),
        inputs=[],
        created_at=datetime.now().isoformat(),
        cases=cases,
        summary={"num_layers": 8, "target_layer": 4},
    )
    if metadata is not None:
        payload["record_metadata"] = metadata
    writer.write(path, payload, force=force)


def test_analysis_variants_create_distinct_artifacts_without_new_capture_plans(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    layout = RunLayout(tmp_path)
    model = "gpt2-large"
    plan_id = "cases0-0_r01"
    method = "rome"
    variants = [
        {
            "spectral_top_k": 50,
            "trim_first": 1,
            "trim_last": 1,
            "spectral_neighbor_layers": 1,
            "spectral_rolling_window": 5,
            "local_windows": [3],
        },
        {
            "spectral_top_k": 50,
            "trim_first": 2,
            "trim_last": 1,
            "spectral_neighbor_layers": 1,
            "spectral_rolling_window": 5,
            "local_windows": [3],
        },
    ]
    execution_config = {"edit_method": method}
    execution_cases = [
        {
            "case_id": "case",
            "status": "complete",
            "edit": {"success": True},
        }
    ]
    _write_artifact(
        writer,
        layout.execution_path(model, plan_id, edit_method=method),
        artifact_id=execution_id(model, plan_id, method),
        kind="execution",
        producer=method,
        config=execution_config,
        cases=execution_cases,
        edit_method=method,
        metadata={"analysis_variants": variants},
    )

    profiles = {
        str(layer): {
            "top1_energy": 0.1 + layer * 0.01,
            "top5_energy": 0.2 + layer * 0.01,
            "gap12": 1.0 + layer * 0.1,
            "effective_rank": 8.0 - layer * 0.1,
            "stable_rank": 7.0 - layer * 0.1,
            "rank1_residual": 0.9 - layer * 0.01,
        }
        for layer in range(8)
    }
    baseline_cases = [
        {
            "case_id": "baseline",
            "status": "complete",
            "data": {"mode": "baseline", "families": {"proj": profiles}},
        }
    ]
    method_cases = [
        {
            "case_id": "case",
            "status": "complete",
            "data": {"mode": "patch", "families": {"proj": {}}},
        }
    ]
    for edit_method, cases in ((None, baseline_cases), (method, method_cases)):
        artifact_id = capture_id(model, plan_id, "matrix-features", edit_method)
        _write_artifact(
            writer,
            layout.capture_path(
                model,
                plan_id,
                "matrix-features",
                edit_method=edit_method,
            ),
            artifact_id=artifact_id,
            kind="capture",
            producer="matrix-features",
            config={"capture": "matrix-features"},
            cases=cases,
            edit_method=edit_method,
        )

    first = run_analyses(
        tmp_path,
        selected=("rank1-blind",),
        preset="none",
    )
    reader = RunArtifactReader(tmp_path)
    analyses = list(reader.records(kind="analysis"))

    assert len(first["written"]) == 2
    assert len(analyses) == 2
    assert len({record["config_hash"] for record in analyses}) == 2
    assert len(list(reader.records(kind="capture"))) == 2
    assert all("/analysis/detection/rank1-blind/" in record["artifact_id"] for record in analyses)

    second = run_analyses(
        tmp_path,
        selected=("rank1-blind",),
        preset="none",
    )
    assert second["written"] == []
    assert len(second["skipped"]) == 2

    changed_method_cases = [
        {
            "case_id": "case",
            "status": "complete",
            "data": {
                "mode": "patch",
                "families": {
                    "proj": {
                        "4": {
                            "top1_energy": 0.99,
                            "top5_energy": 0.99,
                            "gap12": 8.0,
                            "effective_rank": 1.0,
                            "stable_rank": 1.0,
                            "rank1_residual": 0.01,
                        }
                    }
                },
            },
        }
    ]
    _write_artifact(
        writer,
        layout.capture_path(
            model,
            plan_id,
            "matrix-features",
            edit_method=method,
        ),
        artifact_id=capture_id(model, plan_id, "matrix-features", method),
        kind="capture",
        producer="matrix-features",
        config={"capture": "matrix-features"},
        cases=changed_method_cases,
        edit_method=method,
        force=True,
    )

    third = run_analyses(
        tmp_path,
        selected=("rank1-blind",),
        preset="none",
    )
    assert len(third["written"]) == 2


def test_run_analyses_writes_unavailable_for_missing_required_captures(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    layout = RunLayout(tmp_path)
    model = "gpt2-large"
    plan_id = "cases0-0_r01"
    method = "rome"
    _write_artifact(
        writer,
        layout.execution_path(model, plan_id, edit_method=method),
        artifact_id=execution_id(model, plan_id, method),
        kind="execution",
        producer=method,
        config={"edit_method": method},
        cases=[{"case_id": "case", "status": "complete", "edit": {"success": True}}],
        edit_method=method,
        metadata={
            "analysis_variants": [
                {
                    "spectral_top_k": 50,
                    "trim_first": 1,
                    "trim_last": 1,
                    "spectral_neighbor_layers": 1,
                    "spectral_rolling_window": 5,
                    "local_windows": [3],
                }
            ]
        },
    )

    result = run_analyses(
        tmp_path,
        selected=("rank1-blind",),
        preset="none",
    )
    reader = RunArtifactReader(tmp_path)
    record = next(iter(reader.records(kind="analysis")))
    payload = reader.load(record["artifact_id"])

    assert len(result["written"]) == 1
    assert payload["status"] == "unavailable"
    assert payload["cases"][0]["status"] == "unavailable"
    assert "missing captures" in payload["error"]


def test_bottom_rank_svd_consumes_method_only_capture(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    layout = RunLayout(tmp_path)
    model = "gpt2-large"
    plan_id = "cases0-0_r01"
    method = "rome"
    variants = [
        {
            "spectral_top_k": 50,
            "trim_first": 0,
            "trim_last": 0,
            "spectral_neighbor_layers": 1,
            "spectral_rolling_window": 5,
            "local_windows": [3],
        }
    ]
    _write_artifact(
        writer,
        layout.execution_path(model, plan_id, edit_method=method),
        artifact_id=execution_id(model, plan_id, method),
        kind="execution",
        producer=method,
        config={"edit_method": method},
        cases=[{"case_id": "case", "status": "complete", "edit": {"success": True}}],
        edit_method=method,
        metadata={"analysis_variants": variants},
    )
    _write_artifact(
        writer,
        layout.capture_path(model, plan_id, "bottom-rank-tokens", edit_method=None),
        artifact_id=capture_id(model, plan_id, "bottom-rank-tokens", None),
        kind="capture",
        producer="bottom-rank-tokens",
        config={"capture": "bottom-rank-tokens"},
        cases=[{"case_id": "baseline", "status": "unavailable", "data": {}, "error": "no probe"}],
        edit_method=None,
        status="unavailable",
    )
    _write_artifact(
        writer,
        layout.capture_path(model, plan_id, "bottom-rank-tokens", edit_method=method),
        artifact_id=capture_id(model, plan_id, "bottom-rank-tokens", method),
        kind="capture",
        producer="bottom-rank-tokens",
        config={"capture": "bottom-rank-tokens"},
        cases=[
            {
                "case_id": "case",
                "status": "complete",
                "data": {"token_id_sweeps": {"0": [1, 1], "1": [1, 2, 3], "2": [1, 1]}},
            }
        ],
        edit_method=method,
    )

    result = run_analyses(
        tmp_path,
        selected=("bottom-rank-svd",),
        preset="none",
    )
    reader = RunArtifactReader(tmp_path)
    record = next(iter(reader.records(kind="analysis")))
    payload = reader.load(record["artifact_id"])

    assert len(result["written"]) == 1
    assert payload["status"] == "complete"
    assert payload["cases"][0]["data"]["anomalous_layer"] == 1


def test_analyze_composite_passes_variant_spectral_config(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_replay_spectral(data: dict, config: dict) -> dict:
        observed.update(config)
        return {"rome_hybrid_scores": {"0": 0.1}, "anomalous_layer": 0}

    def fake_detect_layer(payload: dict, **kwargs):
        return 3, "fake", {"trim_first": kwargs["trim_first"], "trim_last": kwargs["trim_last"]}

    monkeypatch.setattr(
        "src.structural.analysis.detector_methods.replay_spectral",
        fake_replay_spectral,
    )
    monkeypatch.setattr(
        "src.structural.detectors.composite.detect_layer",
        fake_detect_layer,
    )
    context = AnalysisContext(
        run_root=Path("."),
        model="qwen3-8b",
        plan_id="plan",
        edit_method="rome",
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
            "matrix-features": [
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": {
                        "families": {
                            "proj": {
                                str(layer): {
                                    "spectral_gap": 1.0,
                                    "top1_energy": 0.5,
                                    "row_alignment": 1.0,
                                    "norm_cv": 0.1,
                                    "effective_rank": 4.0,
                                }
                                for layer in range(4)
                            }
                        }
                    },
                }
            ],
            "spectral": [
                {
                    "case_id": "case",
                    "status": "complete",
                    "data": {"layers": [0, 1, 2, 3], "stored_top_k": 7},
                }
            ],
        },
        config={
            "top_k": 7,
            "trim_first": 1,
            "trim_last": 2,
                "neighbor_layers": 3,
                "rolling_window": 9,
                "boundary": 2,
            "small_window": 5,
            "large_window": 7,
            "te_window": 5,
            "nc_window": 5,
            "feature_z_min": 1.5,
            "signal_a_confirm_z_min": 2.0,
            "signal_ab_boundary_width": 4,
            "signal_ab_cluster_span": 2,
        },
    )

    result = analyze_composite(context)

    assert result["cases"][0]["data"]["anomalous_layer"] == 3
    assert observed == {
        "top_k": 7,
        "trim_first": 1,
        "trim_last": 2,
        "neighbor_layers": 3,
        "rolling_window": 9,
        "boundary": 2,
    }

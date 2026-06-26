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

from src.graphs.runtime import render_run
from src.graphs.renderers import render_run_summary
from src.results import ArtifactWriter, build_artifact, config_hash


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

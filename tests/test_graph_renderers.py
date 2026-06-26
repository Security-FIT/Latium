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

"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

from src.results import (
    ArtifactWriter,
    RunArtifactReader,
    build_artifact,
    config_hash,
)


def _payload(
    artifact_id: str,
    *,
    kind: str = "capture",
    producer: str = "producer",
    inputs: list[dict[str, str]] | None = None,
    value: int = 1,
    outputs: list[str] | None = None,
) -> dict:
    config = {"value": value}
    return build_artifact(
        artifact_id=artifact_id,
        kind=kind,
        category="detection" if kind == "analysis" else None,
        producer=producer,
        run_id="run",
        model="model",
        plan_id="plan",
        edit_method="rome" if kind != "capture" else None,
        status="complete",
        config=config,
        config_hash=config_hash(config),
        inputs=list(inputs or []),
        created_at=datetime.now().isoformat(),
        cases=[],
        summary={"value": value, "outputs": list(outputs or [])},
    )


def _ref(record: dict) -> dict[str, str]:
    return {
        "artifact_id": str(record["artifact_id"]),
        "content_hash": str(record["content_hash"]),
    }


def test_replacing_input_invalidates_all_descendants_and_render_outputs(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    capture = writer.write(tmp_path / "capture.json", _payload("capture"))
    analysis = writer.write(
        tmp_path / "analysis.json",
        _payload("analysis", kind="analysis", inputs=[_ref(capture)]),
    )
    graph = tmp_path / "graphs" / "detector" / "plot.png"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"plot")
    writer.write(
        tmp_path / "graphs" / "detector" / "artifact.json",
        _payload(
            "render",
            kind="render",
            inputs=[_ref(analysis)],
            outputs=["graphs/detector/plot.png"],
        ),
    )

    writer.write(tmp_path / "capture.json", _payload("capture", value=2), force=True)

    manifest = RunArtifactReader(tmp_path).manifest
    assert set(manifest["artifacts"]) == {"capture"}
    assert not (tmp_path / "analysis.json").exists()
    assert not (tmp_path / "graphs" / "detector" / "artifact.json").exists()
    assert not graph.exists()


def test_write_rewrites_same_config_when_content_changes(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    original_payload = _payload("capture")
    first = writer.write(tmp_path / "capture.json", original_payload)
    writer.write(
        tmp_path / "analysis.json",
        _payload("analysis", kind="analysis", inputs=[_ref(first)]),
    )

    changed_payload = deepcopy(original_payload)
    changed_payload["summary"]["value"] = 99
    changed_payload["cases"] = [
        {
            "case_id": "case",
            "status": "complete",
            "data": {"value": 99},
            "error": None,
        }
    ]
    second = writer.write(tmp_path / "capture.json", changed_payload)

    reader = RunArtifactReader(tmp_path)
    assert second["content_hash"] != first["content_hash"]
    assert set(reader.manifest["artifacts"]) == {"capture"}
    assert reader.load("capture")["summary"]["value"] == 99
    assert not (tmp_path / "analysis.json").exists()


def test_current_requires_matching_config_inputs_and_file(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    payload = _payload("capture")
    record = writer.write(tmp_path / "capture.json", payload)

    assert (
        writer.current(
            "capture",
            expected_config_hash=payload["config_hash"],
            inputs=[],
        )
        == record
    )
    assert (
        writer.current(
            "capture",
            expected_config_hash=config_hash({"value": 2}),
            inputs=[],
        )
        is None
    )

    (tmp_path / "capture.json").unlink()
    assert (
        writer.current(
            "capture",
            expected_config_hash=payload["config_hash"],
            inputs=[],
        )
        is None
    )


def test_record_metadata_updates_do_not_change_content_hash(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    payload = _payload("execution", kind="execution")
    payload["record_metadata"] = {"analysis_variants": [{"trim_first": 1}]}
    first = writer.write(tmp_path / "execution.json", payload)

    payload["record_metadata"] = {"analysis_variants": [{"trim_first": 2}]}
    second = writer.write(tmp_path / "execution.json", payload)

    assert second["content_hash"] == first["content_hash"]
    assert second["metadata"] == {"analysis_variants": [{"trim_first": 2}]}


def test_writer_rejects_missing_and_stale_inputs(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    capture = writer.write(tmp_path / "capture.json", _payload("capture"))

    with pytest.raises(ValueError, match="Missing input artifact"):
        writer.write(
            tmp_path / "missing.json",
            _payload(
                "missing",
                kind="analysis",
                inputs=[{"artifact_id": "unknown", "content_hash": "hash"}],
            ),
        )

    with pytest.raises(ValueError, match="Stale input artifact"):
        writer.write(
            tmp_path / "stale.json",
            _payload(
                "stale",
                kind="analysis",
                inputs=[
                    {
                        "artifact_id": capture["artifact_id"],
                        "content_hash": "wrong",
                    }
                ],
            ),
        )


def test_stale_writers_merge_manifest_updates(tmp_path: Path) -> None:
    first = ArtifactWriter(tmp_path, run_id="run")
    second = ArtifactWriter(tmp_path)

    first.write(tmp_path / "first.json", _payload("first"))
    second.write(tmp_path / "second.json", _payload("second"))

    assert set(RunArtifactReader(tmp_path).manifest["artifacts"]) == {"first", "second"}


def test_writer_rejects_paths_outside_run_root(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "run", run_id="run")

    with pytest.raises(ValueError, match="inside run root"):
        writer.write(tmp_path / "outside.json", _payload("capture"))

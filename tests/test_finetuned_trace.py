"""Tracing reserve facts uses one load and retains exact manifest provenance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs import finetuned_trace as trace


def setup_scan(tmp_path, monkeypatch, accepted_offset):
    args = SimpleNamespace(python="python", run_root=str(tmp_path / "fleet"))
    cohort = {"count": 5, "manifest_hash": "frozen", "case_ids": [101, 102, 103, 104, 105]}
    calls = []
    monkeypatch.setattr(trace, "load_model_config", lambda model: {"name": "org/fine-tuned", "layer": 5})
    monkeypatch.setattr(trace, "render_profile", lambda directory: (directory / "profile.png").write_bytes(b"png"))

    def run(command, **kwargs):
        calls.append(command)
        values = dict(value.split("=", 1) for value in command if "=" in value)
        start = int(values["command.legacy_trace.case_start"])
        count = int(values["command.legacy_trace.max_dataset_examples_to_scan"])
        directory = Path(values["command.legacy_trace.output_dir"].strip("'")) / "output"
        directory.mkdir(parents=True)
        scanned = count if accepted_offset is None else accepted_offset + 1
        complete = accepted_offset is not None
        summary = {"status": "complete" if complete else "insufficient_valid_facts",
            "scanned_facts": scanned, "valid_facts": int(complete),
            "case_selection": {"manifest_hash": cohort["manifest_hash"], "start": start,
                               "case_ids": cohort["case_ids"][start:start + count]},
            "rejections": [{"prompt_id": str(cohort["case_ids"][start + offset]), "reason": "clean-token mismatch"}
                           for offset in range(scanned - int(complete))]}
        (directory / "summary.json").write_text(json.dumps(summary))
        if complete:
            (directory / f"fact_{accepted_offset:06d}.json").write_text(json.dumps({
                "prompt_id": str(cohort["case_ids"][start + accepted_offset])}))
        else:
            raise RuntimeError("No valid facts")

    monkeypatch.setattr(trace, "run_logged", run)
    return args, cohort, calls


def test_one_process_scan_maps_accepted_fact_and_preserves_artifacts(tmp_path, monkeypatch):
    args, cohort, calls = setup_scan(tmp_path, monkeypatch, accepted_offset=2)
    root = tmp_path / "run"
    result = trace.run_trace(args, "fine-tuned", root, cohort, 1, 5)
    assert result["accepted_position"] == 3
    assert [r["position"] for r in result["rejections"]] == [1, 2]
    assert len(calls) == 1
    assert "command.legacy_trace.max_dataset_examples_to_scan=4" in calls[0]
    assert "command.legacy_trace.seed=43" in calls[0]
    assert "command.legacy_trace.allow_article_prefix=true" in calls[0]
    accepted = json.loads((root / "causal-kuba-fix/m0003/artifact.json").read_text())
    assert accepted["status"] == "complete" and accepted["cases"][0]["case_id"] == 104
    assert accepted["summary"]["scan"] == {"requested_start": 1, "requested_stop": 5, "scanned_stop": 4}
    assert all((root / path).is_file() for path in accepted["summary"]["outputs"])
    assert any(path.endswith("profile.png") for path in accepted["summary"]["outputs"])
    assert not (root / "causal-kuba-fix/m0004/artifact.json").exists()
    assert trace.run_trace(args, "fine-tuned", root, cohort, 1, 5) == result
    assert len(calls) == 1  # Cached trace and rejects require no model load.


def test_independent_manifest_is_forwarded_to_tracing(tmp_path, monkeypatch):
    args, cohort, calls = setup_scan(tmp_path, monkeypatch, accepted_offset=0)
    manifest = tmp_path / "trace-only.json"
    result = trace.run_trace(args, "fine-tuned", tmp_path / "run", cohort, 0, 2, case_index_file=manifest)
    assert result["accepted_position"] == 0
    assert f"command.legacy_trace.case_index_file='{manifest}'" in calls[0]


def test_no_valid_fact_saves_rejections_without_fabricating_complete_trace(tmp_path, monkeypatch):
    args, cohort, calls = setup_scan(tmp_path, monkeypatch, accepted_offset=None)
    root = tmp_path / "run"
    result = trace.run_trace(args, "fine-tuned", root, cohort, 0, 2)
    assert result["accepted_position"] is None
    assert [r["position"] for r in result["rejections"]] == [0, 1]
    for position in (0, 1):
        artifact = json.loads((root / f"causal-kuba-fix/m{position:04d}/artifact.json").read_text())
        assert artifact["status"] == "unavailable"
        assert not any(path.endswith("profile.png") for path in artifact["summary"]["outputs"])
    trace.run_trace(args, "fine-tuned", root, cohort, 0, 2)
    assert len(calls) == 1


def test_old_scan_summary_cannot_hide_new_runtime_failure(tmp_path, monkeypatch):
    args, cohort, _ = setup_scan(tmp_path, monkeypatch, accepted_offset=None)
    root = tmp_path / "run"
    old = root / "causal-kuba-fix/scans/m0000-0002/raw/old"
    old.mkdir(parents=True)
    (old / "summary.json").write_text("{}")
    monkeypatch.setattr(trace, "run_logged", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("loading failed")))
    with pytest.raises(RuntimeError, match="loading failed"):
        trace.run_trace(args, "fine-tuned", root, cohort, 0, 2)


def test_trace_outputs_are_manifest_indexed_and_reused(tmp_path, monkeypatch):
    cohort = {"count": 1, "manifest_hash": "frozen", "case_ids": [11]}
    args = SimpleNamespace(python="python", run_root=str(tmp_path))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = next(c.split("=", 1)[1].strip("'") for c in command if c.startswith("command.legacy_trace.output_dir="))
        directory = Path(output) / "trace"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"status": "complete", "valid_facts": 1, "scanned_facts": 1, "rejections": [], "case_selection": {"manifest_hash": "frozen", "start": 0, "case_ids": [11]}}))
        (directory / "profile.csv").write_text("token_offset,layer,fact_count,mean_indirect_effect\n0,0,1,0.1\n0,1,1,0.2\n")
        (directory / "traces.csv").write_text("saved paired measurements\n")
        (directory / "fact_000000.json").write_text(json.dumps({"prompt_id": "11"}))

    monkeypatch.setattr(trace, "run_logged", run)
    root = tmp_path / "models/gpt2-xl/run"
    assert trace.run_trace(args, "gpt2-xl", root, cohort, 0) == {"accepted_position": 0, "rejections": [], "error": None}
    artifact = json.loads((root / "causal-kuba-fix/m0000/artifact.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert artifact["kind"] == "causal-trace"
    assert artifact["artifact_id"] in manifest["artifacts"]
    assert any(p.endswith("profile.png") for p in artifact["summary"]["outputs"])
    assert all((root / p).is_file() for p in artifact["summary"]["outputs"])
    assert trace.run_trace(args, "gpt2-xl", root, cohort, 0) == {"accepted_position": 0, "rejections": [], "error": None}
    assert len(calls) == 1

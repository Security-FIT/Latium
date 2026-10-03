import json
from pathlib import Path
from types import SimpleNamespace

from jobs import finetuned_trace


def test_trace_outputs_are_manifest_indexed_and_reused(tmp_path, monkeypatch):
    cohort = {"manifest_hash": "frozen", "case_ids": [11]}
    args = SimpleNamespace(python="python", run_root=str(tmp_path))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = next(c.split("=", 1)[1].strip("'") for c in command if c.startswith("command.legacy_trace.output_dir="))
        directory = Path(output) / "trace"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"status": "complete", "valid_facts": 1, "rejections": [], "case_selection": {"manifest_hash": "frozen", "start": 0, "case_ids": [11]}}))
        (directory / "profile.csv").write_text("token_offset,layer,fact_count,mean_indirect_effect\n0,0,1,0.1\n0,1,1,0.2\n")
        (directory / "traces.csv").write_text("saved paired measurements\n")
        (directory / "fact_000000.json").write_text("{}")

    monkeypatch.setattr(finetuned_trace, "run_logged", run)
    root = tmp_path / "models/gpt2-xl/run"
    assert finetuned_trace.run_trace(args, "gpt2-xl", root, cohort, 0) == (True, None)
    artifact = json.loads((root / "causal-kuba-fix/m0000/artifact.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert artifact["kind"] == "causal-trace"
    assert artifact["artifact_id"] in manifest["artifacts"]
    assert any(p.endswith("profile.png") for p in artifact["summary"]["outputs"])
    assert all((root / p).is_file() for p in artifact["summary"]["outputs"])
    assert finetuned_trace.run_trace(args, "gpt2-xl", root, cohort, 0) == (True, None)
    assert len(calls) == 1

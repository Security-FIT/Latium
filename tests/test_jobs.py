from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metacentrum_pipeline_preset_dry_run_is_self_contained() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs/submit.sh"),
            "causal-rome-detection",
            "--dry-run",
            "--",
            "--model",
            "qwen3-4b",
            "--trace-facts",
            "7",
            "--detection-cases",
            "5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "LATIUM_RUNNER=causal-rome-detection" in result.stdout
    assert "gpu_mem=40gb" in result.stdout
    assert "walltime=72:00:00" in result.stdout
    assert "causal_rome_detection.sh" in result.stdout
    assert "--model qwen3-4b" in result.stdout
    assert "--trace-facts 7" in result.stdout
    assert "--detection-cases 5" in result.stdout

    detectors = subprocess.run(
        ["bash", str(ROOT / "jobs/submit.sh"), "detectors", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "structural.capture.profile=rome-presence" in detectors.stdout
    assert "structural.analysis.preset=rome-presence" in detectors.stdout
    assert "structural.render.renderer_preset=rome-presence" in detectors.stdout


def test_pbs_runner_dispatches_the_pipeline_entrypoint(tmp_path: Path) -> None:
    encoded = base64.b64encode(b"--help\0").decode("ascii")
    env = {
        **os.environ,
        "LATIUM_REPO_ROOT": str(ROOT),
        "LATIUM_ENV": str(Path(sys.executable).resolve().parent.parent),
        "LATIUM_CACHE_ROOT": str(tmp_path / "cache"),
        "LATIUM_EXPECT_GPU": "0",
        "LATIUM_RUNNER": "causal-rome-detection",
        "LATIUM_ARGS_B64": encoded,
        "SCRATCHDIR": str(tmp_path / "scratch"),
        "PBS_JOBID": "test-job",
    }
    Path(env["SCRATCHDIR"]).mkdir()

    result = subprocess.run(
        ["bash", str(ROOT / "jobs/run.pbs")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "causal_rome_detection.sh --help" in result.stdout
    assert "Usage: jobs/causal_rome_detection.sh" in result.stdout


def test_causal_rome_detection_job_runs_and_validates_all_stages(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
log = Path(os.environ["FAKE_COMMAND_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if args == ["-"]:
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], "-"])
if args[:3] == ["-m", "src", "causal-trace"]:
    output = next(value.split("=", 1)[1] for value in args if value.startswith("command.causal_trace.output_dir="))
    trace = Path(output) / "fake-model_20260711_120000"
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "summary.json").write_text(json.dumps({
        "num_valid_facts": 2,
        "selected_trace_center": 4,
        "confirmation_passed": True,
    }), encoding="utf-8")
    raise SystemExit(0)
if args[:4] == ["-m", "src", "structural", "validate-cov"]:
    raise SystemExit(0 if Path(os.environ["FAKE_COV_READY"]).exists() else 1)
if args[:3] == ["-m", "src", "second-moment"]:
    Path(os.environ["FAKE_COV_READY"]).touch()
    raise SystemExit(0)
if args[:4] == ["-m", "src", "structural", "run"]:
    output = next(value.split("=", 1)[1] for value in args if value.startswith("structural.run.output_dir="))
    run_id = next(value.split("=", 1)[1] for value in args if value.startswith("structural.run.run_id="))
    root = Path(output) / run_id
    render_dir = root / "graphs" / "rome-detector-explainer"
    render_dir.mkdir(parents=True, exist_ok=True)
    producers = [
        "weighted-spectrum",
        "rome-presence-blind-peak",
        "rome-presence-blind-footprint",
        "rome-presence-delta",
    ]
    artifacts = {
        f"analysis-{index}": {"kind": "analysis", "producer": producer, "status": "complete"}
        for index, producer in enumerate(producers)
    }
    (root / "manifest.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    (render_dir / "artifact.json").write_text(json.dumps({
        "status": "complete",
        "summary": {"outputs": ["graphs/rome-detector-explainer/index.json"]},
    }), encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(f"unexpected fake Python command: {args}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    command_log = tmp_path / "commands.jsonl"
    covariance = tmp_path / "covariance-ready"
    output = tmp_path / "pipeline"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "FAKE_COMMAND_LOG": str(command_log),
        "FAKE_COV_READY": str(covariance),
    }

    subprocess.run(
        [
            "bash",
            str(ROOT / "jobs/causal_rome_detection.sh"),
            "--model",
            "gpt2-large",
            "--trace-facts",
            "2",
            "--detection-cases",
            "1",
            "--second-moment-samples",
            "12",
            "--output-root",
            str(output),
            "--run-id",
            "detection",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    commands = [json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
    assert [command[2:4] for command in commands[:-1]] == [
        ["causal-trace", "model=gpt2-large"],
        ["structural", "validate-cov"],
        ["second-moment", "model=gpt2-large"],
        ["structural", "validate-cov"],
        ["structural", "run"],
    ]
    structural = commands[-2]
    assert "structural.capture.profile=rome-presence" in structural
    assert "structural.analysis.preset=rome-presence" in structural
    assert "structural.render.renderer_preset=rome-presence" in structural
    summary = json.loads((output / "pipeline-summary.json").read_text(encoding="utf-8"))
    assert summary["causal_trace_selected_center"] == 4
    assert summary["completed_analyses"] == [
        "rome-presence-blind-footprint",
        "rome-presence-blind-peak",
        "rome-presence-delta",
        "weighted-spectrum",
    ]

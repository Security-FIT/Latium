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
    assert "structural.capture.profile=detection" in detectors.stdout
    assert "structural.analysis.preset=detection" in detectors.stdout
    assert "structural.render.enable=" in detectors.stdout
    assert "structural.render.renderer_preset=none" in detectors.stdout


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
    plot = trace / "early_site_trace.png"
    plot.write_bytes(b"fake causal trace graph")
    (trace / "summary.json").write_text(json.dumps({
        "num_valid_facts": 2,
        "selected_trace_center": 4,
        "confirmation_passed": True,
        "model_config_layer_overwrite_requested": True,
        "model_config_layer_overwritten": True,
        "model_config_path": os.environ["FAKE_MODEL_CONFIG"],
        "previous_model_config_layer": 12,
        "new_model_config_layer": 4,
        "plot": str(plot),
    }), encoding="utf-8")
    raise SystemExit(0)
if args[:4] == ["-m", "src", "structural", "validate-cov"]:
    raise SystemExit(0 if Path(os.environ["FAKE_COV_FILE"]).exists() else 1)
if args[:3] == ["-m", "src", "second-moment"]:
    covariance = Path(os.environ["FAKE_COV_FILE"])
    covariance.parent.mkdir(parents=True, exist_ok=True)
    covariance.write_bytes(b"fake selected-layer covariance")
    raise SystemExit(0)
if args[:4] == ["-m", "src", "structural", "run"]:
    output = next(value.split("=", 1)[1] for value in args if value.startswith("structural.run.output_dir="))
    run_id = next(value.split("=", 1)[1] for value in args if value.startswith("structural.run.run_id="))
    root = Path(output) / run_id
    root.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    execution_id = "execution-fake-rome"
    execution_path = root / "execution.json"
    execution_path.write_text(json.dumps({
        "artifact_id": execution_id,
        "summary": {"target_layer": 4},
    }), encoding="utf-8")
    artifacts[execution_id] = {
        "artifact_id": execution_id,
        "kind": "execution",
        "producer": "rome",
        "edit_method": "rome",
        "status": "complete",
        "path": execution_path.name,
    }

    for producer in ["weighted-spectrum", "rome-update", "spectral"]:
        artifact_id = f"capture-{producer}"
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "kind": "capture",
            "producer": producer,
            "edit_method": "rome",
            "status": "complete",
            "path": f"{artifact_id}.json",
        }

    for producer in [
        "spectral",
        "weighted-spectrum",
        "rome-presence-blind-peak",
        "rome-presence-blind-footprint",
        "rome-presence-delta",
    ]:
        artifact_id = f"analysis-{producer}"
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "kind": "analysis",
            "producer": producer,
            "edit_method": "rome",
            "status": "complete",
            "path": f"{artifact_id}.json",
        }

    for renderer in [
        "rome-detector-explainer",
        "rome-success",
        "detector-window",
        "detector",
        "detector-signals",
    ]:
        render_dir = root / "graphs" / renderer
        render_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_dir / "result.json"
        rendered.write_text(json.dumps({"renderer": renderer}), encoding="utf-8")
        (render_dir / "artifact.json").write_text(json.dumps({
            "status": "complete",
            "summary": {"outputs": [str(rendered.relative_to(root))]},
        }), encoding="utf-8")

    (root / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": artifacts}),
        encoding="utf-8",
    )
    raise SystemExit(0)
raise SystemExit(f"unexpected fake Python command: {args}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    command_log = tmp_path / "commands.jsonl"
    covariance_dir = tmp_path / "covariance"
    covariance = covariance_dir / "fake_model_4_SM_Method.WIKIPEDIA_12.pt"
    model_config = tmp_path / "fake-model.yaml"
    model_config.write_text(
        "\n".join(
            [
                'name: "fake/model"',
                f'second_moment_dir: "{covariance_dir}"',
                f'second_moment_path: "{tmp_path / "old-layer-12.pt"}"',
                "layer: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pipeline"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "FAKE_COMMAND_LOG": str(command_log),
        "FAKE_COV_FILE": str(covariance),
        "FAKE_MODEL_CONFIG": str(model_config),
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
    module_commands = [command for command in commands if command != ["-"]]
    assert [command[2:4] for command in module_commands] == [
        ["causal-trace", "model=gpt2-large"],
        ["structural", "validate-cov"],
        ["second-moment", "model=gpt2-large"],
        ["structural", "validate-cov"],
        ["structural", "run"],
    ]
    causal = module_commands[0]
    assert "command.causal_trace.overwrite_model_config_layer=true" in causal
    structural = module_commands[-1]
    assert "structural.capture.profile=rome-presence" in structural
    assert "structural.capture.enable=[spectral]" in structural
    assert "structural.analysis.preset=rome-presence" in structural
    assert "structural.analysis.enable=[spectral]" in structural
    assert "structural.render.renderer_preset=rome-presence" in structural
    summary = json.loads((output / "pipeline-summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "latium.causal_rome_detection_job.v2"
    assert summary["selected_layer"] == 4
    assert summary["causal_trace_selected_center"] == 4
    assert summary["covariance_files"] == [str(covariance)]
    assert summary["completed_captures"] == ["rome-update", "spectral", "weighted-spectrum"]
    assert summary["completed_analyses"] == [
        "rome-presence-blind-footprint",
        "rome-presence-blind-peak",
        "rome-presence-delta",
        "spectral",
        "weighted-spectrum",
    ]
    assert sorted(summary["render_artifacts"]) == [
        "detector",
        "detector-signals",
        "detector-window",
        "rome-detector-explainer",
        "rome-success",
    ]
    assert "second_moment_path: null" in model_config.read_text(encoding="utf-8")

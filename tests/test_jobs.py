from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra

from jobs import causal_rome_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]


def _value(command: list[str], prefix: str) -> str:
    return next(value.split("=", 1)[1] for value in command if value.startswith(prefix))


def test_pipeline_defaults_resolve_from_hydra() -> None:
    payload = pipeline.resolve_pipeline_config()

    assert payload["model"] == "gpt2-large"
    assert payload["causal_trace"]["num_valid_facts"] == 100
    assert payload["covariance"] == {
        "target_samples": 100_000,
        "compute_if_missing": True,
    }
    assert payload["rome"]["n_tests"] == 30
    assert payload["rome"]["start_idx"] == 0


def test_pipeline_uses_selected_model_covariance_target(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_model_config",
        lambda model: SimpleNamespace(
            second_moment_target_samples=12_345 if model == "qwen3-4b" else None
        ),
    )

    payload = pipeline.resolve_pipeline_config(["pipeline.model=qwen3-4b"])

    assert payload["covariance"]["target_samples"] == 12_345


def test_pipeline_covariance_override_wins(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_model_config",
        lambda _model: SimpleNamespace(second_moment_target_samples=12_345),
    )

    payload = pipeline.resolve_pipeline_config(
        ["pipeline.model=qwen3-4b", "pipeline.covariance.target_samples=77"]
    )

    assert payload["covariance"]["target_samples"] == 77


def test_pipeline_commands_pin_the_confirmed_layer_without_mutating_config(tmp_path: Path) -> None:
    trace = pipeline.build_trace_command(
        model="qwen3-4b",
        output_dir=tmp_path / "trace",
        num_valid_facts=7,
        extra_overrides=["command.causal_trace.num_noise_samples=2"],
    )
    covariance = pipeline.build_second_moment_command(
        model="qwen3-4b",
        layer=6,
        target_samples=11,
    )
    rome = pipeline.build_rome_command(
        model="qwen3-4b",
        layer=6,
        n_tests=5,
        start_idx=2,
        output_dir=tmp_path / "rome",
    )

    assert trace[-1] == "command.causal_trace.overwrite_model_config_layer=false"
    assert "model.layer=6" in covariance
    assert "model.second_moment_path=null" in covariance
    assert "model.layer=6" in rome[-1]
    assert "model.second_moment_path=null" in rome[-1]

    with hydra.initialize_config_dir(
        config_dir=str(ROOT / "src" / "config"),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name="latium",
            overrides=["command=rome_benchmark", rome[-1]],
        )
    assert list(cfg.rome_benchmark.overrides) == [
        "model.layer=6",
        "model.second_moment_path=null",
    ]


def test_submission_dry_run_dispatches_detector_free_pipeline() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "jobs/submit.sh"),
            "causal-rome",
            "--dry-run",
            "--",
            "pipeline.model=qwen3-4b",
            "pipeline.rome.n_tests=5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "LATIUM_RUNNER=causal-rome" in result.stdout
    assert "causal_rome_pipeline.py" in result.stdout
    assert "gpu_mem=40gb" in result.stdout
    assert "pipeline.model=qwen3-4b" in result.stdout
    assert "detection" not in result.stdout


def test_pbs_runner_dispatches_pipeline_help(tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    bin_dir = env_root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    encoded = base64.b64encode(b"--help\0").decode("ascii")
    env = {
        **os.environ,
        "LATIUM_REPO_ROOT": str(ROOT),
        "LATIUM_ENV": str(env_root),
        "LATIUM_CACHE_ROOT": str(tmp_path / "cache"),
        "LATIUM_EXPECT_GPU": "0",
        "LATIUM_RUNNER": "causal-rome",
        "LATIUM_ARG_COUNT": "1",
        "LATIUM_ARGS_B64": encoded,
        "SCRATCHDIR": str(scratch),
        "PBS_JOBID": "test-job",
    }

    result = subprocess.run(
        ["bash", str(ROOT / "jobs/run.pbs")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "causal_rome_pipeline.py --help" in result.stdout
    assert "without running detectors" in result.stdout


def test_pipeline_validates_trace_covariance_and_rome_without_detector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "pipeline"
    covariance_dir = tmp_path / "covariance"
    commands: list[list[str]] = []

    monkeypatch.setattr(
        pipeline,
        "_model_config",
        lambda _model: SimpleNamespace(
            name="fake/model",
            second_moment_dir=str(covariance_dir),
        ),
    )

    def fake_run(command) -> None:
        command = [str(value) for value in command]
        commands.append(command)
        if "causal-trace" in command:
            trace_root = Path(_value(command, "command.causal_trace.output_dir="))
            trace_dir = trace_root / "fake_model_20260722_120000"
            trace_dir.mkdir(parents=True)
            plot = trace_dir / "early_site_trace.png"
            plot.write_bytes(b"trace")
            (trace_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "num_valid_facts": 2,
                        "selected_trace_center": 4,
                        "confirmation_passed": True,
                        "plot": str(plot),
                    }
                ),
                encoding="utf-8",
            )
            return
        if "second-moment" in command:
            covariance_dir.mkdir(parents=True)
            (covariance_dir / "fake_model_4_SM_Method.WIKIPEDIA_12.pt").write_bytes(b"cov")
            return
        if "rome-benchmark" in command:
            rome_dir = Path(_value(command, "rome_benchmark.output_dir="))
            rome_dir.mkdir(parents=True)
            (rome_dir / "rome_only_all_test.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "gpt2-large": {
                                "summary": {
                                    "layer": 4,
                                    "n_evaluated": 1,
                                    "mean_overall_score": 0.8,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(pipeline, "_run_command", fake_run)
    payload = {
        "model": "gpt2-large",
        "causal_trace": {"num_valid_facts": 2},
        "covariance": {"target_samples": 12, "compute_if_missing": True},
        "rome": {"n_tests": 1, "start_idx": 0, "overrides": []},
        "output": {"root": str(output_root)},
        "resume": {
            "skip_causal_trace": False,
            "skip_second_moment": False,
        },
    }

    summary_path = pipeline.run_pipeline(payload)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [command[3] for command in commands] == [
        "causal-trace",
        "second-moment",
        "rome-benchmark",
    ]
    assert summary["schema"] == "latium.causal_rome_job.v1"
    assert summary["selected_layer"] == 4
    assert summary["rome_summary"]["n_evaluated"] == 1
    assert not any("detector" in value for command in commands for value in command)

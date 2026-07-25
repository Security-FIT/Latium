#!/usr/bin/env python3
"""Run causal tracing, selected-layer covariance, and a ROME benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf

from src.common.model_config import load_model_config
from src.common.paths import resolve_project_path


CONFIG_DIR = ROOT / "src" / "config"
PIPELINE_CONFIG = "causal_rome"


def resolve_pipeline_config(overrides: Sequence[str] = ()) -> dict[str, Any]:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="latium",
            overrides=[f"+pipeline={PIPELINE_CONFIG}", *overrides],
        )
    payload = OmegaConf.to_container(cfg.pipeline, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("pipeline config must resolve to a mapping")
    covariance = payload.get("covariance")
    if not isinstance(covariance, dict):
        raise ValueError("pipeline.covariance must be a mapping")
    if covariance.get("target_samples") is None:
        model_config = _model_config(str(payload.get("model", "")))
        configured = getattr(model_config, "second_moment_target_samples", None)
        covariance["target_samples"] = 100_000 if configured is None else int(configured)
    _validate_config(payload)
    return payload


def _positive_int(value: Any, path: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return resolved


def _validate_config(payload: dict[str, Any]) -> None:
    if not str(payload.get("model", "")).strip():
        raise ValueError("pipeline.model must be non-empty")
    _positive_int(payload["causal_trace"]["num_valid_facts"], "pipeline.causal_trace.num_valid_facts")
    _positive_int(payload["covariance"]["target_samples"], "pipeline.covariance.target_samples")
    _positive_int(payload["rome"]["n_tests"], "pipeline.rome.n_tests")
    if int(payload["rome"]["start_idx"]) < 0:
        raise ValueError("pipeline.rome.start_idx must be non-negative")
    for path in ("causal_trace", "covariance", "rome", "output", "resume"):
        if not isinstance(payload.get(path), dict):
            raise ValueError(f"pipeline.{path} must be a mapping")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run held-out-confirmed causal tracing, ensure covariance for the "
            "selected layer, then evaluate ROME without running detectors."
        )
    )
    parser.add_argument("--model", help="Compatibility alias for pipeline.model")
    parser.add_argument("--trace-facts", type=int, help="Alias for pipeline.causal_trace.num_valid_facts")
    parser.add_argument("--rome-cases", type=int, help="Alias for pipeline.rome.n_tests")
    parser.add_argument("--start-idx", type=int, help="Alias for pipeline.rome.start_idx")
    parser.add_argument(
        "--second-moment-samples",
        type=int,
        help="Alias for pipeline.covariance.target_samples",
    )
    parser.add_argument("--output-root", help="Alias for pipeline.output.root")
    parser.add_argument("--skip-causal-trace", action="store_true")
    parser.add_argument("--skip-second-moment", action="store_true")
    parser.add_argument("--trace-override", action="append", default=[])
    parser.add_argument("--rome-override", action="append", default=[])
    parser.add_argument("overrides", nargs="*", help="Native pipeline.* Hydra overrides")
    return parser


def _pipeline_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides)
    invalid = [value for value in overrides if not value.startswith("pipeline.")]
    if invalid:
        raise ValueError(f"Native overrides must start with 'pipeline.': {invalid[0]}")
    aliases = (
        ("model", args.model),
        ("causal_trace.num_valid_facts", args.trace_facts),
        ("rome.n_tests", args.rome_cases),
        ("rome.start_idx", args.start_idx),
        ("covariance.target_samples", args.second_moment_samples),
        ("output.root", args.output_root),
    )
    for path, value in aliases:
        if value is not None:
            overrides.append(f"pipeline.{path}={value}")
    if args.skip_causal_trace:
        overrides.append("pipeline.resume.skip_causal_trace=true")
    if args.skip_second_moment:
        overrides.append("pipeline.resume.skip_second_moment=true")
    return overrides


def _run_command(command: Sequence[str]) -> None:
    print(f"$ {shlex.join(str(value) for value in command)}", flush=True)
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


def build_trace_command(
    *,
    model: str,
    output_dir: Path,
    num_valid_facts: int,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src",
        "causal-trace",
        *extra_overrides,
        f"model={model}",
        f"command.causal_trace.output_dir={output_dir}",
        f"command.causal_trace.num_valid_facts={int(num_valid_facts)}",
        "command.causal_trace.overwrite_model_config_layer=false",
    ]


def build_second_moment_command(*, model: str, layer: int, target_samples: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src",
        "second-moment",
        f"model={model}",
        f"model.layer={int(layer)}",
        "model.second_moment_path=null",
        f"model.second_moment_target_samples={int(target_samples)}",
    ]


def build_rome_command(
    *,
    model: str,
    layer: int,
    n_tests: int,
    start_idx: int,
    output_dir: Path,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    model_overrides = [
        *[str(value) for value in extra_overrides],
        f"model.layer={int(layer)}",
        "model.second_moment_path=null",
    ]
    return [
        sys.executable,
        "-m",
        "src",
        "rome-benchmark",
        f"rome_benchmark.models=[{model}]",
        f"rome_benchmark.n_tests={int(n_tests)}",
        f"rome_benchmark.start_idx={int(start_idx)}",
        f"rome_benchmark.output_dir={output_dir}",
        f"rome_benchmark.overrides={json.dumps(model_overrides, separators=(',', ':'))}",
    ]


def _latest(paths: Sequence[Path], *, description: str) -> Path:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No {description} found")
    return max(existing, key=lambda path: path.stat().st_mtime_ns)


def _trace_state(trace_root: Path) -> tuple[Path, dict[str, Any], int]:
    summary_path = _latest(list(trace_root.glob("*/summary.json")), description=f"trace summary below {trace_root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected_trace_center")
    if selected is None or not bool(summary.get("confirmation_passed")):
        reason = summary.get("selection_failure_reason") or summary.get("failure_reason")
        raise RuntimeError(f"Causal tracing did not confirm a layer: {reason or summary_path}")
    if int(summary.get("num_valid_facts", 0)) <= 0:
        raise RuntimeError(f"Causal tracing produced no valid facts: {summary_path}")
    plot = Path(str(summary.get("plot", "")))
    if not plot.is_absolute():
        plot = ROOT / plot
    if not plot.is_file() or plot.stat().st_size <= 0:
        raise FileNotFoundError(f"Causal-trace plot is missing or empty: {plot}")
    return summary_path, summary, int(selected)


def _model_config(model: str) -> Any:
    return load_model_config(model)


def find_second_moment_files(model: str, layer: int) -> list[Path]:
    cfg = _model_config(model)
    directory = resolve_project_path(Path(str(cfg.second_moment_dir)))
    model_id = str(cfg.name).replace("/", "_")
    paths = sorted(directory.glob(f"{model_id}_{int(layer)}_*_*.pt"))
    paths += sorted(directory.glob(f"{model_id}_{int(layer)}_*_*.npz"))
    return [path for path in paths if path.is_file() and path.stat().st_size > 0]


def _rome_state(rome_dir: Path, *, model: str, selected_layer: int) -> tuple[Path, dict[str, Any]]:
    report_path = _latest(list(rome_dir.glob("rome_only_all_*.json")), description=f"ROME report below {rome_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report.get("results", {}).get(model)
    if not isinstance(result, dict) or not isinstance(result.get("summary"), dict):
        raise RuntimeError(f"ROME report has no result for {model}: {report_path}")
    summary = result["summary"]
    if int(summary.get("layer", -1)) != int(selected_layer):
        raise RuntimeError(
            f"ROME ran layer {summary.get('layer')} instead of causal layer {selected_layer}"
        )
    if int(summary.get("n_evaluated", 0)) <= 0:
        raise RuntimeError(f"ROME completed no evaluable cases: {report_path}")
    return report_path, summary


def _output_root(configured: Any) -> Path:
    raw = str(configured or "").strip()
    if not raw:
        job = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in os.environ.get("PBS_JOBID", "local")
        )
        raw = f"analysis_out/jobs/{job}-causal-rome"
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def run_pipeline(
    payload: dict[str, Any],
    *,
    trace_overrides: Sequence[str] = (),
    rome_overrides: Sequence[str] = (),
) -> Path:
    model = str(payload["model"])
    output_root = _output_root(payload["output"].get("root"))
    trace_root = output_root / "causal-trace"
    rome_dir = output_root / "rome"
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "pipeline-config.json"
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not bool(payload["resume"]["skip_causal_trace"]):
        _run_command(
            build_trace_command(
                model=model,
                output_dir=trace_root,
                num_valid_facts=int(payload["causal_trace"]["num_valid_facts"]),
                extra_overrides=trace_overrides,
            )
        )
    summary_path, trace, selected_layer = _trace_state(trace_root)

    covariance_files = find_second_moment_files(model, selected_layer)
    if not covariance_files:
        if bool(payload["resume"]["skip_second_moment"]):
            raise FileNotFoundError(
                f"No covariance for model={model} layer={selected_layer} and skip_second_moment=true"
            )
        if not bool(payload["covariance"]["compute_if_missing"]):
            raise FileNotFoundError(
                f"No covariance for model={model} layer={selected_layer} and compute_if_missing=false"
            )
        _run_command(
            build_second_moment_command(
                model=model,
                layer=selected_layer,
                target_samples=int(payload["covariance"]["target_samples"]),
            )
        )
        covariance_files = find_second_moment_files(model, selected_layer)
    if not covariance_files:
        raise FileNotFoundError(f"No saved covariance matches model={model} layer={selected_layer}")

    _run_command(
        build_rome_command(
            model=model,
            layer=selected_layer,
            n_tests=int(payload["rome"]["n_tests"]),
            start_idx=int(payload["rome"]["start_idx"]),
            output_dir=rome_dir,
            extra_overrides=[*payload["rome"].get("overrides", []), *rome_overrides],
        )
    )
    rome_report_path, rome_summary = _rome_state(
        rome_dir,
        model=model,
        selected_layer=selected_layer,
    )

    summary = {
        "schema": "latium.causal_rome_job.v1",
        "model": model,
        "selected_layer": selected_layer,
        "causal_trace_summary": str(summary_path),
        "causal_trace_plot": str(trace["plot"]),
        "causal_trace_confirmation_passed": True,
        "covariance_files": [str(path) for path in covariance_files],
        "rome_report": str(rome_report_path),
        "rome_summary": rome_summary,
        "pipeline_config": str(resolved_path),
    }
    summary_out = output_root / "pipeline-summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary_out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = _pipeline_overrides(args)
    payload = resolve_pipeline_config(overrides)
    run_pipeline(
        payload,
        trace_overrides=args.trace_override,
        rome_overrides=args.rome_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one or more causal-trace -> covariance -> ROME/structural jobs.

The driver keeps each model independent and resumable.  A model failure is
written to its state file and does not prevent later models from running.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODELS = (
    "granite4-micro",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "llama2-7b",
    "falcon-7b",
    "opt-6.7b",
    "deepseek-7b-base",
    "granite-4.1-8b",
    "ministral-3-8b",
    "gemma-4-12b",
    "olmo-3-1025-7b",
)
PAPER_ANALYSES = (
    "ccs-composite",
    "spectral",
    "gram-localization",
    "rome-profile-experiments",
    "rome-directional-experiments",
    "rome-cross-layer-experiments",
    "rome-token-alignment-experiments",
)

LOGGER = logging.getLogger("latium.paper_fleet")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def configure_logging(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = __import__("time").gmtime
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    fleet_file = logging.FileHandler(run_root / "fleet.log", encoding="utf-8")
    fleet_file.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[stream, fleet_file], force=True)


def add_model_log(model_log: Path) -> logging.FileHandler:
    handler = logging.FileHandler(model_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"))
    handler.formatter.converter = __import__("time").gmtime
    logging.getLogger().addHandler(handler)
    return handler


def run_logged(command: list[str], *, stage: str, model: str) -> None:
    rendered = " ".join(subprocess.list2cmdline([part]) for part in command)
    LOGGER.info("[%s][%s] start: %s", model, stage, rendered)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        LOGGER.info("[%s][%s] %s", model, stage, line.rstrip())
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{stage} failed for {model} with exit code {return_code}")
    LOGGER.info("[%s][%s] complete", model, stage)


def model_second_moment_files(model: str, layer: int) -> list[Path]:
    from src.common.model_config import load_model_config
    from src.common.paths import resolve_project_path

    config = load_model_config(model)
    directory = resolve_project_path(Path(str(config.second_moment_dir)))
    model_id = str(config.name).replace("/", "_")
    candidates = sorted(directory.glob(f"{model_id}_{int(layer)}_*_*.pt"))
    candidates += sorted(directory.glob(f"{model_id}_{int(layer)}_*_*.npz"))
    return [path for path in candidates if path.is_file() and path.stat().st_size > 0]


def verify_causal_trace(
    trace_root: Path,
    *,
    model: str,
    model_config: Any,
) -> dict[str, Any]:
    """Validate the held-out trace and its compatibility with the ROME config."""
    summaries = sorted(trace_root.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No causal-trace summary below {trace_root}")
    summary_path = max(summaries, key=lambda path: path.stat().st_mtime_ns)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_model = str(summary.get("model", ""))
    if summary_model not in {str(model), str(model_config.name)}:
        raise RuntimeError(
            f"Causal-trace summary is for {summary_model!r}, expected {model!r} "
            f"({model_config.name!r})"
        )
    if not bool(summary.get("confirmation_passed")):
        reason = summary.get("selection_failure_reason") or summary.get("failure_reason")
        raise RuntimeError(f"Causal tracing did not confirm a layer: {reason or summary_path}")
    selected_layer = summary.get("selected_trace_center")
    if selected_layer is None:
        raise RuntimeError(f"Causal-trace summary has no selected layer: {summary_path}")
    selected_layer = int(selected_layer)
    traced_modules = summary.get("trace_mlp_output_modules") or []
    if not traced_modules or not 0 <= selected_layer < len(traced_modules):
        raise RuntimeError(
            f"Causal-trace selected layer {selected_layer} is outside its module map"
        )
    plot = Path(str(summary.get("plot", "")))
    if not plot.is_absolute():
        plot = ROOT / plot
    if not plot.is_file() or plot.stat().st_size <= 0:
        raise FileNotFoundError(f"Causal-trace plot is missing or empty: {plot}")

    # The trace intentionally hooks the enclosing MLP while ROME edits the
    # final projection. Accept both hook locations, but reject an architecture
    # mismatch before spending time on covariance or edits.
    projection = str(model_config.layer_name_template).format(selected_layer)
    accepted = {projection}
    for marker in (".mlp.", ".shared_mlp."):
        if marker in projection:
            accepted.add(projection.split(marker, 1)[0] + marker.rstrip("."))
    restore_template = str(getattr(model_config, "restore_layer_name_template", "") or "")
    block = restore_template.format(selected_layer) if restore_template else ""
    if block:
        accepted.update(
            f"{block}.{suffix}"
            for suffix in ("mlp", "shared_mlp", "feed_forward", "ffn", "fc2")
        )
    traced_module = str(traced_modules[selected_layer])
    if traced_module not in accepted:
        raise RuntimeError(
            f"Causal-trace hook {traced_module!r} is incompatible with "
            f"model.layer_name_template at layer {selected_layer}: {projection!r}"
        )

    configured_layer = int(model_config.layer)
    return {
        "complete": True,
        "summary": str(summary_path),
        "plot": str(plot),
        "selected_layer": selected_layer,
        "configured_layer": configured_layer,
        "layer_matches_config": selected_layer == configured_layer,
        "trace_module": traced_module,
        "rome_projection_module": projection,
        "num_layers": len(traced_modules),
        "num_valid_facts": int(summary.get("num_valid_facts", 0)),
        "confirmation_passed": True,
    }


def manifest_records(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing structural manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for record in manifest.get("artifacts", {}).values() if isinstance(record, dict)]
    return manifest, records


def record_payload(run_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = run_root / str(record["path"])
    return json.loads(path.read_text(encoding="utf-8"))


def verify_structural_run(run_root: Path, *, model: str, selected_layer: int) -> dict[str, Any]:
    manifest, records = manifest_records(run_root)
    executions = [record for record in records if record.get("kind") == "execution"]
    baseline_execution = [record for record in executions if record.get("edit_method") in (None, "baseline")]
    method_executions = [record for record in executions if record.get("edit_method") not in (None, "baseline")]
    if len(baseline_execution) != 1:
        raise RuntimeError(f"Expected exactly one unedited execution, found {len(baseline_execution)}")
    if not method_executions:
        raise RuntimeError("No edited execution artifact was written")

    baseline_record = baseline_execution[0]
    baseline_payload = record_payload(run_root, baseline_record)
    baseline_cases = baseline_payload.get("cases", [])
    if not any(case.get("case_id") == "baseline" and case.get("status") == "complete" for case in baseline_cases):
        raise RuntimeError("The unedited execution was not completed")

    baseline_time = str(baseline_record.get("created_at", ""))
    edited_times = [str(record.get("created_at", "")) for record in method_executions]
    baseline_first = not baseline_time or all(not timestamp or baseline_time <= timestamp for timestamp in edited_times)
    if not baseline_first:
        raise RuntimeError("The unedited execution was not created before edited executions")

    baseline_analyses: dict[str, dict[str, Any]] = {}
    edited_analyses: dict[str, int] = {}
    for record in records:
        if record.get("kind") != "analysis":
            continue
        producer = str(record.get("producer"))
        edit_method = record.get("edit_method")
        if producer not in PAPER_ANALYSES:
            continue
        payload = record_payload(run_root, record)
        if edit_method in (None, "baseline"):
            baseline_analyses[producer] = {
                "status": record.get("status"),
                "cases": payload.get("cases", []),
                "error": record.get("error"),
            }
        else:
            edited_analyses[producer] = edited_analyses.get(producer, 0) + 1

    missing_baseline = [name for name in PAPER_ANALYSES if name not in baseline_analyses]
    if missing_baseline:
        raise RuntimeError(f"Missing unedited analyses: {', '.join(missing_baseline)}")
    for name, result in baseline_analyses.items():
        if result["status"] == "error":
            raise RuntimeError(f"Unedited analysis failed for {name}: {result['error']}")
        if not any(case.get("case_id") == "baseline" for case in result["cases"]):
            raise RuntimeError(f"Unedited analysis has no baseline case: {name}")

    missing_edited = [name for name in PAPER_ANALYSES if edited_analyses.get(name, 0) == 0]
    if missing_edited:
        raise RuntimeError(f"Missing edited analyses: {', '.join(missing_edited)}")

    method_layers = {
        str(record.get("edit_method")): record_payload(run_root, record).get("summary", {}).get("target_layer")
        for record in method_executions
    }
    if any(layer is not None and int(layer) != int(selected_layer) for layer in method_layers.values()):
        raise RuntimeError(f"Edited structural layer does not match configured layer {selected_layer}: {method_layers}")
    return {
        "run_id": manifest.get("run_id"),
        "baseline_first": baseline_first,
        "baseline_analyses": sorted(baseline_analyses),
        "edited_analyses": edited_analyses,
        "method_layers": method_layers,
    }


def verify_graphs(run_root: Path) -> dict[str, Any]:
    required = {
        "ccs-report": (
            run_root / "graphs" / "paper",
            run_root / "graphs" / "detector",
            run_root / "graphs" / "rome-success",
            run_root / "graphs" / "detector-window",
            run_root / "graphs" / "structural-ccs-lines",
        ),
        "full": (
            run_root / "graphs" / "structural-artifact-grid",
            run_root / "graphs" / "rome-relative-profile-grid",
            run_root / "graphs" / "detector-signals",
            run_root / "graphs" / "run-summary",
        ),
    }
    result: dict[str, Any] = {}
    for group, directories in required.items():
        missing = [str(directory.relative_to(run_root)) for directory in directories if not directory.is_dir()]
        empty = [
            str(directory.relative_to(run_root))
            for directory in directories
            if directory.is_dir() and not any(path.is_file() and path.stat().st_size > 0 for path in directory.rglob("*"))
        ]
        if missing or empty:
            raise RuntimeError(f"{group} graph validation failed; missing={missing}, empty={empty}")
        result[group] = {"directories": [str(directory.relative_to(run_root)) for directory in directories]}
    ccs_json = list((run_root / "graphs" / "structural-ccs-lines").glob("*.json"))
    if not ccs_json:
        raise RuntimeError("CCS graph JSON output is missing")
    result["ccs_json"] = [str(path.relative_to(run_root)) for path in ccs_json]
    return result


def log_graph_artifact(run_root: Path, *, model: str, project: str, group: str, run_name: str) -> str:
    import wandb

    graph_run = wandb.init(
        project=project,
        group=group,
        name=f"{run_name}-graphs",
        job_type="paper-fleet-graphs",
        config={"model": model, "run_root": str(run_root)},
    )
    if graph_run is None:
        raise RuntimeError("wandb.init returned no graph run")
    try:
        artifact = wandb.Artifact(f"{slug(model)}-{run_name}-graphs", type="latium-graphs")
        artifact.add_dir(str(run_root / "graphs"))
        graph_run.log_artifact(artifact)
        graph_run.log({"graphs/files": len(list((run_root / "graphs").rglob("*")))})
        return str(graph_run.url or graph_run.id)
    finally:
        graph_run.finish()


class FleetRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = Path(args.run_root).resolve()
        # Do not resolve the venv executable: on this host it is a symlink and
        # resolving it changes the interpreter back to system Python.
        self.python = str(Path(args.python).expanduser())
        self.group = args.wandb_group or f"paper-fleet-n{args.n_tests}-{self.run_root.name}"

    def run_model(self, model: str) -> dict[str, Any]:
        model_dir = self.run_root / slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        state_path = model_dir / "state.json"
        state: dict[str, Any] = {
            "model": model,
            "started_at": utc_now(),
            "status": "running",
            "stages": {},
        }
        if self.args.resume and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "running"
        write_json(state_path, state)
        model_handler = add_model_log(model_dir / "model.log")
        try:
            from src.common.model_config import load_model_config

            model_config = load_model_config(model)
            trace_root = model_dir / "causal-trace"
            trace_stage = state["stages"].get("causal_trace", {})
            trace_summary_exists = any(trace_root.glob("*/summary.json"))
            trace_resume_valid = bool(
                trace_stage.get("complete")
                and trace_stage.get("trace_module")
                and trace_summary_exists
            )
            if not trace_resume_valid:
                trace_command = [
                    self.python,
                    "-m",
                    "src",
                    "causal-trace",
                    f"model={model}",
                    f"command.causal_trace.output_dir={trace_root}",
                    f"command.causal_trace.num_valid_facts={self.args.trace_facts}",
                    f"command.causal_trace.minimum_confirmation_facts={self.args.minimum_confirmation_facts}",
                    f"command.causal_trace.bootstrap_samples={self.args.trace_bootstrap_samples}",
                    "command.causal_trace.overwrite_model_config_layer=false",
                ]
                run_logged(trace_command, stage="causal-trace", model=model)
            trace_check = verify_causal_trace(
                trace_root,
                model=model,
                model_config=model_config,
            )
            selected_layer = int(trace_check["selected_layer"])
            state["stages"]["causal_trace"] = trace_check
            write_json(state_path, state)
            LOGGER.info(
                "[%s] causal trace selected layer=%d (configured layer=%d)",
                model,
                selected_layer,
                int(trace_check["configured_layer"]),
            )

            covariance_files = model_second_moment_files(model, selected_layer)
            covariance_command = [
                self.python,
                "-m",
                "src",
                "second-moment",
                f"model={model}",
                f"model.layer={selected_layer}",
                "model.second_moment_path=null",
                f"model.second_moment_target_samples={self.args.covariance_samples}",
            ]
            if not covariance_files:
                run_logged(covariance_command, stage="covariance", model=model)
                covariance_files = model_second_moment_files(model, selected_layer)
            if not covariance_files:
                raise FileNotFoundError(f"No covariance file for {model} layer {selected_layer}")
            state["stages"]["covariance"] = {
                "complete": True,
                "files": [str(path) for path in covariance_files],
                "samples": self.args.covariance_samples,
                "layer": selected_layer,
            }
            write_json(state_path, state)

            structural_output = model_dir / "structural-output"
            structural_run_id = f"{slug(model)}-n{self.args.n_tests}"
            structural_root = structural_output / structural_run_id
            layer_override = "{" + model + ":" + str(selected_layer) + "}"
            structural_command = [
                self.python,
                "-m",
                "src",
                "structural",
                "run",
                f"structural.run.models=[{model}]",
                f"+structural.run.model_layer_overrides={layer_override}",
                f"structural.run.n_tests={self.args.n_tests}",
                "structural.run.start_idx=0",
                f"structural.run.output_dir={structural_output}",
                f"structural.run.run_id={structural_run_id}",
                "structural.analysis.preset=paper",
                "structural.analysis.enable=[gram-localization,rome-profile-experiments,rome-directional-experiments,rome-cross-layer-experiments,rome-token-alignment-experiments]",
                "structural.analysis.continue_on_error=false",
                "structural.run.edit_methods=[rome]",
                "structural.run.fail_on_missing_second_moment=true",
                "structural.render.enabled=true",
                "structural.render.renderer_preset=full",
                "structural.render.renderers.structural-artifact-grid.formats=[png,json]",
                "structural.render.renderers.rome-relative-profile-grid.formats=[png,json]",
                "structural.render.renderers.rome-relative-profile-grid.case_pages=none",
                "structural.render.renderers.structural-ccs-lines.formats=[png,json]",
                "structural.render.continue_on_error=false",
                "structural.tracking.provider=wandb",
                f"structural.tracking.project={self.args.wandb_project}",
                f"structural.tracking.group={self.group}",
                f"structural.tracking.run_name={structural_run_id}",
                "structural.tracking.mode=online",
                f"structural.run.progress_file={model_dir / 'progress.json'}",
            ]
            if not state["stages"].get("structural", {}).get("complete"):
                run_logged(structural_command, stage="structural", model=model)
            structural_check = verify_structural_run(structural_root, model=model, selected_layer=selected_layer)
            state["stages"]["structural"] = {"complete": True, "run_root": str(structural_root), **structural_check}
            write_json(state_path, state)

            graph_command = [
                self.python,
                "-m",
                "src",
                "graphs",
                "run",
                str(structural_root),
                "graphs.renderer_preset=ccs-report",
                "graphs.renderers.structural-ccs-lines.formats=[png,json]",
            ]
            if not state["stages"].get("graphs", {}).get("complete"):
                run_logged(graph_command, stage="graphs-ccs-report", model=model)
                run_logged(
                    [
                        self.python,
                        "-m",
                        "src",
                        "graphs",
                        "run",
                        str(structural_root),
                        "graphs.renderer_preset=full",
                        "graphs.renderers.structural-artifact-grid.formats=[png,json]",
                        "graphs.renderers.rome-relative-profile-grid.formats=[png,json]",
                        "graphs.renderers.rome-relative-profile-grid.case_pages=none",
                        "graphs.renderers.structural-ccs-lines.formats=[png,json]",
                    ],
                    stage="graphs-full",
                    model=model,
                )
            graph_check = verify_graphs(structural_root)
            graph_url = log_graph_artifact(
                structural_root,
                model=model,
                project=self.args.wandb_project,
                group=self.group,
                run_name=structural_run_id,
            )
            state["stages"]["graphs"] = {
                "complete": True,
                "run_root": str(structural_root),
                "validation": graph_check,
                "wandb_graph_run": graph_url,
            }
            state["status"] = "complete"
            state["completed_at"] = utc_now()
            write_json(state_path, state)
            LOGGER.info("[%s] fleet model complete", model)
            return state
        except Exception as exc:
            state["status"] = "failed"
            state["failed_at"] = utc_now()
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["traceback"] = traceback.format_exc()
            write_json(state_path, state)
            LOGGER.error("[%s] model failed: %s", model, exc)
            return state
        finally:
            logging.getLogger().removeHandler(model_handler)
            model_handler.close()

    def run(self) -> int:
        configure_logging(self.run_root)
        fleet_state = {
            "started_at": utc_now(),
            "run_root": str(self.run_root),
            "models": list(self.args.models),
            "n_tests": self.args.n_tests,
            "trace_facts": self.args.trace_facts,
            "minimum_confirmation_facts": self.args.minimum_confirmation_facts,
            "trace_bootstrap_samples": self.args.trace_bootstrap_samples,
            "covariance_samples": self.args.covariance_samples,
            "wandb_project": self.args.wandb_project,
            "wandb_group": self.group,
            "status": "running",
        }
        if not self.args.worker:
            write_json(self.run_root / "fleet.json", fleet_state)
        results = []
        for model in self.args.models:
            results.append(self.run_model(model))
        failures = [result for result in results if result.get("status") != "complete"]
        fleet_state.update(
            {
                "status": "failed" if failures else "complete",
                "completed_at": utc_now(),
                "failed_models": [result.get("model") for result in failures],
                "completed_models": [result.get("model") for result in results if result.get("status") == "complete"],
            }
        )
        if self.args.worker:
            write_json(self.run_root / slug(self.args.models[0]) / "worker-summary.json", fleet_state)
        else:
            write_json(self.run_root / "fleet.json", fleet_state)
        LOGGER.info("Fleet complete: %d succeeded, %d failed", len(results) - len(failures), len(failures))
        return 1 if failures else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(ROOT / "analysis_out" / "paper-fleet" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-tests", type=int, default=50)
    parser.add_argument("--trace-facts", type=int, default=50)
    parser.add_argument("--minimum-confirmation-facts", type=int, default=25)
    parser.add_argument("--trace-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--covariance-samples", type=int, default=100000)
    parser.add_argument("--wandb-project", default="latium")
    parser.add_argument("--wandb-group")
    parser.add_argument("--worker", action="store_true", help="Do not write a shared fleet.json (for parallel PBS workers)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if (
        args.n_tests <= 0
        or args.trace_facts <= 0
        or args.minimum_confirmation_facts < 2
        or args.minimum_confirmation_facts >= args.trace_facts
        or args.trace_bootstrap_samples <= 0
        or args.covariance_samples <= 0
    ):
        parser.error("test, trace, confirmation, bootstrap, and covariance counts are invalid")
    if args.worker and len(args.models) != 1:
        parser.error("--worker accepts exactly one model")
    return args


if __name__ == "__main__":
    raise SystemExit(FleetRunner(parse_args()).run())

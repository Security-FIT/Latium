#!/usr/bin/env python3
"""Run one or more configured-layer covariance -> ROME/structural jobs.

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
    "deepseek-7b-base",
    "falcon-7b",
    "gemma-4-12b",
    "gpt2-xl",
    "granite-4.1-8b",
    "granite4-micro",
    "llama2-7b",
    "ministral-3-8b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "olmo-3-1025-7b",
    "opt-6.7b",
    "qwen3-8b",
)
PAPER_ANALYSES = (
    "ccs-composite",
    "spectral",
    "gram-localization",
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


def model_second_moment_files(model: str, layer: int, samples: int, *, config=None) -> list[Path]:
    """Return only the configured, non-empty covariance for this exact run."""

    from src.common.model_config import load_model_config
    from src.common.paths import resolve_project_path

    config = load_model_config(model) if config is None else config
    directory = resolve_project_path(Path(str(config.second_moment_dir)))
    model_id = str(getattr(config, "second_moment_model_name", config.name)).replace("/", "_")
    stem = f"{model_id}_{int(layer)}_*_{int(samples)}"
    candidates = sorted(directory.glob(f"{stem}.pt"))
    candidates += sorted(directory.glob(f"{stem}.npz"))
    available = [
        path.resolve()
        for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]

    configured = getattr(config, "second_moment_path", None)
    if not configured:
        return available
    configured_path = resolve_project_path(Path(str(configured))).resolve()
    return [configured_path] if configured_path in available else []


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


def ccs_supported_model(model: str) -> bool:
    from src.structural.analysis.registry import ANALYSES, supports_model

    return supports_model(ANALYSES.get("ccs-composite"), model)


def verify_graphs(run_root: Path, *, require_ccs: bool) -> dict[str, Any]:
    required = {
        "full": (
            run_root / "graphs" / "structural-artifact-grid",
            run_root / "graphs" / "detector-signals",
            run_root / "graphs" / "run-summary",
        ),
    }
    if require_ccs:
        required["ccs-report"] = (
            run_root / "graphs" / "paper",
            run_root / "graphs" / "detector",
            run_root / "graphs" / "rome-success",
            run_root / "graphs" / "detector-window",
            run_root / "graphs" / "structural-ccs-lines",
        )
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
    if require_ccs:
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
            selected_layer = int(model_config.layer)
            state["stages"]["layer_selection"] = {
                "complete": True, "selection_method": "configured_model_layer", "selected_layer": selected_layer,
            }
            write_json(state_path, state)

            covariance_files = model_second_moment_files(
                model,
                selected_layer,
                self.args.covariance_samples,
            )
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
                if self.args.skip_second_moment:
                    raise FileNotFoundError(
                        f"No covariance file for {model} layer {selected_layer}; "
                        "--skip-second-moment forbids recomputation"
                    )
                run_logged(covariance_command, stage="covariance", model=model)
                covariance_files = model_second_moment_files(
                model,
                selected_layer,
                self.args.covariance_samples,
            )
            if not covariance_files:
                raise FileNotFoundError(f"No covariance file for {model} layer {selected_layer}")
            state["stages"]["covariance"] = {
                "complete": True,
                "files": [str(path) for path in covariance_files],
                "samples": self.args.covariance_samples,
                "layer": selected_layer,
            }
            write_json(state_path, state)

            if self.args.covariance_only:
                state["status"] = "complete"
                state["completed_at"] = utc_now()
                write_json(state_path, state)
                LOGGER.info("[%s] covariance-only worker complete", model)
                return state

            structural_output = model_dir / "structural-output"
            structural_run_id = f"{slug(model)}-n{self.args.n_tests}"
            structural_root = structural_output / structural_run_id
            structural_command = [
                self.python,
                "-m",
                "src",
                "structural",
                "run",
                f"structural.run.models=[{model}]",
                f"structural.run.n_tests={self.args.n_tests}",
                "structural.run.start_idx=0",
                f"structural.run.output_dir={structural_output}",
                f"structural.run.run_id={structural_run_id}",
                "structural.analysis.preset=paper",
                "structural.analysis.enable=[gram-localization]",
                "structural.analysis.continue_on_error=false",
                "structural.run.edit_methods=[rome]",
                "structural.run.fail_on_missing_second_moment=true",
                "structural.render.enabled=true",
                "structural.render.renderer_preset=full",
                "++structural.render.renderers.structural-artifact-grid.formats=[png,json]",
                "++structural.render.renderers.structural-ccs-lines.formats=[png,json]",
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

            require_ccs = ccs_supported_model(model)
            if not state["stages"].get("graphs", {}).get("complete"):
                if require_ccs:
                    run_logged(
                        [
                            self.python,
                            "-m",
                            "src",
                            "graphs",
                            "run",
                            str(structural_root),
                            "graphs.renderer_preset=ccs-report",
                            "++graphs.renderers.structural-ccs-lines.formats=[png,json]",
                        ],
                        stage="graphs-ccs-report",
                        model=model,
                    )
                run_logged(
                    [
                        self.python,
                        "-m",
                        "src",
                        "graphs",
                        "run",
                        str(structural_root),
                        "graphs.renderer_preset=full",
                        "++graphs.renderers.structural-artifact-grid.formats=[png,json]",
                        "++graphs.renderers.structural-ccs-lines.formats=[png,json]",
                    ],
                    stage="graphs-full",
                    model=model,
                )
            graph_check = verify_graphs(structural_root, require_ccs=require_ccs)
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
        if self.args.workflow == "gram":
            from jobs.gram_fleet import run
            return run(self.args)
        configure_logging(self.run_root)
        fleet_state = {
            "started_at": utc_now(),
            "run_root": str(self.run_root),
            "models": list(self.args.models),
            "n_tests": self.args.n_tests,
            "covariance_samples": self.args.covariance_samples,
            "covariance_only": self.args.covariance_only,
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
    # run.pbs selects the configured MetaCentrum environment on PATH. Reuse
    # the interpreter that launched this driver so child stages do not fall
    # back to a repository-local .venv that may not exist.
    parser.add_argument("--python", default=os.environ.get("LATIUM_PYTHON", sys.executable))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-tests", type=int, default=50)
    parser.add_argument("--workflow", choices=("paper", "gram"), default="paper")
    parser.add_argument("--case-index-file")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-stop", type=int)
    parser.add_argument("--tracking", choices=("none", "wandb"), default="none")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--covariance-samples", type=int, default=100000)
    parser.add_argument(
        "--covariance-only",
        action="store_true",
        help="Compute or verify the exact covariance, then stop before structural analysis",
    )
    parser.add_argument("--wandb-project", default="latium")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--skip-second-moment",
        "--reuse-covariance",
        dest="skip_second_moment",
        action="store_true",
        help="Require an existing covariance file instead of computing one",
    )
    parser.add_argument("--worker", action="store_true", help="Do not write a shared fleet.json (for parallel PBS workers)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.workflow == "gram":
        if not args.case_index_file:
            parser.error("--workflow gram requires --case-index-file")
        args.case_index_file = str(Path(args.case_index_file).resolve())
        from src.counterfact_selection import load_case_manifest
        cohort = load_case_manifest(args.case_index_file)
        args.case_stop = args.case_stop if args.case_stop is not None else args.case_start + args.n_tests
        if args.case_start < 0 or args.case_stop <= args.case_start or args.case_stop > cohort["count"]:
            parser.error("invalid manifest range: use zero-based positions with an exclusive stop")
        args.n_tests = args.case_stop - args.case_start
    elif args.case_index_file or args.case_start or args.case_stop is not None or args.prepare_only:
        parser.error("manifest ranges and --prepare-only require --workflow gram")
    if (
        args.n_tests <= 0
        or args.covariance_samples <= 0
    ):
        parser.error("test and covariance counts are invalid")
    if args.worker and len(args.models) != 1:
        parser.error("--worker accepts exactly one model")
    return args


if __name__ == "__main__":
    raise SystemExit(FleetRunner(parse_args()).run())

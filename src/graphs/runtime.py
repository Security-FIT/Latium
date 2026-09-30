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
from typing import Any, Mapping, Sequence

from src.graphs.context import RenderContext, RenderExecutionError, RendererUnavailableError
from src.graphs.registry import RENDERERS, resolve_renderers
from src.results import ArtifactWriter, RunArtifactReader, build_artifact, config_hash
from src.results.ids import render_id


REPORT_OUTPUT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "paper": (".json",),
    "detector": (".png",),
    "rome-success": (".png",),
    "detector-window": (".png",),
    "structural-ccs-lines": (".png", ".pdf", ".json"),
}


def render_run(
    run_root: str | Path,
    *,
    preset: str = "ccs-report",
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
    style_preset: str = "default",
    renderer_options: Mapping[str, Mapping[str, Any]] | None = None,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    root = Path(run_root)
    reader = RunArtifactReader(root)
    writer = ArtifactWriter(root)
    renderer_ids = resolve_renderers(
        preset,
        enabled=enabled,
        disabled=disabled,
    )
    all_analysis_records = list(reader.records(kind="analysis"))
    all_execution_records = list(reader.records(kind="execution"))
    all_capture_records = list(reader.records(kind="capture"))
    written: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []
    report_failures: list[str] = []
    configured_options = dict(renderer_options or {})
    complete_report = preset == "ccs-report"

    for renderer_id in renderer_ids:
        spec = RENDERERS.get(renderer_id)
        artifact_id = render_id(renderer_id)
        options = configured_options.get(renderer_id, {})
        selected_options = {key: options[key] for key in spec.option_keys if key in options}
        config = {
            "renderer": renderer_id,
            "style_preset": style_preset,
            "options": selected_options,
        }
        input_records, missing, warnings = _renderer_input_records(
            spec,
            executions=all_execution_records,
            captures=all_capture_records,
            analyses=all_analysis_records,
        )
        inputs = [reader.ref(str(record["artifact_id"])) for record in input_records]
        digest = config_hash(config)
        current = writer.current(
            artifact_id,
            expected_config_hash=digest,
            inputs=inputs,
        )
        current_complete = (
            complete_report
            and current is not None
            and current.get("status") == "complete"
            and _render_outputs_exist(root, current, selected_options)
        )
        if not force and current is not None and (not complete_report or current_complete):
            skipped.append(artifact_id)
            continue
        output_dir = root / "graphs" / renderer_id
        if missing:
            outputs = []
            status = "unavailable"
            error = f"missing renderer inputs: {', '.join(missing)}"
        else:
            try:
                context = _make_context(
                    reader,
                    root=root,
                    output_dir=output_dir,
                    input_records=input_records,
                    options=selected_options,
                    style_preset=style_preset,
                    warnings=tuple(warnings),
                )
                outputs = spec.load()(context)
                status = "complete" if outputs else "unavailable"
                error = None if outputs else "renderer produced no outputs"
                if complete_report and outputs and not _report_outputs_valid(renderer_id, outputs, selected_options):
                    status = "error"
                    error = "required report files were not produced"
            except RendererUnavailableError as exc:
                outputs = []
                status = "unavailable"
                error = str(exc)
            except Exception as exc:
                outputs = []
                status = "error"
                error = str(exc)
                failures.append(f"{renderer_id}: {exc}")
        if complete_report and status != "complete":
            report_failures.append(f"{renderer_id}: {error}")
        payload = build_artifact(
            artifact_id=artifact_id,
            kind="render",
            producer=renderer_id,
            run_id=str(reader.manifest["run_id"]),
            model=None,
            plan_id=None,
            edit_method=None,
            status=status,
            config=config,
            config_hash=digest,
            inputs=inputs,
            created_at=datetime.now().isoformat(),
            cases=[],
            summary={
                "outputs": [_relative_output_path(root, output) for output in outputs],
            },
            error=error,
        )
        writer.write(output_dir / "artifact.json", payload, force=force or (complete_report and current is not None))
        written.append(artifact_id)

    if report_failures:
        raise RenderExecutionError("ccs-report incomplete: " + "; ".join(report_failures))
    if failures and not continue_on_error:
        raise RenderExecutionError("renderer failures: " + "; ".join(failures))

    return {
        "run_id": reader.manifest["run_id"],
        "written": written,
        "skipped": skipped,
        "errors": failures,
    }


def _renderer_input_records(
    spec: Any,
    *,
    executions: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    selected: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    warnings: list[str] = []

    for required, kind, records in (
        (spec.requires_execution, "execution", executions),
        (spec.requires_analyses, "analysis", analyses),
    ):
        if required:
            if not records:
                missing.append(kind)
            selected.update((str(record["artifact_id"]), record) for record in records)

    for kind, records, required, optional in (
        ("capture", captures, spec.required_captures, spec.optional_captures),
        ("analysis", analyses, spec.required_analyses, spec.optional_analyses),
    ):
        for producer in (*required, *optional):
            found = [record for record in records if record.get("producer") == producer]
            selected.update((str(record["artifact_id"]), record) for record in found)
            if not found:
                if producer in required:
                    missing.append(f"{kind}:{producer}")
                else:
                    warnings.append(f"optional {kind} unavailable: {producer}")

    return list(selected.values()), missing, warnings


def _make_context(
    reader: RunArtifactReader,
    *,
    root: Path,
    output_dir: Path,
    input_records: list[dict[str, Any]],
    options: Mapping[str, Any],
    style_preset: str,
    warnings: tuple[str, ...],
) -> RenderContext:
    executions: list[dict[str, Any]] = []
    captures: dict[str, list[dict[str, Any]]] = {}
    analyses: dict[str, list[dict[str, Any]]] = {}
    for record in input_records:
        payload = reader.load(str(record["artifact_id"]))
        kind = record.get("kind")
        producer = str(record.get("producer"))
        if kind == "execution":
            executions.append(payload)
        elif kind == "capture":
            captures.setdefault(producer, []).append(payload)
        elif kind == "analysis":
            analyses.setdefault(producer, []).append(payload)
    return RenderContext(
        run_root=root,
        output_dir=output_dir,
        manifest=reader.manifest,
        executions=tuple(executions),
        captures={key: tuple(value) for key, value in captures.items()},
        analyses={key: tuple(value) for key, value in analyses.items()},
        options=dict(options),
        style_preset=style_preset,
        warnings=warnings,
    )


def _relative_output_path(root: Path, output: str) -> str:
    path = Path(output)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_outputs_exist(
    root: Path,
    record: Mapping[str, Any],
    options: Mapping[str, Any],
) -> bool:
    try:
        artifact_path = (root / str(record["path"])).resolve()
        if not artifact_path.is_relative_to(root.resolve()):
            return False
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        outputs = artifact["summary"]["outputs"]
        if not outputs:
            return False
        for output in outputs:
            path = (root / str(output)).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                return False
        return _report_outputs_valid(
            str(record["producer"]),
            [(root / str(output)) for output in outputs],
            options,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _report_outputs_valid(
    renderer_id: str,
    outputs: Sequence[str | Path],
    options: Mapping[str, Any],
) -> bool:
    paths = [Path(output) for output in outputs]
    required_suffixes = REPORT_OUTPUT_SUFFIXES.get(renderer_id, ())
    if renderer_id == "structural-ccs-lines" and "formats" in options:
        formats = options["formats"]
        if isinstance(formats, str):
            formats = (part.strip() for part in formats.split(","))
        required_suffixes = tuple(f".{str(fmt)}" for fmt in formats)
    return (
        bool(paths)
        and all(path.is_file() for path in paths)
        and all(any(path.suffix == suffix for path in paths) for suffix in required_suffixes)
    )

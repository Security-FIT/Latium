"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.graphs.context import RenderContext, RenderExecutionError, RendererUnavailableError
from src.graphs.registry import RENDERERS, resolve_renderers
from src.results import ArtifactWriter, RunArtifactReader, build_artifact, config_hash
from src.results.ids import render_id


def render_run(
    run_root: str | Path,
    *,
    preset: str = "paper",
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
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
    configured_options = dict(renderer_options or {})

    for renderer_id in renderer_ids:
        spec = RENDERERS.get(renderer_id)
        artifact_id = render_id(renderer_id)
        selected_options = {
            key: dict(configured_options.get(renderer_id, {})).get(key)
            for key in spec.option_keys
            if key in dict(configured_options.get(renderer_id, {}))
        }
        config = {
            "renderer": renderer_id,
            "schema_version": spec.schema_version,
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
        if not force and current is not None:
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
                    warnings=tuple(warnings),
                )
                outputs = spec.load()(context)
                status = "complete" if outputs else "unavailable"
                error = None if outputs else "renderer produced no outputs"
            except RendererUnavailableError as exc:
                outputs = []
                status = "unavailable"
                error = str(exc)
            except Exception as exc:
                outputs = []
                status = "error"
                error = str(exc)
                failures.append(f"{renderer_id}: {exc}")
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
        writer.write(output_dir / "artifact.json", payload, force=force)
        written.append(artifact_id)

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
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    warnings: list[str] = []
    if spec.requires_execution:
        if executions:
            selected.extend(executions)
        else:
            missing.append("execution")
    if spec.requires_analyses:
        if analyses:
            selected.extend(analyses)
        else:
            missing.append("analysis")

    for producer in spec.required_captures:
        found = [record for record in captures if record.get("producer") == producer]
        if found:
            selected.extend(found)
        else:
            missing.append(f"capture:{producer}")
    for producer in spec.optional_captures:
        found = [record for record in captures if record.get("producer") == producer]
        if found:
            selected.extend(found)
        else:
            warnings.append(f"optional capture unavailable: {producer}")

    for producer in spec.required_analyses:
        found = [record for record in analyses if record.get("producer") == producer]
        if found:
            selected.extend(found)
        else:
            missing.append(f"analysis:{producer}")
    for producer in spec.optional_analyses:
        found = [record for record in analyses if record.get("producer") == producer]
        if found:
            selected.extend(found)
        else:
            warnings.append(f"optional analysis unavailable: {producer}")

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in selected:
        artifact_id = str(record["artifact_id"])
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        deduped.append(record)
    return deduped, missing, warnings


def _make_context(
    reader: RunArtifactReader,
    *,
    root: Path,
    output_dir: Path,
    input_records: list[dict[str, Any]],
    options: Mapping[str, Any],
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
        warnings=warnings,
    )


def _relative_output_path(root: Path, output: str) -> str:
    path = Path(output)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

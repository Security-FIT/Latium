"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.graphs.registry import RENDERERS, resolve_renderers
from src.results import ArtifactWriter, RunArtifactReader, build_artifact, config_hash
from src.results.ids import render_id


def render_run(
    run_root: str | Path,
    *,
    preset: str = "paper",
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
    force: bool = False,
) -> dict[str, Any]:
    root = Path(run_root)
    reader = RunArtifactReader(root)
    writer = ArtifactWriter(root)
    renderer_ids = resolve_renderers(
        preset,
        enabled=enabled,
        disabled=disabled,
    )
    analysis_records = list(reader.records(kind="analysis"))
    analyses = [reader.load(str(record["artifact_id"])) for record in analysis_records]
    inputs = [reader.ref(str(record["artifact_id"])) for record in analysis_records]
    written: list[str] = []
    skipped: list[str] = []

    for renderer_id in renderer_ids:
        spec = RENDERERS.get(renderer_id)
        artifact_id = render_id(renderer_id)
        config = {"renderer": renderer_id}
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
        try:
            outputs = spec.load()(
                {
                    "run_root": root,
                    "output_dir": output_dir,
                    "analyses": analyses,
                }
            )
            status = "complete" if outputs else "unavailable"
            error = None if outputs else "renderer produced no outputs"
        except Exception as exc:
            outputs = []
            status = "error"
            error = str(exc)
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

    return {
        "run_id": reader.manifest["run_id"],
        "written": written,
        "skipped": skipped,
    }


def _relative_output_path(root: Path, output: str) -> str:
    path = Path(output)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

#!/usr/bin/env python3
"""Render bundle-style graphs for one pipeline run.

This is a thin bridge from `pipeline_out/<run>` to the existing bundle graph
renderers. It creates the minimal `data/final_n500_models_only/<model>/...`
layout expected by the renderers, then calls the renderer entry points in this
directory. The produced graphs are written under `<run>/graphs/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root  # noqa: E402


IMPORT_ROOT = add_import_root(__file__)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, run_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    repo_candidate = (IMPORT_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (run_root / path).resolve()


def selected_models(summary_payload: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    requested_set = set(requested or [])
    models = []
    for item in summary_payload.get("models", []):
        model = str(item.get("model") or "").strip()
        if not model:
            continue
        if requested_set and model not in requested_set:
            continue
        models.append(model)
    return models


def summary_by_model(summary_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in summary_payload.get("models", []):
        model = str(item.get("model") or "").strip()
        if model:
            rows[model] = item
    return rows


def copy_matching(src_dir: Path, dest_dir: Path, patterns: Sequence[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, dest_dir / path.name)


def prepare_renderer_bundle(run_root: Path, models: Sequence[str]) -> None:
    data_root = run_root / "data"
    final_root = data_root / "final_n500_models_only"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    # These empty roots keep bundle-wide renderers on their normal code paths
    # while limiting this per-run invocation to the copied final-model run data.
    (data_root / "prefixtest_n500_runs").mkdir(parents=True, exist_ok=True)
    (data_root / "fleet_single_edit_runs" / "fleet_out_remote").mkdir(parents=True, exist_ok=True)

    structural_root = run_root / "structural"
    detector_root = run_root / "detector"
    summary_payload = load_json(run_root / "run_summary_latest.json")
    rows = summary_by_model(summary_payload)

    for model in models:
        row = rows.get(model, {})
        model_root = final_root / model
        model_structural = model_root / "structural"
        model_detector = model_root / "detector"

        copy_matching(
            structural_root,
            model_structural,
            [
                f"rome_structural_{model}_*.json",
                f"baseline_structural_{model}_*.json",
            ],
        )

        detector_summary = str(row.get("detector_summary") or "").strip()
        if detector_summary:
            detector_path = resolve_path(detector_summary, run_root)
            if detector_path.exists():
                model_detector.mkdir(parents=True, exist_ok=True)
                shutil.copy2(detector_path, model_detector / f"{model}_detector_summary.json")

        success_path = run_root / "success.txt"
        if success_path.exists():
            shutil.copy2(success_path, model_root / "success.txt")


def run_renderer(script_name: str, bundle_root: Path, extra_args: Sequence[str] = ()) -> None:
    script_path = SCRIPT_DIR / script_name
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{IMPORT_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(IMPORT_ROOT)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    cmd = [sys.executable, str(script_path), "--bundle-root", str(bundle_root), *extra_args]
    print("[run-graphs][renderer]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def write_readme(run_root: Path, models: Sequence[str], window_radius: int) -> None:
    graph_root = run_root / "graphs"
    lines = [
        "# Pipeline Bundle Graphs",
        "",
        "These graphs were rendered by the existing `scripts/bundle_graphs` entry points against a mini bundle prepared from this pipeline run.",
        "",
        f"Run root: `{run_root}`",
        f"Models: `{', '.join(models)}`",
        "",
        "## Renderers",
        "",
        "- `scripts/bundle_graphs/render_bundle_rome_success_metrics.py`",
        "- `scripts/bundle_graphs/render_bundle_detector_stacked_variants.py`",
        f"- `scripts/bundle_graphs/render_final_n500_detector_layer_window.py --window-radius {window_radius}`",
        "- `scripts/bundle_graphs/render_artifacts/render_artifacts_grid.py --allow-missing-models`",
        "",
    ]
    graph_root.mkdir(parents=True, exist_ok=True)
    (graph_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="A pipeline_out/<run> directory after local posthoc has run.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model keys to render from the run summary.")
    parser.add_argument("--window-radius", type=int, default=1)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    summary_path = run_root / "run_summary_latest.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing post-hoc summary: {summary_path}")
    if args.window_radius < 0:
        raise ValueError("--window-radius must be >= 0")

    summary_payload = load_json(summary_path)
    models = selected_models(summary_payload, args.models)
    if not models:
        raise RuntimeError("No matching models found in run_summary_latest.json")

    graph_root = run_root / "graphs"
    if graph_root.exists():
        shutil.rmtree(graph_root)
    legacy_graph_root = run_root / "paper_graphs"
    if legacy_graph_root.exists():
        shutil.rmtree(legacy_graph_root)

    prepare_renderer_bundle(run_root, models)
    run_renderer("render_bundle_rome_success_metrics.py", run_root)
    run_renderer("render_bundle_detector_stacked_variants.py", run_root)
    run_renderer(
        "render_final_n500_detector_layer_window.py",
        run_root,
        extra_args=("--window-radius", str(args.window_radius)),
    )
    run_renderer(
        "render_artifacts/render_artifacts_grid.py",
        run_root,
        extra_args=("--allow-missing-models",),
    )
    write_readme(run_root, models, args.window_radius)
    print(f"[run-graphs] bundle renderers wrote graph outputs to {graph_root}")


if __name__ == "__main__":
    main()

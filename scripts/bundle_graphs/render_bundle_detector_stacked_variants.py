#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


IMPORT_ROOT = add_import_root(__file__)

from detector.composite_detector_v2 import _find_baseline, plot_signal_profiles_stacked


GRAPH_DIRNAME = "detector_stacked_variants"


def _latest_structural(run_root: Path, prefix: str) -> Path:
    structural_dir = run_root / "structural"
    search_root = structural_dir if structural_dir.is_dir() else run_root
    matches = sorted(search_root.glob(f"{prefix}_structural_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} structural JSON under {run_root}")
    return matches[-1]


def _run_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "run"


def _output_name(run_root: Path) -> str:
    return f"signals_stacked_{_run_slug(run_root.name)}.png"


def _iter_detector_runs(data_root: Path):
    for summary_path in sorted(data_root.rglob("*detector_summary.json")):
        run_root = summary_path.parent.parent
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        detector_type = str(payload.get("detector_type", "composite") or "composite")
        rome_path = _latest_structural(run_root, "rome")
        baseline_path = _find_baseline(rome_path)
        if baseline_path is None:
            try:
                baseline_path = _latest_structural(run_root, "baseline")
            except FileNotFoundError:
                baseline_path = None

        file_result = {
            "model": payload.get("model") or payload.get("hf_name") or run_root.name,
            "target_layer": payload.get("target_layer", payload.get("layer")),
            "n_tests": payload.get("n_tests", payload.get("det_eval", payload.get("tested", 0))),
            "accuracy": payload.get("accuracy", payload.get("det_rate", 0.0)),
            "path": str(rome_path),
            "baseline_path": str(baseline_path) if baseline_path is not None else None,
            "run_label": str(run_root.relative_to(data_root)),
            "run_slug": _run_slug(str(run_root.relative_to(data_root))),
            "trim": 5 if detector_type == "gpt" else 2,
            "te_window": 5,
            "small_window": 5,
            "large_window": 7,
        }
        yield run_root, file_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render stacked blind detector signal variants for all detector runs in final_n500_bundle.")
    parser.add_argument("--bundle-root", default=str(default_bundle_root(__file__)))
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    data_root = bundle_root / "data"
    graph_root = bundle_root / "graphs" / GRAPH_DIRNAME
    if graph_root.exists():
        shutil.rmtree(graph_root)
    graph_root.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for run_root, file_result in _iter_detector_runs(data_root):
        rel_parent = run_root.relative_to(data_root).parent
        output_dir = graph_root / rel_parent
        plot_signal_profiles_stacked(
            file_result,
            output_dir=output_dir,
            output_name=_output_name(run_root),
            include_signal_a=False,
            include_signal_b=False,
            include_te_raw=True,
            extra_te_windows=(7,),
        )
        rendered += 1

    print(f"Rendered {rendered} stacked detector-variant graphs under {graph_root}")


if __name__ == "__main__":
    main()

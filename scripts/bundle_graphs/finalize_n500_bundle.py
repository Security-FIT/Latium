#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


REPO_ROOT = add_import_root(__file__)
BUNDLE_ROOT = default_bundle_root(__file__)
INDEX_DIR = BUNDLE_ROOT / "index"
MANIFEST_DIR = BUNDLE_ROOT / "manifests"
FINAL_MODEL_GRAPHS_DIR = BUNDLE_ROOT / "graphs" / "final_per_model"
FINAL_AGGREGATE_GRAPHS_DIR = BUNDLE_ROOT / "graphs" / "final_aggregate"
SCRIPTS_FOR_GRAPHS_DIR = BUNDLE_ROOT / "scripts_for_graphs"

FINAL_ROOT = BUNDLE_ROOT / "data" / "final_n500_models_only"
PREFIX_ROOT = BUNDLE_ROOT / "data" / "prefixtest_n500_runs"
FLEET_RAW_ROOT = BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_remote"

SOURCE_N500_MANIFEST = REPO_ROOT / "manifests" / "counterfact_seed20260423_n500.json"
SOURCE_FLEET_MANIFEST = REPO_ROOT / "finetuned_qwen3_8b_fleet.json"
BUNDLE_FLEET_MANIFEST_NAME = "finetuned_qwen3_8b_fleet_in_bundle_by_downloads.json"

GRAPH_SCRIPT_SOURCE_FILES: list[tuple[Path, str]] = [
    (Path("pipeline_posthoc.py"), "entry_script"),
    (Path("scripts/bundle_graphs/bundle_paths.py"), "helper_module"),
    (Path("scripts/bundle_graphs/finalize_n500_bundle.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_bundle_rome_success_metrics.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_bundle_fleet_summary.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_bundle_prefixtest_graphs.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_bundle_windowed_detector_summary.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_bundle_detector_stacked_variants.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_final_n500_detector_layer_window.py"), "entry_script"),
    (Path("scripts/bundle_graphs/export_qwen_paper_graph_jsons.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_cohort_envelope_figure.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_cohort_fp_figure.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_artifacts/render_artifacts_grid.py"), "entry_script"),
    (Path("scripts/bundle_graphs/render_artifacts/finetune_models.txt"), "support_data"),
    (Path("paper_graphs/paper_graphs_support.py"), "helper_module"),
    (Path("paper_graphs/_newgen_utils.py"), "helper_module"),
    (Path("detector/composite_detector_v2.py"), "helper_module"),
    (Path("detector/gpt_detector.py"), "helper_module"),
    (Path("src/__init__.py"), "helper_module"),
    (Path("src/model_config.py"), "helper_module"),
    (Path("requirements.txt"), "environment"),
]

PRUNED_BUNDLE_PATHS = [
    BUNDLE_ROOT / "data" / "final_n500_local_subset",
    PREFIX_ROOT / "qwen3-8b-prefixtest-self-short-deprecated",
    BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_local_subset",
    BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_remote_first31_posthoc",
    BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_smoke",
    FLEET_RAW_ROOT / "_worker_logs",
    FLEET_RAW_ROOT / "logs",
    BUNDLE_ROOT / "graphs" / "fleet_posthoc",
    FINAL_AGGREGATE_GRAPHS_DIR,
    MANIFEST_DIR / "finetuned_qwen3_8b_fleet_top100_by_downloads.json",
]


def _source_file(repo_rel: Path, bundle_rel: Path | None = None) -> Path:
    repo_candidate = REPO_ROOT / repo_rel
    if repo_candidate.exists():
        return repo_candidate
    return BUNDLE_ROOT / (bundle_rel or repo_rel)


def configure_bundle_root(bundle_root: Path) -> None:
    global BUNDLE_ROOT, INDEX_DIR, MANIFEST_DIR
    global FINAL_MODEL_GRAPHS_DIR, FINAL_AGGREGATE_GRAPHS_DIR, SCRIPTS_FOR_GRAPHS_DIR
    global FINAL_ROOT, PREFIX_ROOT, FLEET_RAW_ROOT, SOURCE_N500_MANIFEST, SOURCE_FLEET_MANIFEST
    global PRUNED_BUNDLE_PATHS

    BUNDLE_ROOT = bundle_root.resolve()
    INDEX_DIR = BUNDLE_ROOT / "index"
    MANIFEST_DIR = BUNDLE_ROOT / "manifests"
    FINAL_MODEL_GRAPHS_DIR = BUNDLE_ROOT / "graphs" / "final_per_model"
    FINAL_AGGREGATE_GRAPHS_DIR = BUNDLE_ROOT / "graphs" / "final_aggregate"
    SCRIPTS_FOR_GRAPHS_DIR = BUNDLE_ROOT / "scripts_for_graphs"

    FINAL_ROOT = BUNDLE_ROOT / "data" / "final_n500_models_only"
    PREFIX_ROOT = BUNDLE_ROOT / "data" / "prefixtest_n500_runs"
    FLEET_RAW_ROOT = BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_remote"

    SOURCE_N500_MANIFEST = _source_file(
        Path("manifests/counterfact_seed20260423_n500.json"),
        Path("manifests/counterfact_seed20260423_n500.json"),
    )
    SOURCE_FLEET_MANIFEST = _source_file(
        Path("finetuned_qwen3_8b_fleet.json"),
        Path("manifests") / BUNDLE_FLEET_MANIFEST_NAME,
    )

    PRUNED_BUNDLE_PATHS = [
        BUNDLE_ROOT / "data" / "final_n500_local_subset",
        PREFIX_ROOT / "qwen3-8b-prefixtest-self-short-deprecated",
        BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_local_subset",
        BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_remote_first31_posthoc",
        BUNDLE_ROOT / "data" / "fleet_single_edit_runs" / "fleet_out_smoke",
        FLEET_RAW_ROOT / "_worker_logs",
        FLEET_RAW_ROOT / "logs",
        BUNDLE_ROOT / "graphs" / "fleet_posthoc",
        FINAL_AGGREGATE_GRAPHS_DIR,
        MANIFEST_DIR / "finetuned_qwen3_8b_fleet_top100_by_downloads.json",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bundle_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(BUNDLE_ROOT.resolve()))
    except Exception:
        return str(candidate)


def _existing_bundle_rel(path: Path) -> str:
    return _bundle_rel(path) if path.exists() else ""


def _repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(candidate)


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.replace("/", "_"))


def _latest_json(search_root: Path, prefix: str) -> Optional[Path]:
    matches = sorted(search_root.glob(f"{prefix}_*.json"))
    return matches[-1] if matches else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _prune_bundle() -> list[str]:
    removed: list[str] = []
    for path in PRUNED_BUNDLE_PATHS:
        if _remove_path(path):
            removed.append(_bundle_rel(path))
    removed.extend(_prune_fleet_raw_duplicates())
    return removed


def _prune_fleet_raw_duplicates() -> list[str]:
    removed: list[str] = []
    patterns = [
        ("baseline_structural_fleet_*.json", "baseline_structural_"),
        ("rome_structural_fleet_*.json", "rome_structural_"),
    ]
    for pattern, prefix in patterns:
        grouped: dict[str, list[Path]] = {}
        for path in sorted(FLEET_RAW_ROOT.glob(pattern)):
            grouped.setdefault(_parse_fleet_model_slug(path, prefix), []).append(path)
        for paths in grouped.values():
            keeper = _latest_valid_json(paths) or sorted(paths)[-1]
            for path in paths:
                if path == keeper:
                    continue
                if _remove_path(path):
                    removed.append(_bundle_rel(path))
    return removed


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    indices = [int(value) for value in manifest.get("indices", [])]
    case_ids = [int(value) for value in manifest.get("case_ids", [])] if manifest.get("case_ids") else []
    rows = []
    for order, dataset_index in enumerate(indices, start=1):
        row = {
            "order": order,
            "dataset_index": dataset_index,
            "case_id": case_ids[order - 1] if order - 1 < len(case_ids) else "",
        }
        rows.append(row)
    return rows


def _read_model_summary(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = run_root / "run_summary_latest.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        model_entry = (summary.get("models") or [{}])[0]
        return summary, model_entry

    structural_root = run_root / "structural"
    baseline_json = _latest_json(structural_root, "baseline_structural")
    rome_json = _latest_json(structural_root, "rome_structural")
    payload_path = rome_json or baseline_json
    payload = _load_json(payload_path) if payload_path else {}
    remote_status = _load_json(run_root / "remote_status.json") if (run_root / "remote_status.json").exists() else {}
    metadata = payload.get("metadata") or {}
    summary = payload.get("summary") or {}
    model_name = str(metadata.get("model") or metadata.get("model_name") or run_root.name)
    tested = _safe_int(summary.get("cases_total", summary.get("total", len(payload.get("tests") or []))))
    successful = _safe_int(summary.get("cases_completed", summary.get("successful", 0)))
    rome_ok = _safe_int(summary.get("rome_success_count", 0))
    det_eval = _safe_int(summary.get("detector_eligible_count", 0))

    fallback_summary = {
        "remote_status": remote_status,
        "case_selection": metadata.get("case_selection") or {},
        "run_root": "",
        "models": [
            {
                "model": model_name,
                "hf_name": str(metadata.get("hf_name") or ""),
                "tested": tested,
                "errors": _safe_int(summary.get("cases_error", summary.get("skipped", 0))),
                "rome_ok": rome_ok,
                "rome_rate": _safe_float(summary.get("rome_success_rate")),
                "det_eval": det_eval,
                "det_ok": _safe_int(summary.get("detector_success_count", 0)),
                "det_rate": _safe_float(summary.get("detector_success_rate")),
                "mean_overall_score": _safe_float(summary.get("mean_overall_score")),
                "mean_efficacy_score": _safe_float(summary.get("mean_efficacy_score")),
                "mean_paraphrase_score": _safe_float(summary.get("mean_paraphrase_score")),
                "mean_neighborhood_score": _safe_float(summary.get("mean_neighborhood_score")),
                "status": "partial_snapshot" if rome_json is None else "ok",
                "structural_json": str(payload_path or ""),
            }
        ],
    }
    fallback_entry = fallback_summary["models"][0]
    if not fallback_entry["tested"] and payload.get("tests"):
        fallback_entry["tested"] = len(payload.get("tests") or [])
    if not fallback_entry["rome_ok"] and successful:
        fallback_entry["rome_ok"] = successful
    return fallback_summary, fallback_entry


def _build_nested_run_row(run_root: Path, collection: str, deprecated: bool = False) -> dict[str, Any]:
    summary, model_entry = _read_model_summary(run_root)
    structural_root = run_root / "structural"
    baseline_json = _latest_json(structural_root, "baseline_structural")
    rome_json = _latest_json(structural_root, "rome_structural")
    case_selection = summary.get("case_selection") or {}
    remote_status = summary.get("remote_status") or {}
    manifest_dir = run_root / "manifest"
    manifest_json = _latest_json(manifest_dir, "counterfact") if manifest_dir.exists() else None

    return {
        "collection": collection,
        "run_slug": run_root.name,
        "model": str(model_entry.get("model") or run_root.name),
        "hf_name": str(model_entry.get("hf_name") or ""),
        "status": str(model_entry.get("status") or ""),
        "deprecated": bool(deprecated),
        "tested": _safe_int(model_entry.get("tested")),
        "rome_ok": _safe_int(model_entry.get("rome_ok")),
        "rome_rate": _safe_float(model_entry.get("rome_rate")),
        "det_eval": _safe_int(model_entry.get("det_eval")),
        "det_ok": _safe_int(model_entry.get("det_ok")),
        "det_rate": _safe_float(model_entry.get("det_rate")),
        "ovr": _safe_float(model_entry.get("mean_overall_score")),
        "es": _safe_float(model_entry.get("mean_efficacy_score")),
        "ps": _safe_float(model_entry.get("mean_paraphrase_score")),
        "ns": _safe_float(model_entry.get("mean_neighborhood_score")),
        "baseline_structural_json": _bundle_rel(baseline_json),
        "rome_structural_json": _bundle_rel(rome_json),
        "run_summary_latest_json": _existing_bundle_rel(run_root / "run_summary_latest.json"),
        "run_summary_latest_csv": _existing_bundle_rel(run_root / "run_summary_latest.csv"),
        "success_txt": _existing_bundle_rel(run_root / "success.txt"),
        "paper_graphs_dir": _existing_bundle_rel(run_root / "paper_graphs"),
        "manifest_in_run": _bundle_rel(manifest_json),
        "manifest_hash": str(case_selection.get("manifest_hash") or ""),
        "manifest_seed": case_selection.get("seed", ""),
        "manifest_count": _safe_int(case_selection.get("count")),
        "run_root": _bundle_rel(run_root),
        "remote_gpu": str(remote_status.get("gpu_name") or ""),
        "source_run_root": str(summary.get("run_root") or ""),
    }


def _parse_fleet_model_slug(path: Path, prefix: str) -> str:
    stem = path.stem.removeprefix(prefix)
    marker = "_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_s01_r01_"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def _latest_valid_json(paths: Iterable[Path]) -> Optional[Path]:
    for path in sorted(paths, reverse=True):
        try:
            _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        return path
    return None


def _fleet_case_row(model_id: str, rome_payload: dict[str, Any], rome_json: Path) -> dict[str, Any]:
    metadata = rome_payload.get("metadata") or {}
    selection = metadata.get("case_selection") or {}
    indices = [int(value) for value in selection.get("selected_dataset_indices", [])]
    case_ids = [int(value) for value in selection.get("selected_case_ids", [])]
    return {
        "model_id": model_id,
        "dataset_index": indices[0] if indices else "",
        "case_id": case_ids[0] if case_ids else "",
        "manifest_hash": str(selection.get("manifest_hash") or ""),
        "manifest_mode": str(selection.get("mode") or ""),
        "rome_structural_json": _bundle_rel(rome_json),
    }


def _final_model_graph_kind(relative_path: Path) -> str:
    nested_parts = relative_path.parts[1:]
    if len(nested_parts) == 1:
        name = nested_parts[0]
        if name == "detector_accuracy.png":
            return "detector_accuracy"
        if name == "rome_metric_means.png":
            return "rome_metric_means"
        if name.endswith("_paper_graphs.png"):
            return "paper_graphs"
    if len(nested_parts) == 2 and nested_parts[0] == "detector":
        return "detector_signal"
    return "other"


def _legacy_flat_graph_path(graph_path: Path) -> str:
    relative_path = graph_path.relative_to(FINAL_MODEL_GRAPHS_DIR)
    model = relative_path.parts[0]
    nested_parts = relative_path.parts[1:]

    if len(nested_parts) == 1:
        name = nested_parts[0]
        if name == "detector_accuracy.png":
            return f"graphs/final_aggregate/{model}_detector_accuracy.png"
        if name == "rome_metric_means.png":
            return f"graphs/final_aggregate/{model}_rome_metric_means.png"
        if name.endswith("_paper_graphs.png"):
            return f"graphs/final_aggregate/{name}"

    if len(nested_parts) == 2 and nested_parts[0] == "detector":
        return f"graphs/final_aggregate/detector/{nested_parts[1]}"

    return ""


def _build_final_model_graph_rows() -> list[dict[str, Any]]:
    if not FINAL_MODEL_GRAPHS_DIR.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(FINAL_MODEL_GRAPHS_DIR.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(FINAL_MODEL_GRAPHS_DIR)
        rows.append(
            {
                "model": relative_path.parts[0],
                "graph_kind": _final_model_graph_kind(relative_path),
                "file_name": path.name,
                "path_within_model": "/".join(relative_path.parts[1:]),
                "canonical_graph_path": _bundle_rel(path),
                "legacy_flat_export": _legacy_flat_graph_path(path),
            }
        )
    return rows


def _copy_graph_script_bundle() -> list[dict[str, Any]]:
    if REPO_ROOT.resolve() == SCRIPTS_FOR_GRAPHS_DIR.resolve():
        copied_files = SCRIPTS_FOR_GRAPHS_DIR / "copied_files.json"
        if copied_files.exists():
            try:
                return json.loads(copied_files.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return [
            {
                "kind": "self_contained",
                "source_path": "scripts_for_graphs",
                "copied_path": "scripts_for_graphs",
            }
        ]

    if SCRIPTS_FOR_GRAPHS_DIR.exists():
        shutil.rmtree(SCRIPTS_FOR_GRAPHS_DIR)
    SCRIPTS_FOR_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rel_path, kind in GRAPH_SCRIPT_SOURCE_FILES:
        source = REPO_ROOT / rel_path
        destination = SCRIPTS_FOR_GRAPHS_DIR / rel_path
        if not _copy_if_exists(source, destination):
            continue
        rows.append(
            {
                "kind": kind,
                "source_path": _repo_rel(source),
                "copied_path": _bundle_rel(destination),
            }
        )

    runner_source = REPO_ROOT / "scripts" / "bundle_graphs" / "run_all_graphs.sh"
    runner_destination = SCRIPTS_FOR_GRAPHS_DIR / "run_all_graphs.sh"
    if _copy_if_exists(runner_source, runner_destination):
        runner_destination.chmod(0o755)
        rows.append(
            {
                "kind": "entry_script",
                "source_path": _repo_rel(runner_source),
                "copied_path": _bundle_rel(runner_destination),
            }
        )

    config_root = REPO_ROOT / "src" / "config" / "model"
    for source in sorted(config_root.glob("*.yaml")):
        destination = SCRIPTS_FOR_GRAPHS_DIR / "src" / "config" / "model" / source.name
        if not _copy_if_exists(source, destination):
            continue
        rows.append(
            {
                "kind": "model_config",
                "source_path": _repo_rel(source),
                "copied_path": _bundle_rel(destination),
            }
        )

    readme = f"""# scripts_for_graphs

This directory is a self-contained copy of the repo files that were used to render the graphs stored in `final_n500_bundle/graphs/`.

It is intentionally limited to plotting code and the local helper/config files those plots depend on.

## Contents

- `run_all_graphs.sh` to regenerate all final-paper graph groups from inside this bundle
- `pipeline_posthoc.py` for the copied final-model graph stack
- `scripts/bundle_graphs/*.py` entry points for the bundle-level graph groups
- `paper_graphs/*.py` helper modules used by the paper-style plots
- `detector/*.py` helper modules used by the detector-based plots
- `src/config/model/*.yaml` model metadata used by `pipeline_posthoc.py`
- `requirements.txt` for the Python package baseline

## Notes

- These files were copied with their repo-relative layout so another person can inspect or adapt the plotting logic without the rest of the benchmark pipeline.
- From a downloaded bundle, run `bash scripts_for_graphs/run_all_graphs.sh`.
- The JSON inputs are the ones already stored elsewhere in this same bundle under `data/` and `graphs/windowed_detector/`.
- `index/render_sources.csv` maps each graph group to the entry script that produced it.
- `copied_files.csv` lists every file copied into this directory.
"""
    (SCRIPTS_FOR_GRAPHS_DIR / "README.md").write_text(readme, encoding="utf-8")
    _write_csv(SCRIPTS_FOR_GRAPHS_DIR / "copied_files.csv", rows)
    _write_json(SCRIPTS_FOR_GRAPHS_DIR / "copied_files.json", rows)
    return rows


def build_catalogs(bundle_root: Path) -> dict[str, Any]:
    final_rows = [
        _build_nested_run_row(run_root, collection="final_models_n500")
        for run_root in sorted(FINAL_ROOT.iterdir())
        if run_root.is_dir()
    ]
    prefix_rows = [
        _build_nested_run_row(
            run_root,
            collection="prefix_qwens_n500",
            deprecated=False,
        )
        for run_root in sorted(PREFIX_ROOT.iterdir())
        if run_root.is_dir() and "deprecated" not in run_root.name
    ]

    source_fleet_manifest = _load_json(SOURCE_FLEET_MANIFEST)
    baseline_map: dict[str, list[Path]] = {}
    rome_map: dict[str, list[Path]] = {}
    for path in sorted(FLEET_RAW_ROOT.glob("baseline_structural_fleet_*.json")):
        baseline_map.setdefault(_parse_fleet_model_slug(path, "baseline_structural_"), []).append(path)
    for path in sorted(FLEET_RAW_ROOT.glob("rome_structural_fleet_*.json")):
        rome_map.setdefault(_parse_fleet_model_slug(path, "rome_structural_"), []).append(path)

    fleet_rows: list[dict[str, Any]] = []
    fleet_edit_rows: list[dict[str, Any]] = []
    bundle_fleet_models: list[dict[str, Any]] = []
    for entry in source_fleet_manifest.get("models", []):
        model_id = str(entry["model_id"])
        slug = f"fleet_{_sanitize(model_id)}"
        baseline_json = _latest_valid_json(baseline_map.get(slug, []))
        rome_json = _latest_valid_json(rome_map.get(slug, []))
        tried = baseline_json is not None or rome_json is not None
        completed = rome_json is not None
        if not tried:
            continue

        bundle_fleet_models.append(entry)
        baseline_payload = _load_json(baseline_json) if baseline_json else {}
        rome_payload = _load_json(rome_json) if rome_json else {}
        baseline_summary = baseline_payload.get("summary") or {}
        rome_summary = rome_payload.get("summary") or {}
        row = {
            "rank_by_hf_downloads": len(bundle_fleet_models),
            "model_id": model_id,
            "downloads": _safe_int(entry.get("downloads")),
            "likes": _safe_int(entry.get("likes")),
            "tried_in_bundle": tried,
            "completed_rome": completed,
            "baseline_structural_json": _bundle_rel(baseline_json),
            "rome_structural_json": _bundle_rel(rome_json),
            "baseline_tests": _safe_int(baseline_summary.get("total", len(baseline_payload.get("tests") or []))),
            "rome_tests": _safe_int(rome_summary.get("total", len(rome_payload.get("tests") or []))),
            "rome_success_count": _safe_int(rome_summary.get("rome_success_count")),
            "mean_overall_score": _safe_float(rome_summary.get("mean_overall_score")),
            "mean_efficacy_score": _safe_float(rome_summary.get("mean_efficacy_score")),
            "mean_paraphrase_score": _safe_float(rome_summary.get("mean_paraphrase_score")),
            "mean_neighborhood_score": _safe_float(rome_summary.get("mean_neighborhood_score")),
            "tags": ";".join(str(tag) for tag in entry.get("tags", [])),
        }
        fleet_rows.append(row)
        if completed:
            fleet_edit_rows.append(_fleet_case_row(model_id, rome_payload, rome_json))

    fleet_manifest = dict(source_fleet_manifest)
    fleet_manifest["models"] = bundle_fleet_models
    fleet_manifest["bundle_model_count"] = len(bundle_fleet_models)
    fleet_manifest["bundle_manifest_name"] = BUNDLE_FLEET_MANIFEST_NAME
    fleet_manifest["bundle_selection_note"] = (
        "Filtered to the models actually present in final_n500_bundle/data/fleet_single_edit_runs/fleet_out_remote."
    )

    n500_manifest = _load_json(SOURCE_N500_MANIFEST)
    manifest_rows = _manifest_rows(n500_manifest)

    return {
        "final_rows": final_rows,
        "prefix_rows": prefix_rows,
        "fleet_rows": fleet_rows,
        "fleet_edit_rows": fleet_edit_rows,
        "manifest_rows": manifest_rows,
        "n500_manifest": n500_manifest,
        "fleet_manifest": fleet_manifest,
    }


def _write_render_sources() -> None:
    rows = [
        {
            "graph_group": "final_per_model",
            "output_dir": "graphs/final_per_model",
            "source_script": "pipeline_posthoc.py + paper_graphs/paper_graphs_support.py",
            "input_data_roots": "data/final_n500_models_only",
        },
        {
            "graph_group": "rome_success_metrics",
            "output_dir": "graphs/rome_success_metrics",
            "source_script": "scripts/bundle_graphs/render_bundle_rome_success_metrics.py",
            "input_data_roots": "data/final_n500_models_only; data/prefixtest_n500_runs; data/fleet_single_edit_runs/fleet_out_remote",
        },
        {
            "graph_group": "fleet_summary",
            "output_dir": "graphs/fleet_summary",
            "source_script": "scripts/bundle_graphs/render_bundle_fleet_summary.py",
            "input_data_roots": "data/fleet_single_edit_runs/fleet_out_remote; data/final_n500_models_only/qwen3-8b",
        },
        {
            "graph_group": "prefixtest_summary",
            "output_dir": "graphs/prefixtest_summary",
            "source_script": "scripts/bundle_graphs/render_bundle_prefixtest_graphs.py",
            "input_data_roots": "data/prefixtest_n500_runs; data/final_n500_models_only/qwen3-8b",
        },
        {
            "graph_group": "windowed_detector",
            "output_dir": "graphs/windowed_detector",
            "source_script": "detector/composite_detector_v2.py + scripts/bundle_graphs/render_bundle_windowed_detector_summary.py",
            "input_data_roots": "data/fleet_single_edit_runs/fleet_out_remote",
        },
        {
            "graph_group": "detector_stacked_variants",
            "output_dir": "graphs/detector_stacked_variants",
            "source_script": "scripts/bundle_graphs/render_bundle_detector_stacked_variants.py",
            "input_data_roots": "data/final_n500_models_only; data/prefixtest_n500_runs; data/fleet_single_edit_runs",
        },
        {
            "graph_group": "detector_layer_window",
            "output_dir": "graphs/detector_layer_window",
            "source_script": "scripts/bundle_graphs/render_final_n500_detector_layer_window.py",
            "input_data_roots": "data/final_n500_models_only",
        },
        {
            "graph_group": "qwen_paper_graph_json",
            "output_dir": "graphs/qwen_paper_graph_json",
            "source_script": "scripts/bundle_graphs/export_qwen_paper_graph_jsons.py",
            "input_data_roots": "data/final_n500_models_only/qwen3-8b",
        },
        {
            "graph_group": "cohort_envelope",
            "output_dir": "graphs/fleet_summary",
            "source_script": "scripts/bundle_graphs/render_cohort_envelope_figure.py",
            "input_data_roots": "data/fleet_single_edit_runs/fleet_out_remote; data/final_n500_models_only/qwen3-8b",
        },
        {
            "graph_group": "cohort_false_positive",
            "output_dir": "graphs/fleet_summary",
            "source_script": "scripts/bundle_graphs/render_cohort_fp_figure.py",
            "input_data_roots": "data/fleet_single_edit_runs/fleet_out_remote; graphs/windowed_detector",
        },
        {
            "graph_group": "artifact_grids",
            "output_dir": "graphs/artifacts_grid",
            "source_script": "scripts/bundle_graphs/render_artifacts/render_artifacts_grid.py",
            "input_data_roots": "data/final_n500_models_only",
        },
    ]
    _write_csv(INDEX_DIR / "render_sources.csv", rows)
    _write_json(INDEX_DIR / "render_sources.json", rows)


def _flatten_all_runs(
    final_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    fleet_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in final_rows + prefix_rows:
        rows.append(
            {
                "group": row["collection"],
                "name": row["run_slug"],
                "model": row["model"],
                "hf_name": row["hf_name"],
                "status": row.get("status", ""),
                "deprecated": row.get("deprecated", False),
                "tested": row.get("tested", 0),
                "rome_ok": row.get("rome_ok", 0),
                "det_ok": row.get("det_ok", 0),
                "ovr": row.get("ovr", 0.0),
                "baseline_structural_json": row.get("baseline_structural_json", ""),
                "rome_structural_json": row.get("rome_structural_json", ""),
                "manifest_hash": row.get("manifest_hash", ""),
                "source_path": row.get("run_root", ""),
            }
        )
    for row in fleet_rows:
        rows.append(
            {
                "group": "hf_finetuned_qwens",
                "name": row["model_id"],
                "model": row["model_id"],
                "hf_name": row["model_id"],
                "status": "completed" if row["completed_rome"] else ("baseline_only" if row["tried_in_bundle"] else "not_copied"),
                "deprecated": False,
                "tested": row.get("rome_tests", 0),
                "rome_ok": row.get("rome_success_count", 0),
                "det_ok": "",
                "ovr": row.get("mean_overall_score", 0.0),
                "baseline_structural_json": row.get("baseline_structural_json", ""),
                "rome_structural_json": row.get("rome_structural_json", ""),
                "manifest_hash": "",
                "source_path": row.get("baseline_structural_json") or row.get("rome_structural_json") or "",
            }
        )
    return rows


def _write_inventory_markdown(payload: dict[str, Any]) -> None:
    final_rows = payload["final_rows"]
    prefix_rows = payload["prefix_rows"]
    fleet_rows = payload["fleet_rows"]
    fleet_edit_rows = payload["fleet_edit_rows"]

    completed_fleet = sum(1 for row in fleet_rows if row["completed_rome"])
    tried_fleet = len(fleet_rows)
    text = f"""# Final N500 Bundle Index

## Fact-checked overview

- Final N500 model runs: {len(final_rows)}
- Prefix Qwen runs copied into bundle: {len(prefix_rows)}
- HF fleet models kept in bundle: {len(fleet_rows)}
- HF fleet models with copied edited JSON in bundle: {completed_fleet}
- Fleet one-edit cases recovered from edited JSON metadata: {len(fleet_edit_rows)}

## What was used to pick the HF fleet?

- Bundle manifest: `manifests/{BUNDLE_FLEET_MANIFEST_NAME}`
- Query filter: `base_model:finetune:Qwen/Qwen3-8B-Base`
- Sort key: Hugging Face `downloads`, descending
- Raw query limit in the fetch script: 300 models
- Then filtered to keep text-generation or untagged models, excluding official/base redistributions
- Final kept set in this bundle: {tried_fleet} copied fleet models

## Canonical entry points

- Final models table: `index/final_models_n500.csv`
- Final-model graph lookup: `index/final_model_graphs.csv`
- Rebuild all graphs from bundle: `scripts_for_graphs/run_all_graphs.sh`
- Graph script handoff bundle: `scripts_for_graphs/`
- Prefix Qwens table: `index/prefix_qwens_n500.csv`
- HF fleet table: `index/hf_finetuned_qwens.csv`
- JSON layout note: `index/json_structure.md`
- Shared N=500 edit manifest: `manifests/counterfact_seed20260423_n500.json`
- Flattened N=500 edit list: `index/counterfact_seed20260423_n500.csv`
- Fleet one-edit case list: `index/fleet_edit_cases.csv`
- Graph/input mapping: `index/render_sources.csv`

## Graph layout

- `graphs/final_per_model/` is the canonical final-model graph root.
- The old flattened `graphs/final_aggregate/` copies were removed because they were exact byte-for-byte duplicates.
- Use `index/final_model_graphs.csv` if you want the old flat lookup without storing a second copy of the same PNGs.
- `scripts_for_graphs/` contains verbatim copies of the plotting scripts, helper modules, and config files used to render these graphs.
"""
    (INDEX_DIR / "bundle_inventory.md").write_text(text, encoding="utf-8")


def _write_json_structure_markdown() -> None:
        text = """# Bundle JSON Structure

This is the bundle-local quick reference for the copied JSON artifacts.

It condenses the existing repo notes from `preparation.md`, `docs/detector.md`, and `docs/remote_h100_runbook.md` so the bundle stays self-describing.

## 1. Structural benchmark JSONs

Main locations:

- `data/final_n500_models_only/*/structural/*.json`
- `data/prefixtest_n500_runs/*/structural/*.json`
- `data/fleet_single_edit_runs/fleet_out_remote/*.json`

Representative top-level layout:

```json
{
    "metadata": { ... },
    "tests": [ ... ],
    "summary": { ... }
}
```

Common `metadata` keys:

- `model`
- `target_layer`
- `n_tests`, `n_prompts`
- `analysis_profile`
- `baseline_only`
- `spectral_config`
- `analytics_config`
- `run_index`, `run_ordinal`, `runs_per_model`
- `start_idx_used`, `end_idx_used`
- `sweep_index`, `sweep_size`, `sweep_config`, `sweep_slug`
- `total_runs_for_model`
- `case_selection`

Common `metadata.case_selection` keys:

- `mode`
- `dataset`, `split`
- `seed`
- `manifest_path`, `manifest_hash`
- `count`
- `selected_dataset_indices`
- `selected_case_ids`

Representative `tests[]` fields:

- `case_id`
- `subject`
- `error`
- `rome`
- `accuracy`
- `blind_detection.layer_features[layer_idx]`
- `spectral_detection` for `paper` and `full` payloads

Representative `summary` fields:

- `total`, `successful`, `skipped`
- `cases_total`, `cases_completed`, `cases_error`
- `rome_success_count`, `rome_success_rate`
- `detector_eligible_count`
- `mean_efficacy_score`, `mean_efficacy_magnitude`
- `mean_paraphrase_score`, `mean_neighborhood_score`, `mean_overall_score`

Notes:

- Edited files are the `rome_structural_*.json` artifacts.
- Baseline files are the `baseline_structural_*.json` artifacts.
- Final-model and prefix runs are N=500 payloads with the shared manifest hash shown in `index/final_models_n500.csv` and `index/prefix_qwens_n500.csv`.
- Fleet files are one-model mirrors; many are one baseline case plus at most one edited case per model.

## 2. Run summary JSONs

Main locations:

- `data/final_n500_models_only/*/run_summary_latest.json`
- `data/prefixtest_n500_runs/*/run_summary_latest.json`

Representative top-level layout:

```json
{
    "timestamp": "...",
    "remote_host": "...",
    "remote_status": { ... },
    "start_idx": 0,
    "n_tests": 500,
    "counterfact_policy": { ... },
    "case_selection": { ... },
    "run_root": "...",
    "models": [ ... ]
}
```

Common `remote_status` keys:

- `hostname`
- `gpu_name`
- `worker_id`
- `models`
- `n_tests`
- `start_idx`
- `slice_policy`
- `model_assignments`
- `compute_cov`, `cov_source`
- `case_index_file`
- `started_at`, `finished_at`
- `phase`, `failed`

Representative `models[]` fields:

- `model`, `hf_name`, `layer`
- `assigned_start_idx`, `assigned_end_idx`
- `actual_start_idx`, `actual_end_idx`
- `status`, `tested`, `errors`
- `rome_ok`, `rome_rate`
- `det_eval`, `det_ok`, `det_rate`
- `mean_efficacy_score`, `mean_efficacy_magnitude`
- `mean_paraphrase_score`, `mean_neighborhood_score`, `mean_overall_score`
- `structural_json`
- `detector_summary`
- `detector_sweep_report`
- `detector_sweep_best_accuracy`, `detector_sweep_best_config`
- `detector_type`
- `case_selection_mode`, `case_selection_manifest`, `case_selection_manifest_hash`
- `case_selection_seed`, `case_selection_count`
- `case_selection_indices`, `case_selection_case_ids`
- `structural_error`

## 3. Manifest JSONs

Bundle-local manifest files:

- `manifests/counterfact_seed20260423_n500.json`
- `manifests/finetuned_qwen3_8b_fleet_in_bundle_by_downloads.json`

The CounterFact manifest is the shared edit list for the final-model and prefix N=500 runs.

The HF fleet manifest is already filtered down to the 99 models actually kept in this bundle, ordered by the original HF-download ranking.

## 4. Preferred Entry Points

Use the flat `index/*.csv` tables first when you want to inspect the bundle programmatically.

Use the raw JSONs under `data/` only when you need per-test payloads, detector inputs, or exact run metadata.
"""
        (INDEX_DIR / "json_structure.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize final_n500_bundle indices and graph-script handoff files.")
    parser.add_argument("--bundle-root", type=Path, default=default_bundle_root(__file__))
    args = parser.parse_args()
    configure_bundle_root(args.bundle_root)

    removed_paths = _prune_bundle()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    payload = build_catalogs(BUNDLE_ROOT)

    copied_manifests = []
    if _copy_if_exists(SOURCE_N500_MANIFEST, MANIFEST_DIR / SOURCE_N500_MANIFEST.name):
        copied_manifests.append("counterfact_seed20260423_n500.json")
    _write_json(MANIFEST_DIR / BUNDLE_FLEET_MANIFEST_NAME, payload["fleet_manifest"])
    copied_manifests.append(BUNDLE_FLEET_MANIFEST_NAME)

    _write_csv(INDEX_DIR / "final_models_n500.csv", payload["final_rows"])
    _write_json(INDEX_DIR / "final_models_n500.json", payload["final_rows"])
    _write_csv(INDEX_DIR / "prefix_qwens_n500.csv", payload["prefix_rows"])
    _write_json(INDEX_DIR / "prefix_qwens_n500.json", payload["prefix_rows"])
    _write_csv(INDEX_DIR / "hf_finetuned_qwens.csv", payload["fleet_rows"])
    _write_json(INDEX_DIR / "hf_finetuned_qwens.json", payload["fleet_rows"])
    _write_csv(INDEX_DIR / "fleet_edit_cases.csv", payload["fleet_edit_rows"])
    _write_json(INDEX_DIR / "fleet_edit_cases.json", payload["fleet_edit_rows"])
    _write_csv(INDEX_DIR / "counterfact_seed20260423_n500.csv", payload["manifest_rows"])
    _write_json(INDEX_DIR / "counterfact_seed20260423_n500.json", payload["manifest_rows"])
    all_run_rows = _flatten_all_runs(payload["final_rows"], payload["prefix_rows"], payload["fleet_rows"])
    _write_csv(INDEX_DIR / "all_bundle_runs.csv", all_run_rows)
    _write_json(INDEX_DIR / "all_bundle_runs.json", all_run_rows)
    final_model_graph_rows = _build_final_model_graph_rows()
    _write_csv(INDEX_DIR / "final_model_graphs.csv", final_model_graph_rows)
    _write_json(INDEX_DIR / "final_model_graphs.json", final_model_graph_rows)

    overview_rows = [
        {
            "collection": "final_models_n500",
            "count": len(payload["final_rows"]),
            "root": "data/final_n500_models_only",
            "notes": f"{len(payload['final_rows'])} final N=500 model runs with baseline and edited structural JSONs.",
        },
        {
            "collection": "prefix_qwens_n500",
            "count": len(payload["prefix_rows"]),
            "root": "data/prefixtest_n500_runs",
            "notes": "9 current prefix variants with the deprecated self-short snapshot removed.",
        },
        {
            "collection": "hf_finetuned_qwens",
            "count": len(payload["fleet_rows"]),
            "root": "data/fleet_single_edit_runs/fleet_out_remote",
            "notes": (
                f"Kept only the {len(payload['fleet_rows'])} copied fleet models, ordered by HF downloads after filtering; "
                f"{sum(1 for row in payload['fleet_rows'] if row['completed_rome'])} edited outputs completed."
            ),
        },
    ]
    _write_csv(INDEX_DIR / "collection_overview.csv", overview_rows)
    _write_json(INDEX_DIR / "collection_overview.json", overview_rows)
    _write_render_sources()
    _write_inventory_markdown(payload)
    _write_json_structure_markdown()
    copied_graph_scripts = _copy_graph_script_bundle()

    selection_md = f"""# HF Fleet Selection

- Bundle manifest file: `manifests/finetuned_qwen3_8b_fleet_in_bundle_by_downloads.json`
- Source URL: `https://huggingface.co/models?other=base_model:finetune:Qwen%2FQwen3-8B-Base`
- Selection script: `scripts/fetch_finetuned_qwen3_8b.py`
- Raw query: up to 300 models tagged `base_model:finetune:Qwen/Qwen3-8B-Base`
- Sort key: Hugging Face `downloads`, descending
- Filters after the query:
  - exclude official Qwen/Qwen3-8B variants and other base redistributions
  - keep `text-generation` or untagged models
- Bundle manifest size: {len(payload["fleet_rows"])} copied models
- Copied raw bundle coverage fact-check: {len(payload["fleet_rows"])} baseline models, {sum(1 for row in payload["fleet_rows"] if row["completed_rome"])} completed edited structural outputs
"""
    (INDEX_DIR / "hf_download_selection.md").write_text(selection_md, encoding="utf-8")

    if removed_paths:
        print(f"Pruned bundle paths: {', '.join(removed_paths)}")
    print(f"Copied manifests: {', '.join(copied_manifests) if copied_manifests else 'none'}")
    print(f"Copied graph script bundle files: {len(copied_graph_scripts)}")
    print(f"Wrote flattened bundle indices to {INDEX_DIR}")


if __name__ == "__main__":
    main()

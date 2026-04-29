#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root, display_path  # noqa: E402


IMPORT_ROOT = add_import_root(__file__)


SUMMARY_DIRNAME = "rome_success_metrics"
METRIC_SPECS = [
    ("OVR", "ovr", "#6a3d9a"),
    ("ES", "es", "#1f77b4"),
    ("PS", "ps", "#2ca02c"),
    ("NS", "ns", "#ff7f0e"),
]


@dataclass(frozen=True)
class CollectionSpec:
    slug: str
    label: str
    color: str
    nested: bool
    root: str


@dataclass(frozen=True)
class RomeMetricRecord:
    collection_slug: str
    collection_label: str
    display_name: str
    model: str
    run_root: str
    structural_json: str
    tested: int
    completed: int
    errors: int
    rome_ok: int
    rome_rate: float
    ovr: float
    es: float
    ps: float
    ns: float
    target_layer: int | None

    def as_row(self) -> dict:
        return {
            "collection": self.collection_slug,
            "collection_label": self.collection_label,
            "display_name": self.display_name,
            "model": self.model,
            "run_root": self.run_root,
            "target_layer": "" if self.target_layer is None else self.target_layer,
            "tested": self.tested,
            "completed": self.completed,
            "errors": self.errors,
            "rome_success_count": self.rome_ok,
            "rome_success_rate": self.rome_rate,
            "OVR": self.ovr,
            "ES": self.es,
            "PS": self.ps,
            "NS": self.ns,
            "structural_json": self.structural_json,
        }


COLLECTIONS: Sequence[CollectionSpec] = (
    CollectionSpec(
        slug="final_n500_models_only",
        label="Final N500 Models",
        color="#1f77b4",
        nested=True,
        root="data/final_n500_models_only",
    ),
    CollectionSpec(
        slug="prefixtest_n500_runs",
        label="Prefixtest N500 Runs",
        color="#0f766e",
        nested=True,
        root="data/prefixtest_n500_runs",
    ),
    CollectionSpec(
        slug="fleet_single_edit_runs",
        label="Fleet Single-Edit Runs",
        color="#b45309",
        nested=False,
        root="data/fleet_single_edit_runs/fleet_out_remote",
    ),
)


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.22,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.framealpha": 0.95,
        }
    )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Iterable[object]) -> float:
    numeric = [safe_float(value, default=np.nan) for value in values]
    filtered = [value for value in numeric if np.isfinite(value)]
    if not filtered:
        return 0.0
    return float(sum(filtered) / len(filtered))


def rel_to_bundle(path: Path, bundle_root: Path) -> str:
    return display_path(path, bundle_root=bundle_root, import_root=IMPORT_ROOT)


def summarize_payload(payload: dict) -> dict:
    summary = payload.get("summary") or {}
    tests = payload.get("tests") or []
    completed = [test for test in tests if not test.get("error") and not test.get("skipped")]
    errors = [test for test in tests if test.get("error") or test.get("skipped")]
    rome_success_tests = [test for test in completed if test.get("rome", {}).get("success")]

    tested = safe_int(summary.get("cases_total", summary.get("total", len(tests))))
    completed_count = safe_int(summary.get("cases_completed", summary.get("successful", len(completed))))
    error_count = safe_int(summary.get("cases_error", summary.get("skipped", len(errors))))
    rome_ok = safe_int(summary.get("rome_success_count", len(rome_success_tests)))
    rome_rate = safe_float(
        summary.get("rome_success_rate", (rome_ok / completed_count) if completed_count else 0.0)
    )

    return {
        "tested": tested,
        "completed": completed_count,
        "errors": error_count,
        "rome_ok": rome_ok,
        "rome_rate": rome_rate,
        "es": safe_float(summary.get("mean_efficacy_score"), safe_mean(test.get("rome", {}).get("efficacy_score") for test in completed)),
        "ps": safe_float(summary.get("mean_paraphrase_score"), safe_mean(test.get("rome", {}).get("paraphrase_score") for test in completed)),
        "ns": safe_float(summary.get("mean_neighborhood_score"), safe_mean(test.get("rome", {}).get("neighborhood_score") for test in completed)),
        "ovr": safe_float(summary.get("mean_overall_score"), safe_mean(test.get("rome", {}).get("overall_score") for test in completed)),
    }


def shorten_label(text: str, max_len: int = 34) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def fleet_display_name(model: str) -> str:
    clean = model.removeprefix("fleet_")
    return clean.replace("_", " / ", 1)


def prefix_sort_key(name: str) -> tuple[int, int, int, str]:
    match = re.search(r"prefixtest-(self|template|external)-(short|medium|long)", name)
    mode_order = {"self": 0, "template": 1, "external": 2}
    length_order = {"short": 0, "medium": 1, "long": 2}
    if not match:
        return (99, 99, 1 if "deprecated" in name else 0, name)
    mode, length = match.groups()
    deprecated = 1 if "deprecated" in name else 0
    return (mode_order[mode], length_order[length], deprecated, name)


def record_sort_key(record: RomeMetricRecord) -> tuple:
    if record.collection_slug == "final_n500_models_only":
        return (0, record.display_name.lower())
    if record.collection_slug == "prefixtest_n500_runs":
        return (1, *prefix_sort_key(record.display_name))
    return (2, -record.ovr, record.display_name.lower())


def collect_json_paths(bundle_root: Path, spec: CollectionSpec) -> List[Path]:
    root = bundle_root / spec.root
    if spec.nested:
        paths = []
        for run_root in sorted(child for child in root.iterdir() if child.is_dir()):
            matches = sorted((run_root / "structural").glob("rome_structural_*.json"))
            paths.extend(matches)
        return paths
    return sorted(root.glob("rome_structural_*.json"))


def nested_display_name(run_root: Path, json_path: Path) -> str:
    matches = sorted((run_root / "structural").glob("rome_structural_*.json"))
    if len(matches) <= 1:
        return run_root.name

    stem_tail = json_path.stem.removeprefix("rome_structural_")
    prefix = run_root.name + "_"
    if stem_tail.startswith(prefix):
        stem_tail = stem_tail[len(prefix):]

    if stem_tail.startswith("runner_failure"):
        tag = "runner_failure"
    else:
        parts = stem_tail.split("_")
        tag = "_".join(parts[-2:]) if len(parts) >= 2 else stem_tail
    return f"{run_root.name} [{tag}]"


def build_record(bundle_root: Path, json_path: Path, spec: CollectionSpec) -> RomeMetricRecord:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    summary = summarize_payload(payload)
    rel_json = rel_to_bundle(json_path, bundle_root)

    if spec.nested:
        run_root = json_path.parent.parent
        display_name = nested_display_name(run_root, json_path)
    else:
        run_root = json_path.parent
        display_name = fleet_display_name(str(metadata.get("model") or json_path.stem.removeprefix("rome_structural_")))

    model = str(metadata.get("model") or display_name)
    return RomeMetricRecord(
        collection_slug=spec.slug,
        collection_label=spec.label,
        display_name=display_name,
        model=model,
        run_root=rel_to_bundle(run_root, bundle_root),
        structural_json=str(rel_json),
        tested=summary["tested"],
        completed=summary["completed"],
        errors=summary["errors"],
        rome_ok=summary["rome_ok"],
        rome_rate=summary["rome_rate"],
        ovr=summary["ovr"],
        es=summary["es"],
        ps=summary["ps"],
        ns=summary["ns"],
        target_layer=safe_int(metadata.get("target_layer"), default=None) if metadata.get("target_layer") is not None else None,
    )


def collect_records(bundle_root: Path) -> List[RomeMetricRecord]:
    records: List[RomeMetricRecord] = []
    for spec in COLLECTIONS:
        for json_path in collect_json_paths(bundle_root, spec):
            records.append(build_record(bundle_root, json_path, spec))
    return sorted(records, key=record_sort_key)


def write_tables(records: Sequence[RomeMetricRecord], output_dir: Path) -> None:
    rows = [record.as_row() for record in records]
    json_path = output_dir / "rome_success_metrics_all_runs.json"
    csv_path = output_dir / "rome_success_metrics_all_runs.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for spec in COLLECTIONS:
        group_rows = [record.as_row() for record in records if record.collection_slug == spec.slug]
        if not group_rows:
            continue
        group_csv = output_dir / f"rome_success_metrics_{spec.slug}.csv"
        with group_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)


def render_metric_heatmap(records: Sequence[RomeMetricRecord], out_path: Path, title: str) -> None:
    setup_style()
    matrix = np.array([[getattr(record, key) for _label, key, _color in METRIC_SPECS] for record in records], dtype=float)
    y_labels = [shorten_label(record.display_name, max_len=42) for record in records]
    metric_labels = [label for label, _key, _color in METRIC_SPECS]
    fig_height = max(4.5, min(36.0, 1.7 + 0.34 * len(records)))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_labels)), metric_labels)
    ax.set_yticks(np.arange(len(y_labels)), y_labels)
    ax.tick_params(axis="y", labelsize=max(6, 11 - 0.04 * len(records)))
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("Mean score", fontsize=12)
    if len(records) <= 20:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value < 0.7 else "black",
                )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metric_bars(records: Sequence[RomeMetricRecord], out_path: Path, title: str) -> None:
    setup_style()
    x = np.arange(len(records))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(records) * 1.05), 6.4))
    for index, (label, key, color) in enumerate(METRIC_SPECS):
        values = [getattr(record, key) for record in records]
        ax.bar(x + (index - 1.5) * width, values, width=width, color=color, label=label)
    ax.set_xticks(x, [record.display_name.replace("-", "\n", 1) for record in records])
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Mean score")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(axis="y")
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_overall_ranking(records: Sequence[RomeMetricRecord], out_path: Path) -> None:
    setup_style()
    ordered = sorted(records, key=lambda record: (record.ovr, record.es, record.ps, record.ns), reverse=True)
    fig_height = max(7.0, min(34.0, 1.8 + 0.26 * len(ordered)))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y = np.arange(len(ordered))
    collection_colors = {spec.slug: spec.color for spec in COLLECTIONS}
    colors = [collection_colors[record.collection_slug] for record in ordered]
    ax.barh(y, [record.ovr for record in ordered], color=colors)
    ax.set_yticks(y, [shorten_label(record.display_name, max_len=44) for record in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("OVR (stored mean_overall_score)")
    ax.set_title("ROME overall score ranking across all bundle runs", fontsize=16, fontweight="bold")
    ax.grid(axis="x")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=spec.color, label=spec.label)
        for spec in COLLECTIONS
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render bundle-wide ROME success metric tables and graphs from stored structural summaries.")
    parser.add_argument("--bundle-root", default=str(default_bundle_root(__file__)))
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    output_dir = bundle_root / "graphs" / SUMMARY_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(bundle_root)
    if not records:
        raise RuntimeError(f"No rome_structural_*.json files found under {bundle_root / 'data'}")

    write_tables(records, output_dir)
    render_metric_heatmap(records, output_dir / "rome_success_metrics_all_runs_heatmap.png", "Stored ROME metrics across all bundle runs")
    render_overall_ranking(records, output_dir / "rome_success_overall_ranking.png")

    for spec in COLLECTIONS:
        group_records = [record for record in records if record.collection_slug == spec.slug]
        if not group_records:
            continue
        render_metric_heatmap(
            group_records,
            output_dir / f"rome_success_metrics_{spec.slug}_heatmap.png",
            f"Stored ROME metrics — {spec.label}",
        )
        if len(group_records) <= 20:
            render_metric_bars(
                group_records,
                output_dir / f"rome_success_metrics_{spec.slug}_bars.png",
                f"Stored ROME metrics — {spec.label}",
            )

    print(f"Wrote ROME success metric outputs to {output_dir}")
    print(f"Included {len(records)} run summaries from {bundle_root / 'data'}")


if __name__ == "__main__":
    main()

"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import csv
from pathlib import Path
from numbers import Real
from typing import Any, Iterable

from src.common.io import write_json
from src.results.naming import safe_slug


def _analysis_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in context.get("analyses", []):
        if not isinstance(payload, dict):
            continue
        run = payload.get("run", {})
        summary = payload.get("summary", {})
        rows.append(
            {
                "model": run.get("model"),
                "plan_id": run.get("plan_id"),
                "edit_method": run.get("edit_method"),
                "analysis": payload.get("producer"),
                "category": payload.get("category"),
                "status": payload.get("status"),
                "accuracy": float(summary.get("accuracy", 0.0) or 0.0),
                "correct": int(summary.get("correct", 0) or 0),
                "cases_evaluated": int(summary.get("cases_evaluated", 0) or 0),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> float:
    numeric = [_safe_float(value, default=float("nan")) for value in values]
    filtered = [value for value in numeric if value == value]
    return float(sum(filtered) / len(filtered)) if filtered else 0.0


def _case_metric(case: dict[str, Any], *names: str) -> Any:
    metrics = case.get("edit", {}).get("metrics", {})
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _execution_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in context.get("executions", []):
        if not isinstance(payload, dict):
            continue
        run = payload.get("run", {})
        edit_method = run.get("edit_method")
        if not edit_method:
            continue
        cases = [case for case in payload.get("cases", []) if isinstance(case, dict)]
        complete = [case for case in cases if case.get("status") == "complete"]
        success = [case for case in complete if bool(case.get("edit", {}).get("success"))]
        summary = payload.get("summary", {})
        rows.append(
            {
                "model": run.get("model"),
                "plan_id": run.get("plan_id"),
                "edit_method": edit_method,
                "status": payload.get("status"),
                "cases_total": int(summary.get("cases_total", len(cases)) or 0),
                "cases_complete": int(summary.get("cases_complete", len(complete)) or 0),
                "edit_success_count": int(summary.get("edit_success_count", len(success)) or 0),
                "edit_success_rate": _safe_float(
                    summary.get("edit_success_rate"),
                    len(success) / len(complete) if complete else 0.0,
                ),
                "overall_score": _mean(
                    _case_metric(case, "overall_score", "overall") for case in complete
                ),
                "efficacy_score": _mean(
                    _case_metric(case, "efficacy_score", "efficacy") for case in complete
                ),
                "paraphrase_score": _mean(
                    _case_metric(case, "paraphrase_score", "paraphrase") for case in complete
                ),
                "neighborhood_score": _mean(
                    _case_metric(case, "neighborhood_score", "neighborhood") for case in complete
                ),
            }
        )
    return rows


def _detection_case_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in context.get("analyses", []):
        if not isinstance(payload, dict) or payload.get("category") != "detection":
            continue
        run = payload.get("run", {})
        for case in payload.get("cases", []):
            if not isinstance(case, dict) or case.get("status") != "complete":
                continue
            accuracy = case.get("accuracy", {})
            target = accuracy.get("target_layer")
            detected = accuracy.get("detected_layer")
            distance = None
            if target is not None and detected is not None:
                distance = abs(int(detected) - int(target))
            rows.append(
                {
                    "model": run.get("model"),
                    "plan_id": run.get("plan_id"),
                    "edit_method": run.get("edit_method"),
                    "analysis": payload.get("producer"),
                    "config_hash": payload.get("config_hash"),
                    "case_id": case.get("case_id"),
                    "target_layer": "" if target is None else int(target),
                    "detected_layer": "" if detected is None else int(detected),
                    "layer_distance": "" if distance is None else int(distance),
                    "correct": accuracy.get("correct"),
                    "detection_score": _safe_float(case.get("data", {}).get("detection_score")),
                }
            )
    return rows


def _analysis_label(row: dict[str, Any]) -> str:
    return f"{safe_slug(str(row.get('model')))}\n{row.get('edit_method')}:{row.get('analysis')}"


def _numeric_layer_series(value: Any) -> dict[int, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    series: dict[int, float] = {}
    for key, item in value.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            return None
        if not isinstance(item, Real):
            return None
        series[layer] = float(item)
    return dict(sorted(series.items())) if series else None


def _find_series(data: dict[str, Any], *, prefix: str = "") -> list[tuple[str, dict[int, float]]]:
    preferred = (
        "combined_score",
        "raw_rank_score",
        "layer_scores",
        "isolation_scores",
        "rome_hybrid_scores",
        "sv_z_scores",
        "sv_ratio_scores",
    )
    found: list[tuple[str, dict[int, float]]] = []
    for key in preferred:
        series = _numeric_layer_series(data.get(key))
        if series:
            found.append((f"{prefix}{key}", series))
    for key, value in data.items():
        if isinstance(value, dict):
            found.extend(_find_series(value, prefix=f"{prefix}{key}."))
    deduped: list[tuple[str, dict[int, float]]] = []
    seen: set[str] = set()
    for name, series in found:
        if name not in seen:
            seen.add(name)
            deduped.append((name, series))
    return deduped


def render_run_summary(context: dict[str, Any]) -> list[str]:
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = write_json(
        output_dir / "analysis-summary.json",
        {"analyses": _analysis_rows(context)},
    )
    return [str(path)]


def render_detector(context: dict[str, Any]) -> list[str]:
    rows = [row for row in _analysis_rows(context) if row["category"] == "detection" and row["status"] == "complete"]
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "detection-summary.json", {"analyses": rows})
    outputs = [str(json_path)]
    if not rows:
        return outputs

    import matplotlib.pyplot as plt

    labels = [f"{safe_slug(row['model'])}\n{row['edit_method']}:{row['analysis']}" for row in rows]
    accuracies = [row["accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.25), 4.8))
    ax.bar(labels, accuracies, color="#1f77b4")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Layer detection accuracy")
    ax.set_title("Artifact-only detection results")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    graph_path = output_dir / "detection-accuracy.png"
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(graph_path))
    return outputs


def render_paper(context: dict[str, Any]) -> list[str]:
    rows = [row for row in _analysis_rows(context) if row["status"] == "complete"]
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        str(
            write_json(
                output_dir / "paper-analysis-data.json",
                {"analyses": rows},
            )
        )
    ]


def render_rome_success(context: dict[str, Any]) -> list[str]:
    rows = _execution_rows(context)
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "rome-success-metrics.json", {"executions": rows})
    csv_path = _write_csv(
        output_dir / "rome-success-metrics.csv",
        rows,
        [
            "model",
            "plan_id",
            "edit_method",
            "status",
            "cases_total",
            "cases_complete",
            "edit_success_count",
            "edit_success_rate",
            "overall_score",
            "efficacy_score",
            "paraphrase_score",
            "neighborhood_score",
        ],
    )
    outputs = [str(json_path), str(csv_path)]
    if not rows:
        return outputs

    import matplotlib.pyplot as plt

    labels = [f"{safe_slug(str(row['model']))}\n{row['edit_method']}" for row in rows]
    success_rates = [100.0 * _safe_float(row["edit_success_rate"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 1.1), 4.8))
    bars = ax.bar(labels, success_rates, color="#2a9d8f", width=0.58)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("ROME success rate (%)")
    ax.set_title("ROME Execution Success")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(104.0, bar.get_height() + 2.0),
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    graph_path = output_dir / "rome-success-rate.png"
    fig.savefig(graph_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(graph_path))
    return outputs


def render_detector_window(context: dict[str, Any]) -> list[str]:
    rows = _detection_case_rows(context)
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "detector-layer-window.json", {"cases": rows})
    csv_path = _write_csv(
        output_dir / "detector-layer-window.csv",
        rows,
        [
            "model",
            "plan_id",
            "edit_method",
            "analysis",
            "case_id",
            "target_layer",
            "detected_layer",
            "layer_distance",
            "correct",
            "detection_score",
        ],
    )
    outputs = [str(json_path), str(csv_path)]
    evaluated = [row for row in rows if row["layer_distance"] != ""]
    if not evaluated:
        return outputs

    import matplotlib.pyplot as plt

    grouped: dict[str, list[int]] = {}
    for row in evaluated:
        grouped.setdefault(_analysis_label(row), []).append(int(row["layer_distance"]))

    labels = list(grouped)
    exact = [100.0 * sum(distance == 0 for distance in vals) / len(vals) for vals in grouped.values()]
    within_one = [100.0 * sum(distance <= 1 for distance in vals) / len(vals) for vals in grouped.values()]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.2), 5.0))
    ax.bar([value - width / 2 for value in x], exact, width=width, label="Exact layer", color="#457b9d")
    ax.bar([value + width / 2 for value in x], within_one, width=width, label="Within +/-1 layer", color="#f4a261")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Cases (%)")
    ax.set_title("Detector Layer-Window Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    graph_path = output_dir / "detector-layer-window.png"
    fig.savefig(graph_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(graph_path))
    return outputs


def render_detector_signals(context: dict[str, Any]) -> list[str]:
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    outputs: list[str] = []

    import matplotlib.pyplot as plt

    for payload in context.get("analyses", []):
        if not isinstance(payload, dict) or payload.get("category") != "detection":
            continue
        run = payload.get("run", {})
        model = safe_slug(str(run.get("model", "model")))
        plan_id = safe_slug(str(run.get("plan_id", "plan")))
        method = safe_slug(str(run.get("edit_method", "method")))
        analysis = safe_slug(str(payload.get("producer", "analysis")))
        plotted = 0
        for case in payload.get("cases", []):
            if plotted >= 8 or not isinstance(case, dict) or case.get("status") != "complete":
                continue
            series_items = _find_series(case.get("data", {}))
            if not series_items:
                continue
            fig, ax = plt.subplots(figsize=(8.6, 4.8))
            for name, series in series_items[:6]:
                layers = list(series)
                values = [series[layer] for layer in layers]
                ax.plot(layers, values, marker="o", linewidth=1.6, label=name)
            accuracy = case.get("accuracy", {})
            target = accuracy.get("target_layer")
            detected = accuracy.get("detected_layer")
            if target not in (None, ""):
                ax.axvline(int(target), color="#2a9d8f", linestyle="--", linewidth=1.2, label="target")
            if detected not in (None, ""):
                ax.axvline(int(detected), color="#e63946", linestyle=":", linewidth=1.4, label="detected")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Signal")
            ax.set_title(f"{run.get('model')} {run.get('edit_method')} {payload.get('producer')} case {case.get('case_id')}")
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            case_slug = safe_slug(str(case.get("case_id", plotted)))
            graph_path = output_dir / model / plan_id / method / analysis / f"{case_slug}.png"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(graph_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(graph_path))
            index.append(
                {
                    "model": run.get("model"),
                    "plan_id": run.get("plan_id"),
                    "edit_method": run.get("edit_method"),
                    "analysis": payload.get("producer"),
                    "case_id": case.get("case_id"),
                    "output": str(graph_path.relative_to(output_dir)),
                    "series": [name for name, _ in series_items[:6]],
                }
            )
            plotted += 1

    index_path = write_json(output_dir / "detector-signal-profiles.json", {"outputs": index})
    return [str(index_path), *outputs]

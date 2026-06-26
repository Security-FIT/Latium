"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

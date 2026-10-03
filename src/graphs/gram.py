"""Gram reports from saved per-case artifacts; no model or detector reruns."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from src.evaluation.rome import summarize_rome_scores
from src.gram_experiment import digest, locked, write_json


def case_rows(executions, analyses):
    lookup = {}
    controls = {}
    for payload in analyses:
        run = payload["run"]
        if payload.get("producer") != "gram-localization":
            continue
        for case in payload.get("cases", []):
            if run.get("edit_method") in (None, "baseline"):
                controls.setdefault(run["model"], case.get("data", {}))
            else:
                key = (run["model"], run["plan_id"], str(case["case_id"]))
                if key in lookup:
                    raise ValueError(f"Multiple Gram analyses for {key}")
                lookup[key] = case
    rows = []
    seen = set()
    for execution in executions:
        run = execution["run"]
        if run.get("edit_method") != "rome":
            continue
        selection = execution.get("config", {}).get("case_selection", {})
        ids = [str(i) for i in selection.get("selected_case_ids", [])]
        positions = {case: selection.get("start_idx", 0) + i for i, case in enumerate(ids)}
        for case in execution.get("cases", []):
            key = (run["model"], str(case["case_id"]))
            if key in seen:
                raise ValueError(f"Duplicate cohort case across plans: {key}")
            seen.add(key)
            analysis = lookup.get((run["model"], run["plan_id"], key[1]), {})
            data = analysis.get("data", {})
            predicted = data.get("anomalous_layer") if analysis.get("status") == "complete" else None
            target = execution.get("summary", {}).get("target_layer")
            rows.append({"model": run["model"], "case_id": key[1], "plan_id": run["plan_id"],
                "cohort_hash": selection.get("manifest_hash"), "cohort_position": positions.get(key[1]),
                "status": case.get("status"), "analysis_status": analysis.get("status", "missing"),
                "error": case.get("error") or analysis.get("error"),
                "target_layer": target, "predicted_layer": predicted,
                "exact": predicted is not None and target is not None and int(predicted) == int(target),
                "rome_success": case.get("edit", {}).get("success", False),
                "metrics": case.get("edit", {}).get("metrics", {}),
                "layer_scores": data.get("localization", {}).get("layer_scores", {}),
                "source_revision": execution.get("metadata", {}).get("source_revision")})
    return rows, controls


def summarize(rows):
    selected = len(rows)
    evaluated = sum(r["predicted_layer"] is not None for r in rows)
    exact = sum(bool(r["exact"]) for r in rows)
    complete = [r for r in rows if r["status"] == "complete"]
    success = sum(bool(r["rome_success"]) for r in rows)
    return {"selected": selected, "completed": len(complete),
        "errors": sum(r["status"] == "error" for r in rows),
        "pending": sum(r["status"] == "pending" for r in rows),
        "evaluated": evaluated, "exact_count": exact,
        "gram_exact": exact / selected if selected else None,
        "gram_exact_evaluated": exact / evaluated if evaluated else None,
        "coverage": evaluated / selected if selected else None,
        "rome_success_count": success, "rome_success": success / selected if selected else None,
        **summarize_rome_scores([r["metrics"] for r in complete]),
        "rome_score_cases": len(complete),
        "detected_layers": dict(sorted(Counter(str(r["predicted_layer"]) for r in rows if r["predicted_layer"] is not None).items()))}


def write_report(output, rows, controls, *, graphs=True, formats=("png", "json"), expected_models=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = sorted(set(expected_models) | {r["model"] for r in rows})
    grouped = {m: [r for r in rows if r["model"] == m] for m in models}
    completed_ids = {m: {r["case_id"] for r in group if r["predicted_layer"] is not None} for m, group in grouped.items()}
    report = {"input_hash": digest([rows, controls]),
        "models": {m: summarize(group) for m, group in grouped.items()},
        "batches": {m: {plan: summarize([r for r in group if r["plan_id"] == plan]) for plan in sorted({r["plan_id"] for r in group})} for m, group in grouped.items()},
        "paired_coverage": {"same_evaluated_case_ids": len({tuple(sorted(ids)) for ids in completed_ids.values()}) <= 1,
                            "common_evaluated_count": len(set.intersection(*completed_ids.values())) if completed_ids else 0},
        "baseline_controls": controls,
        "baseline_note": "Baseline argmax is a control, not a binary edit-presence decision; excluded from all accuracy denominators."}
    outputs = []
    for name, payload in (("summary.json", report), ("cases.json", rows)):
        path = output / name
        write_json(path, payload)
        outputs.append(str(path))
    path = output / "cases.csv"
    fields = [k for k in rows[0] if k not in ("metrics", "layer_scores")] if rows else ["model", "case_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(str(path))
    if graphs and models:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(7, len(models) * .7), 4))
        x = list(range(len(models)))
        ax.bar([i-.2 for i in x], [report["models"][m]["gram_exact"] or 0 for m in models], .4, label="Gram exact / selected")
        ax.bar([i+.2 for i in x], [report["models"][m]["rome_success"] or 0 for m in models], .4, label="ROME success / selected")
        ax.set_xticks(x, models, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction")
        ax.legend()
        fig.tight_layout()
        for fmt in formats:
            if fmt in ("png", "pdf"):
                path = output / f"accuracy.{fmt}"; fig.savefig(path); outputs.append(str(path))
        plt.close(fig)
        fig, axes = plt.subplots(len(models), 2, figsize=(12, max(3, 3 * len(models))), squeeze=False)
        for i, model in enumerate(models):
            group = grouped[model]
            counts = Counter(r["predicted_layer"] for r in group if r["predicted_layer"] is not None)
            axes[i, 0].bar(list(counts), list(counts.values()))
            axes[i, 0].set_title(f"{model}: detected layers (n={len(group)})")
            axes[i, 0].set_ylabel("Cases")
            scores = defaultdict(list)
            for row in group:
                for layer, score in row["layer_scores"].items():
                    scores[int(layer)].append(float(score))
            layers = sorted(scores)
            axes[i, 1].plot(layers, [math.fsum(scores[l])/len(scores[l]) for l in layers], label="Edited mean, all evaluated cases")
            baseline = controls.get(model, {}).get("localization", {}).get("layer_scores", {})
            base_layers = sorted(int(l) for l in baseline)
            axes[i, 1].plot(base_layers, [baseline[str(l)] for l in base_layers], linestyle="--", label="Baseline control")
            axes[i, 1].set_title(f"{model}: Gram score profile")
            axes[i, 1].legend(fontsize=8)
            for axis in axes[i]:
                axis.set_xlabel("Layer")
        fig.tight_layout()
        for fmt in formats:
            if fmt in ("png", "pdf"):
                path = output / f"profiles.{fmt}"; fig.savefig(path); outputs.append(str(path))
        plt.close(fig)
    return outputs


def render_gram_report(context):
    rows, controls = case_rows(context.executions, (context.analyses or {}).get("gram-localization", ()))
    return write_report(context.output_dir, rows, controls, formats=(context.options or {}).get("formats", ("png", "json")))


def report_experiment(root, *, graphs=True):
    from src.results import RunArtifactReader
    root = Path(root)
    with locked(root):
        catalog = json.loads((root / "experiment.json").read_text())
        cohort = json.loads((root / "cases.json").read_text())
        if catalog["cohort_hash"] != cohort.get("manifest_hash"):
            raise ValueError("Experiment cohort changed")
        rows, controls = [], {}
        for model, entry in sorted(catalog["models"].items()):
            run_root = root / entry["run_root"]
            actual = []
            if (run_root / "manifest.json").exists():
                reader = RunArtifactReader(run_root)
                executions = [reader.load(r["artifact_id"]) for r in reader.records(kind="execution")]
                analyses = [reader.load(r["artifact_id"]) for r in reader.records(kind="analysis") if r.get("producer") == "gram-localization"]
                actual, baseline = case_rows(executions, analyses)
                for row in actual:
                    row["source_revision"] = reader.manifest.get("metadata", {}).get("source", {}).get("git_revision")
                    row["source_hash"] = catalog["setup"].get("source_hash")
                controls.update(baseline)
            by_id = {r["case_id"]: r for r in actual}
            used = set()
            for batch in entry["batches"].values():
                for position in range(batch["start"], batch["stop"]):
                    case_id = str(cohort["case_ids"][position])
                    row = by_id.get(case_id, {"model": model, "case_id": case_id, "plan_id": batch["plan_id"],
                        "cohort_hash": catalog["cohort_hash"], "cohort_position": position, "status": "pending",
                        "analysis_status": "missing", "error": None, "target_layer": None, "predicted_layer": None,
                        "exact": False, "rome_success": False, "metrics": {}, "layer_scores": {}, "source_revision": None})
                    if row["cohort_hash"] != catalog["cohort_hash"] or row["plan_id"] != batch["plan_id"]:
                        raise ValueError(f"Mismatched artifact selection for {model}/{case_id}")
                    rows.append(row); used.add(case_id)
            if set(by_id) - used:
                raise ValueError(f"Unregistered results in {model}")
        return write_report(root / "report", rows, controls, graphs=graphs, expected_models=catalog["models"])


def main():
    parser = argparse.ArgumentParser(description="Rebuild a cumulative Gram fleet report from saved artifacts")
    parser.add_argument("experiment_root")
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()
    for path in report_experiment(args.experiment_root, graphs=not args.no_graphs):
        print(path)

if __name__ == "__main__":
    main()

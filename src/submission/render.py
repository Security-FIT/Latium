"""Export the three frozen populations without rewriting archived artifacts."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from src.graphs.gram import case_rows, summarize, write_report
from src.gram_experiment import write_json
from src.submission.archive import campaign_root, sha256
from src.submission.artifacts import run_payloads

INPUTS = Path(__file__).resolve().parents[2] / "manifests/CSF-final"


def _source_map(root):
    return {r["source_path"]: root / r["path"] for r in json.loads((root / "INDEX.json").read_text()) if r.get("source_path")}


def selected_runs(campaign, root):
    if campaign == "classic":
        selection = json.loads((INPUTS / "classic-selections.json").read_text())
        return [(family, family, root / "raw/models" / family / "run", None) for family in selection]
    source_map = _source_map(root)
    if campaign == "finetune-base":
        rows = json.loads((INPUTS / "basecov-selected-checkpoints.json").read_text())
        selected = []
        for row in rows:
            execution = source_map[row["rome_execution"]]
            # Locate the run root through the exact indexed execution path.
            run = execution.parents[0]
            while run.name != "run":
                if run == root or run == run.parent:
                    raise ValueError(f"No run root for {execution}")
                run = run.parent
            selected.append((row["family"], row["model_id"], run, row))
        return selected
    rows = json.loads((INPUTS / "owncov-selected-checkpoints.json").read_text())
    selected = []
    for row in rows:
        if not row["evaluated"]:
            continue
        raw = row["artifact_manifest"].split("/finetune-covariance/", 1)[1]
        selected.append((row["family"], row["model_id"], (root / raw).parent, row))
    return selected


def layer_statistics(rows, *, classic=False):
    family_rows = defaultdict(list)
    for row in rows:
        family_rows[row["family"]].append(row)
    tables = []
    for family, cases in sorted(family_rows.items()):
        targets = {row["target_layer"] for row in cases}
        if len(targets) != 1:
            raise ValueError(f"Different configured target layers in {family}: {targets}")
        for state, key in (("baseline", "baseline_layer_scores"), ("edited", "layer_scores")):
            observations = cases[:1] if classic and state == "baseline" else cases
            layer_sets = {tuple(sorted(row[key], key=int)) for row in observations}
            if len(layer_sets) != 1 or not next(iter(layer_sets)):
                raise ValueError(f"Missing or inconsistent scored layers in {family}/{state}")
            for layer in sorted(observations[0][key], key=int):
                values = np.array([float(row[key][layer]) for row in observations])
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite GRAM score in {family}/{state}/{layer}")
                tables.append({"family": family, "state": state, "layer": int(layer), "target_layer": next(iter(targets)),
                               "n": len(values), "mean": float(np.mean(values)), "min": float(np.min(values)),
                               "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
                               "q75": float(np.quantile(values, .75)), "max": float(np.max(values)),
                               "std": float(np.std(values))})
    return tables


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_profiles(output, tables, *, campaign):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted({r["family"] for r in tables})
    figures = []
    for family in families:
        records = [r for r in tables if r["family"] == family]
        for scale in ("linear", "log"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for state, color, label in (("baseline", "black", "Unedited checkpoint"), ("edited", "#c42b35", "ROME edited")):
                group = [r for r in records if r["state"] == state]
                layers = [r["layer"] for r in group]
                if scale == "log" and any(r["min"] <= 0 for r in group):
                    # Keep actual zeros in the exported table. A log plot cannot
                    # display them; do not fabricate a replacement score.
                    label += " (nonpositive scores omitted on log axis)"
                ax.fill_between(layers, [r["min"] for r in group], [r["max"] for r in group], color=color, alpha=.08)
                ax.fill_between(layers, [r["q25"] for r in group], [r["q75"] for r in group], color=color, alpha=.15)
                ax.plot(layers, [r["mean"] for r in group], color=color, label=label)
            ax.axvline(records[0]["target_layer"], color="#245f9e", linestyle="--", label="Configured ROME layer")
            ax.set(xlabel="Layer (zero-based)", ylabel="Normal GRAM diagonal-relative score", title=f"{family} — {campaign}", yscale=scale)
            ax.legend(fontsize=8)
            fig.tight_layout()
            for fmt in ("png", "pdf", "svg"):
                path = output / f"{family}-{scale}.{fmt}"
                fig.savefig(path, dpi=180)
                figures.append(str(path))
            plt.close(fig)
    return figures


def render_campaign(campaign, archive, output):
    root = campaign_root(archive)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Rendering must write to a separate directory outside the archive")
    output.mkdir(parents=True, exist_ok=True)
    rows, causal, baseline_controls, evidence = [], [], {}, []
    classic_trace = {r["family"]: r for r in json.loads((root / "data/selected-causal-references.json").read_text())} if campaign == "classic" else {}
    for family, checkpoint, run_root, selection in selected_runs(campaign, root):
        reader, executions, analyses = run_payloads(run_root)
        run_rows, controls = case_rows(executions, analyses)
        if not run_rows:
            raise ValueError(f"Missing selected ROME cases in {run_root}")
        if selection and campaign == "finetune-base":
            expected = selection["rome_case_id"]
            run_rows = [r for r in run_rows if str(r["case_id"]) == str(expected)]
        if len(run_rows) != (1000 if campaign == "classic" else 1):
            raise ValueError(f"Selected case count changed for {checkpoint}: {len(run_rows)}")
        for row in run_rows:
            if row["status"] != "complete" or row["analysis_status"] != "complete" or row["baseline_status"] != "complete":
                raise ValueError(f"Incomplete selected workflow: {checkpoint}/{row['case_id']}")
            row.update(family=family, model_id=checkpoint, checkpoint_revision=selection.get("revision") if selection else None,
                       source_manifest=str(run_root / "manifest.json"))
        rows.extend(run_rows)
        baseline_controls.update(controls)
        causal.append({"family": family, "model_id": checkpoint,
                       "records": [dict(r) for r in reader.records() if "causal" in str(r.get("producer", "")) or "trace" in str(r.get("kind", ""))]})
        if campaign == "classic":
            reference = classic_trace.get(family)
            causal[-1].update(status="recorded" if reference else "unavailable",
                             reference=reference, reason=None if reference else "Missing comparable classic DeepSeek 7B base reference")
        evidence.append({"manifest": str(run_root / "manifest.json"), "sha256": sha256(run_root / "manifest.json")})
    expected = {"classic": 18000, "finetune-base": 790, "finetune-own": 744}[campaign]
    if len(rows) != expected:
        raise ValueError(f"Frozen campaign count changed: {len(rows)} != {expected}")
    return export_results(campaign, rows, baseline_controls, causal, evidence, output)


def export_results(campaign, rows, baseline_controls, causal, evidence, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    summaries = []
    for family, cases in sorted(grouped.items()):
        baseline = cases[:1] if campaign == "classic" else cases
        fp = sum(r["baseline_fp"] is True for r in baseline)
        n = sum(r["baseline_fp"] is not None for r in baseline)
        summaries.append({"family": family, **summarize(cases), "baseline_checkpoints": n,
                          "baseline_fp_count": fp, "baseline_fp_rate": fp / n if n else None})
    tables = layer_statistics(rows, classic=campaign == "classic")
    write_report(output / "native", rows, baseline_controls, graphs=False)
    write_json(output / "family-summary.json", summaries)
    _csv(output / "family-summary.csv", [{k: v for k, v in row.items() if k != "detected_layers"} for row in summaries])
    write_json(output / "layer-statistics.json", tables)
    _csv(output / "layer-statistics.csv", tables)
    write_json(output / "causal-statuses.json", causal)
    write_json(output / "source-manifests.json", evidence)
    figures = output / "profiles"
    figures.mkdir(exist_ok=True)
    render_profiles(figures, tables, campaign=campaign)
    if campaign == "finetune-own":
        failures = [r for r in json.loads((INPUTS / "owncov-selected-checkpoints.json").read_text()) if not r["evaluated"]]
        write_json(output / "download-failures.json", failures)
    return {"campaign": campaign, "cases": len(rows), "checkpoints": len({r["model_id"] for r in rows}),
            "source_runs": len(evidence),
            "rome_success": sum(r["rome_success"] for r in rows), "gram_exact": sum(r["exact"] for r in rows),
            "baseline_fp": sum(r["baseline_fp_count"] for r in summaries), "output": str(output)}


def render_new_campaign(campaign, runs, output):
    root, output = Path(runs).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Choose a render output outside the experiment directory")
    rows, controls, evidence = [], {}, []
    seen = set()
    receipts = sorted(root.rglob("reproduction.json"))
    if not receipts:
        raise ValueError(f"No completed reproduction receipts under {root}")
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt["campaign"] != campaign:
            raise ValueError(f"Different campaign in {path}")
        entry, batch = receipt["entry"], receipt["batch"]
        _, executions, analyses = run_payloads(receipt["run_root"])
        current, baseline = case_rows(executions, analyses)
        selected = set(batch.get("selected_case_ids", [str(entry.get("fact_case_id"))]))
        for row in current:
            if str(row["case_id"]) not in selected:
                continue
            key = (entry["model_id"], row["case_id"])
            if key in seen:
                raise ValueError(f"Duplicate independent edit: {key}")
            if row["status"] != "complete" or row["analysis_status"] != "complete":
                raise ValueError(f"Incomplete selected case: {key}")
            seen.add(key)
            revision = receipt.get("resolved_model_config", {}).get("checkpoint_revision") or entry.get("revision") or row.get("checkpoint_revision")
            row.update(family=entry["family"], model_id=entry["model_id"], checkpoint_revision=revision,
                       source_manifest=str(Path(receipt["run_root"]) / "manifest.json"))
            rows.append(row)
        for key, value in baseline.items():
            controls.setdefault(key, value)
        evidence.append({"receipt": str(path), "sha256": sha256(path), "validation_subset": receipt["validation_subset"]})
    expected = {"classic": 18000, "finetune-base": 790, "finetune-own": 744}[campaign]
    subset = any(e["validation_subset"] for e in evidence)
    if not subset and len(rows) != expected:
        raise ValueError(f"Incomplete full campaign: {len(rows)} != {expected}")
    result = export_results(campaign, rows, controls, [], evidence, output)
    write_json(output / "reproduction-scope.json", {"validation_subset": subset, "selected_cases": len(rows)})
    return {**result, "validation_subset": subset}


def render_saved_run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Choose a render output outside the source run")
    _, executions, analyses = run_payloads(root)
    rows, controls = case_rows(executions, analyses)
    return write_report(output, rows, controls, formats=("png", "pdf", "svg", "json"))

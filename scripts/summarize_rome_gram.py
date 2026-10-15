#!/usr/bin/env python3
"""Summarize raw Latium artifacts as CSV; standard library only, inputs read-only.

python summarize_rome_gram.py classic-gram-merged finetune-covariance > results.csv
Add --by-family to aggregate finetunes by their base model (paper tables).
ES/PS/NS/S and rates are percentages. Lift is mean edited target-layer score
divided by mean baseline target-layer score. Missing/failed cases raise an error.
All completed edits count, including ROME efficacy failures. Baselines count once
per run/plan, not once per edit. No saved reports, models or covariance are needed.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def runs(root):
    if (root / "data/checkpoints.json").exists():
        for row in read(root / "data/checkpoints.json"):
            saved = row["manifest"].replace("\\", "/")
            relative = "raw/" + saved.split("/raw/", 1)[1] if "/raw/" in saved else saved
            yield root / relative, row["family"], row["model_id"], row["revision"]
    else:
        manifests = sorted((root / "raw/models").glob("*/run/manifest.json"))
        if not manifests:
            raise ValueError(f"No merged classic runs or checkpoint index in {root}")
        for manifest in manifests:
            family = manifest.parents[1].name
            yield manifest, family, family, ""


def observations(manifest):
    executions, edited, baselines = [], {}, {}
    for record in read(manifest)["artifacts"].values():
        kind = record["kind"]
        if kind != "execution" and not (kind == "analysis" and record["producer"] == "gram-localization"):
            continue
        payload = read(manifest.parent / record["path"])
        plan, method = payload["run"]["plan_id"], payload["run"].get("edit_method")
        if kind == "execution" and method == "rome":
            executions.append(payload)
        elif kind == "analysis":
            for case in payload["cases"]:
                if case["status"] != "complete":
                    raise ValueError(f"Incomplete GRAM analysis: {manifest}")
                table, key = (edited, (plan, str(case["case_id"]))) if method == "rome" else (baselines, plan)
                if key in table:
                    raise ValueError(f"Duplicate GRAM analysis: {manifest}, {key}")
                table[key] = case["data"]
    seen = set()
    for execution in executions:
        plan = execution["run"]["plan_id"]
        target = int(execution["config"]["model"]["target_layer"])
        for case in execution["cases"]:
            case_id = str(case["case_id"])
            if case_id in seen or case["status"] != "complete":
                raise ValueError(f"Duplicate/incomplete edit: {manifest}, {case_id}")
            seen.add(case_id)
            after, before = edited[plan, case_id], baselines[plan]
            metrics = case["edit"]["metrics"]
            values = [float(metrics[k]) for k in ("efficacy_score", "paraphrase_score", "neighborhood_score")]
            a = float(after["localization"]["layer_scores"][str(target)])
            b = float(before["localization"]["layer_scores"][str(target)])
            if not all(math.isfinite(v) for v in [*values, a, b]) or b <= 0:
                raise ValueError(f"Invalid metrics/scores: {manifest}, {case_id}")
            yield dict(metrics=values, target=target, predicted=int(after["anomalous_layer"]),
                       baseline=int(before["anomalous_layer"]), after=a, before=b,
                       control=(str(manifest.resolve()), plan))
    if not seen:
        raise ValueError(f"No completed ROME edits: {manifest}")


def summarize(rows):
    n = len(rows)
    mean = lambda values: math.fsum(values) / n
    es, ps, ns = [100 * mean(r["metrics"][i] for r in rows) for i in range(3)]
    controls = {r["control"]: r for r in rows}
    fp = sum(r["baseline"] == r["target"] for r in controls.values())
    rome = sum(r["metrics"][0] >= 1 for r in rows)
    exact = sum(r["predicted"] == r["target"] for r in rows)
    return dict(n=n, rome_success=rome, rome_pct=100 * rome / n,
                ES=es, PS=ps, NS=ns, S=0 if min(es, ps, ns) == 0 else 3 / (1/es + 1/ps + 1/ns),
                gram_exact=exact, gram_pct=100 * exact / n,
                MAE=mean(abs(r["predicted"] - r["target"]) for r in rows),
                lift=mean(r["after"] for r in rows) / mean(r["before"] for r in rows),
                baseline_n=len(controls), baseline_fp=fp, baseline_fp_pct=100 * fp / len(controls),
                target_layers=";".join(map(str, sorted({r["target"] for r in rows}))),
                baseline_layers=";".join(map(str, sorted({r["baseline"] for r in rows}))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archives", type=Path, nargs="+", help="Extracted classic/finetune archive directories")
    parser.add_argument("--by-family", action="store_true", help="Group finetunes by base-model family")
    args = parser.parse_args()
    groups, seen = defaultdict(list), set()
    for root in args.archives:
        for manifest, family, model, revision in runs(root):
            manifest = manifest.resolve()
            if manifest in seen:
                raise ValueError(f"Run supplied twice: {manifest}")
            seen.add(manifest)
            key = (root.name, family, family if args.by_family else model, "" if args.by_family else revision)
            groups[key].extend(observations(manifest))
    result = [dict(zip(("campaign", "family", "model", "revision"), key), **summarize(rows))
              for key, rows in sorted(groups.items())]
    if not result:
        raise ValueError("No experiment results found")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(result[0]))
    writer.writeheader()
    for row in result:
        writer.writerow({k: round(v, 8) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()

"""Save corrected legacy tracing beside a fine-tuned checkpoint's Gram artifacts."""
import csv
import json
from pathlib import Path

from jobs.paper_fleet import run_logged, utc_now
from src.common.config import plain
from src.common.model_config import load_model_config
from src.results.artifacts import ArtifactWriter, build_artifact, config_hash


def trace_implementation_hash():
    root = Path(__file__).parents[1]
    return config_hash({name: root.joinpath(name).read_text() for name in
        ("src/causal_trace/legacy_fixed.py", "src/causal_trace/tokenization.py",
         "src/causal_trace/model_adapter.py", "src/common/loading.py", "jobs/finetuned_trace.py")})


def run_trace(args, model, root, cohort, position, stop=None):
    """Scan reserve facts with one model load, preserving per-fact artifacts."""
    root = Path(root)
    stop = position + 1 if stop is None else stop
    if not 0 <= position < stop <= cohort["count"]:
        raise ValueError("Invalid causal tracing fact range")
    writer = ArtifactWriter(root, run_id="run")
    implementation = trace_implementation_hash()

    def config_for(selected):
        return {"model": plain(load_model_config(model)), "manifest_hash": cohort["manifest_hash"],
                "position": selected, "noise_samples": 1, "seed": 42 + selected,
                "allow_article_prefix": True, "implementation": implementation}

    rejections = []
    for selected in range(position, stop):
        artifact_id = f"causal-kuba-fix/{model}/m{selected:04d}"
        current = writer.current(artifact_id, expected_config_hash=config_hash(config_for(selected)), inputs=[])
        if not current:
            break
        artifact = json.loads((root / current["path"]).read_text())
        if not all((root / path).is_file() for path in artifact["summary"]["outputs"]):
            break
        if artifact["status"] == "complete":
            return {"accepted_position": selected, "rejections": rejections, "error": None}
        rejections.append({"position": selected, "case_id": cohort["case_ids"][selected],
                           "error": artifact["error"]})
    else:
        return {"accepted_position": None, "rejections": rejections,
                "error": "; ".join(row["error"] for row in rejections)}

    # Cached rejections need not load the model again. The remaining range is
    # contiguous, so the legacy command can stop at the first suitable fact.
    start = position + len(rejections)
    directory = root / "causal-kuba-fix" / "scans" / f"m{start:04d}-{stop:04d}"
    command = [args.python, "-m", "src", "causal-kuba-fix", f"model={model}",
               f"command.legacy_trace.output_dir='{directory / 'raw'}'",
               "command.legacy_trace.num_valid_facts=1",
               f"command.legacy_trace.max_dataset_examples_to_scan={stop - start}",
               "command.legacy_trace.allow_article_prefix=true",
               f"command.legacy_trace.case_index_file='{Path(args.run_root) / 'cases.json'}'",
               f"command.legacy_trace.case_start={start}", f"command.legacy_trace.seed={42 + start}"]
    previous_summaries = set((directory / "raw").glob("*/summary.json"))
    failure = None
    try:
        run_logged(command, stage="causal-kuba-fix", model=model)
    except RuntimeError as exc:
        failure = exc
    summaries = sorted(set((directory / "raw").glob("*/summary.json")) - previous_summaries,
                       key=lambda p: p.stat().st_mtime_ns)
    if not summaries:
        raise failure or RuntimeError("Causal tracing did not save a summary")
    path = summaries[-1]
    summary = json.loads(path.read_text())
    if summary.get("case_selection") != {"manifest_hash": cohort["manifest_hash"], "start": start,
                                         "case_ids": cohort["case_ids"][start:stop]}:
        raise RuntimeError("Causal tracing used a different fact selection")
    complete = summary["status"] == "complete"
    scanned = summary["scanned_facts"]
    if not 0 < scanned <= stop - start or summary["valid_facts"] != int(complete):
        raise RuntimeError("Causal tracing saved inconsistent scan counts")
    if failure and complete:
        raise failure
    new_rejections = summary["rejections"]
    if len(new_rejections) != scanned - int(complete):
        raise RuntimeError("Causal tracing did not account for every scanned fact")
    for offset, row in enumerate(new_rejections):
        selected = start + offset
        if str(row["prompt_id"]) != str(cohort["case_ids"][selected]):
            raise RuntimeError("Causal tracing rejected a different fact")
        rejections.append({"position": selected, "case_id": cohort["case_ids"][selected], "error": row["reason"]})
    accepted = start + scanned - 1 if complete else None
    if complete:
        facts = list(path.parent.glob("fact_*.json"))
        if len(facts) != 1 or str(json.loads(facts[0].read_text())["prompt_id"]) != str(cohort["case_ids"][accepted]):
            raise RuntimeError("Causal tracing saved a different accepted fact")
        render_profile(path.parent)
    outputs = [str(p.relative_to(root)) for p in sorted(path.parent.iterdir()) if p.is_file()]
    scanned_cases = [{"case_id": cohort["case_ids"][selected], "status": "complete" if selected == accepted else "rejected"}
                     for selected in range(start, start + scanned)]
    scan = {"requested_start": start, "requested_stop": stop, "scanned_stop": start + scanned}
    for selected in range(start, start + scanned):
        status = "complete" if selected == accepted else "unavailable"
        error = None if selected == accepted else new_rejections[selected - start]["reason"]
        config = config_for(selected)
        artifact_id = f"causal-kuba-fix/{model}/m{selected:04d}"
        payload = build_artifact(artifact_id=artifact_id, kind="causal-trace", producer="causal-kuba-fix",
            run_id="run", model=model, plan_id=None, edit_method=None, status=status,
            config=config, config_hash=config_hash(config), inputs=[],
            cases=[scanned_cases[selected - start]],
            summary={**summary, "scan": scan, "selected_case_id": cohort["case_ids"][selected],
                     "selected_case_status": status, "accepted_position": accepted, "outputs": outputs}, created_at=utc_now(), error=error)
        writer.write(root / "causal-kuba-fix" / f"m{selected:04d}" / "artifact.json", payload, force=True)
    return {"accepted_position": accepted, "rejections": rejections,
            "error": None if complete else "; ".join(row["error"] for row in rejections)}


def render_profile(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with (directory / "profile.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    offsets = sorted({int(r["token_offset"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    values = np.full((len(offsets), len(layers)), np.nan)
    for row in rows:
        values[offsets.index(int(row["token_offset"])), layers.index(int(row["layer"]))] = float(row["mean_indirect_effect"])
    fig, ax = plt.subplots(figsize=(11, max(3, len(offsets) * .4)))
    shown = ax.imshow(values, aspect="auto", cmap="coolwarm")
    fact = json.loads(next(directory.glob("fact_*.json")).read_text())
    prefix = fact.get("answer_prefix", "")
    condition = f"prompt + predicted article {prefix!r}" if prefix else "original prompt"
    ax.set(xlabel="Layer (zero-based)", ylabel="Subject token offset",
           title=f"causal-kuba-fix: paired indirect effect\nCondition: {condition}")
    ticks = list(range(0, len(layers), max(1, len(layers) // 12)))
    ax.set_xticks(ticks, [layers[i] for i in ticks])
    ax.set_yticks(range(len(offsets)), offsets)
    fig.colorbar(shown, ax=ax, label="Restored − corrupted probability")
    fig.tight_layout()
    fig.savefig(directory / "profile.png", dpi=150)
    plt.close(fig)

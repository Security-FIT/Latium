"""Save corrected legacy tracing beside a fine-tuned checkpoint's Gram artifacts."""
import csv
import json
from pathlib import Path

from jobs.paper_fleet import run_logged, utc_now
from src.common.config import plain
from src.common.model_config import load_model_config
from src.results.artifacts import ArtifactWriter, build_artifact, config_hash


def run_trace(args, model, root, cohort, position):
    root = Path(root)
    writer = ArtifactWriter(root, run_id="run")
    config = {"model": plain(load_model_config(model)), "manifest_hash": cohort["manifest_hash"],
              "position": position, "noise_samples": 1, "seed": 42 + position,
              "implementation": config_hash({"source": Path(__file__).parents[1].joinpath("src/causal_trace/legacy_fixed.py").read_text()})}
    hashed = config_hash(config)
    artifact_id = f"causal-kuba-fix/{model}/m{position:04d}"
    current = writer.current(artifact_id, expected_config_hash=hashed, inputs=[])
    if current:
        artifact = json.loads((root / current["path"]).read_text())
        if all((root / p).is_file() for p in artifact["summary"]["outputs"]):
            return artifact["status"] == "complete", artifact["error"]
    directory = root / "causal-kuba-fix" / f"m{position:04d}"
    command = [args.python, "-m", "src", "causal-kuba-fix", f"model={model}",
               f"command.legacy_trace.output_dir='{directory / 'raw'}'",
               "command.legacy_trace.num_valid_facts=1", "command.legacy_trace.max_dataset_examples_to_scan=1",
               f"command.legacy_trace.case_index_file='{Path(args.run_root) / 'cases.json'}'",
               f"command.legacy_trace.case_start={position}", f"command.legacy_trace.seed={42 + position}"]
    failure = None
    try:
        run_logged(command, stage="causal-kuba-fix", model=model)
    except RuntimeError as exc:
        failure = exc
    summaries = sorted((directory / "raw").glob("*/summary.json"), key=lambda p: p.stat().st_mtime_ns)
    if not summaries:
        raise failure or RuntimeError("Causal tracing did not save a summary")
    path = summaries[-1]
    summary = json.loads(path.read_text())
    if summary.get("case_selection") != {"manifest_hash": cohort["manifest_hash"], "start": position,
                                         "case_ids": [cohort["case_ids"][position]]}:
        raise RuntimeError("Causal tracing used a different fact selection")
    complete = summary["status"] == "complete"
    if failure and complete:
        raise failure
    error = None if complete else "; ".join(r["reason"] for r in summary["rejections"])
    if complete:
        render_profile(path.parent)
    outputs = [str(p.relative_to(root)) for p in sorted(path.parent.iterdir()) if p.is_file()]
    payload = build_artifact(artifact_id=artifact_id, kind="causal-trace", producer="causal-kuba-fix",
        run_id="run", model=model, plan_id=None, edit_method=None,
        status="complete" if complete else "unavailable", config=config, config_hash=hashed,
        inputs=[], cases=[{"case_id": cohort["case_ids"][position], "status": summary["status"]}],
        summary={**summary, "outputs": outputs}, created_at=utc_now(), error=error)
    writer.write(directory / "artifact.json", payload, force=True)
    return complete, error


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
    ax.set(xlabel="Layer (zero-based)", ylabel="Subject token offset", title="causal-kuba-fix: paired indirect effect")
    ticks = list(range(0, len(layers), max(1, len(layers) // 12)))
    ax.set_xticks(ticks, [layers[i] for i in ticks])
    ax.set_yticks(range(len(offsets)), offsets)
    fig.colorbar(shown, ax=ax, label="Restored − corrupted probability")
    fig.tight_layout()
    fig.savefig(directory / "profile.png", dpi=150)
    plt.close(fig)

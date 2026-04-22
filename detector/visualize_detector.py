# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ROME Layer Detector — Visualization
#
# This notebook visualizes post-hoc layer detector results across model families:
# - Composite detector plots for non-GPT architectures
# - GPT detector plots for GPT-2 / GPT-J architectures
# - Shared summary accuracy and method breakdown charts across all models
# - Composite-only top-K sensitivity analysis

# %% Imports & setup
import sys, json, os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

# Make imports/data paths work whether the notebook is launched from the
# repository root or from detector/.
cwd = Path.cwd().resolve()
if (cwd / "detector" / "composite_detector_v2.py").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "detector" / "composite_detector_v2.py").exists():
    PROJECT_ROOT = cwd.parent
else:
    PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.pop("composite_detector_v2", None)
sys.modules.pop("detector.composite_detector_v2", None)
sys.modules.pop("gpt_detector", None)
sys.modules.pop("detector.gpt_detector", None)

from detector.composite_detector_v2 import (
    local_zscore, _curvature,
    process_file as composite_process_file,
    plot_signal_profiles as composite_plot_signal_profiles,
    plot_average_signal_profiles as composite_plot_average_signal_profiles,
    plot_summary_table, plot_method_breakdown,
    aggregate_results_by_model,
)
from detector.gpt_detector import (
    process_file as gpt_process_file,
    plot_average_signals as gpt_plot_average_signals,
    plot_signals as gpt_plot_signals,
)

# Notebook controls. Defaults process every structural run we can find.
# Examples:
#   MODEL_FILTER = []
#   MODEL_FILTER = ["qwen3-4b"]
#   MAX_RUNS = 3
#   MAX_TESTS_PER_RUN = 5
def _env_csv(name):
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name, default=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw or raw.lower() == "none":
        return None
    return int(raw)


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_search_roots():
    raw_roots = _env_csv("VISUALIZE_SEARCH_ROOTS")
    if not raw_roots:
        return [
            (PROJECT_ROOT / "pipeline_out", True),
            (PROJECT_ROOT / "analysis_out", True),
            (PROJECT_ROOT / "ultrasupertest", False),
            (PROJECT_ROOT / "results_n5", False),
        ]

    resolved = []
    for raw_root in raw_roots:
        path = Path(raw_root)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved.append((path.resolve(), True))
    return resolved


MODEL_FILTER = _env_csv("VISUALIZE_MODEL_FILTER")
MAX_RUNS = _env_int("VISUALIZE_MAX_RUNS")
MAX_TESTS_PER_RUN = _env_int("VISUALIZE_MAX_TESTS_PER_RUN", 3)
RUN_TOPK_ANALYSIS = _env_bool("VISUALIZE_RUN_TOPK_ANALYSIS", False)
TRIM = _env_int("VISUALIZE_TRIM", 2)
GPT_TRIM = _env_int("VISUALIZE_GPT_TRIM", 5)
GRAPH_DIR = Path(
    os.getenv("VISUALIZE_GRAPH_DIR", str(PROJECT_ROOT / "detector" / "graphs"))
).resolve()
SEARCH_ROOTS = _resolve_search_roots()


def _matches_model(path):
    if not MODEL_FILTER:
        return True
    text = Path(path).stem.lower()
    return any(model.lower() in text for model in MODEL_FILTER)


def _is_gpt_structural_path(path: Path) -> bool:
    text = path.stem.lower()
    return "gpt-j" in text or "gpt2-" in text


def _process_structural_file(path: Path):
    if _is_gpt_structural_path(path):
        result = gpt_process_file(path, trim=GPT_TRIM, max_tests=MAX_TESTS_PER_RUN)
        result["detector_family"] = "gpt"
        return result

    result = composite_process_file(path, trim=TRIM, max_tests=MAX_TESTS_PER_RUN)
    result["detector_family"] = "composite"
    return result


def discover_structural_json_files():
    candidates = []
    for root, recursive in SEARCH_ROOTS:
        if not root.exists():
            continue
        if root.is_file() and root.name.startswith("rome_structural_") and root.suffix == ".json":
            candidates.append(root)
            continue
        iterator = root.rglob("rome_structural_*.json") if recursive else root.glob("rome_structural_*.json")
        candidates.extend(iterator)

    seen = set()
    filtered = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        resolved = str(path.resolve())
        if resolved in seen or not _matches_model(path):
            continue
        seen.add(resolved)
        filtered.append(path)
    return filtered


def _aggregate_scope_label(results):
    total_tests = sum(int(result.get("n_tests", 0) or 0) for result in results)
    run_count = len(results)
    run_word = "run" if run_count == 1 else "runs"
    test_word = "test" if total_tests == 1 else "tests"
    return f"{total_tests} {test_word} from {run_count} {run_word}"

# %% [markdown]
# ## 1. Run detector on all models

# %%
json_files = discover_structural_json_files()
if MAX_RUNS is not None:
    json_files = json_files[:MAX_RUNS]
print(f"Found {len(json_files)} structural JSON files")

all_results = []
for i, jf in enumerate(json_files, 1):
    print(f"[{i}/{len(json_files)}] Processing {Path(jf).name}")
    result = _process_structural_file(jf)
    all_results.append(result)

total_c = sum(r["correct"] for r in all_results)
total_t = sum(r["n_tests"] for r in all_results)
if total_t:
    print(f"\nOverall: {total_c}/{total_t} ({total_c/total_t:.1%})")
else:
    print("\nOverall: 0/0 (no matching tests)")

# Print per-run summary
for r in all_results:
    short = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
    mc = ", ".join(f"{k}:{v}" for k, v in
                   sorted(r["method_counts"].items(), key=lambda x: -x[1]))
    flag = "✓" if r["accuracy"] >= 0.9 else ("✗" if r["accuracy"] < 0.5 else "~")
    print(f"  {short:<20s} {r['run_label']:<34s} L{r['target_layer']:>2d}  "
          f"{r['correct']:>2}/{r['n_tests']:<2d} {r['accuracy']:>4.0%}  {mc} {flag}")

results_by_model = defaultdict(list)
for result in all_results:
    results_by_model[result["model"]].append(result)

model_average_results = aggregate_results_by_model(all_results)
if model_average_results:
    print("\nPer-model aggregate across tests and runs:")
    for r in model_average_results:
        short = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
        weighted = r.get("weighted_accuracy", 0.0)
        print(
            f"  {short:<20s} tests={r['n_tests']:<2d} runs={r['n_runs']:<2d} "
            f"mean_acc={r['accuracy']:.1%} std={r['accuracy_std']:.1%} weighted={weighted:.1%}"
        )

# %% [markdown]
# ## 2. Signal Profiles (all runs + average per model)
#
# Each plot shows 4 signals across layers:
# - one averaged panel across all runs for a model
# - one panel for each individual run
# - per-run panels are averaged across valid tests in that run
# - **SG (raw)**: spectral gap = σ₁/σ₂
# - **TE (lz5)**: |local z-score| of top1_energy with window=5
# - **SG (lz5)**: |local z-score| of spectral_gap with window=5
# - **SG (lz7)**: |local z-score| of spectral_gap with window=7

# %%
graph_dir = GRAPH_DIR
run_graph_dir = graph_dir / "runs"
avg_graph_dir = graph_dir / "averages"

for model, runs in sorted(results_by_model.items()):
    short = model.split("/")[-1] if "/" in model else model
    safe = short.replace("/", "_").replace(" ", "_")
    display(Markdown(f"### {short}"))

    detector_family = runs[0].get("detector_family", "composite")
    if detector_family == "gpt":
        gpt_plot_average_signals(runs, output_dir=avg_graph_dir)
        avg_img_path = avg_graph_dir / f"signals_{safe}_average.png"
        if avg_img_path.exists():
            display(Markdown(f"**Average across {_aggregate_scope_label(runs)}**"))
            display(Image(filename=str(avg_img_path), width=900))
        else:
            print(f"  [no GPT average graph for {short}]")
        for stale_gpt_img in run_graph_dir.glob(f"gpt_signals_{safe}_*.png"):
            stale_gpt_img.unlink()
        for r in runs:
            stale_run_img = run_graph_dir / f"signals_{safe}_{r['run_slug']}.png"
            if stale_run_img.exists():
                stale_run_img.unlink()
            output_name = f"gpt_signals_{safe}_{r['run_slug']}_t{int(r.get('trim', GPT_TRIM))}.png"
            gpt_plot_signals(
                Path(r["path"]),
                trim=int(r.get("trim", GPT_TRIM)),
                output_dir=run_graph_dir,
                output_name=output_name,
            )
            img_path = run_graph_dir / output_name
            if img_path.exists():
                display(Markdown(f"**{r['run_label']}**"))
                display(Image(filename=str(img_path), width=900))
            else:
                print(f"  [no GPT graph for {short} / {r['run_label']}]")
        continue

    composite_plot_average_signal_profiles(runs, output_dir=avg_graph_dir)
    avg_img_path = avg_graph_dir / f"signals_{safe}_average.png"
    if avg_img_path.exists():
        display(Markdown(f"**Average across {_aggregate_scope_label(runs)}**"))
        display(Image(filename=str(avg_img_path), width=900))
    else:
        print(f"  [no average graph for {short}]")

    for r in runs:
        composite_plot_signal_profiles(r, output_dir=run_graph_dir)
        img_path = run_graph_dir / f"signals_{safe}_{r['run_slug']}.png"
        if img_path.exists():
            display(Markdown(f"**{r['run_label']}**"))
            display(Image(filename=str(img_path), width=900))
        else:
            print(f"  [no graph for {short} / {r['run_label']}]")

# %% [markdown]
# ## 3. Summary Accuracy

# %%
plot_summary_table(
    all_results,
    output_dir=graph_dir,
    output_name="summary_accuracy_all_runs.png",
    title="Post-hoc Layer Detector — Accuracy by Run",
)
plot_summary_table(
    model_average_results,
    output_dir=graph_dir,
    output_name="summary_accuracy_model_average.png",
    title="Post-hoc Layer Detector — Mean Run Accuracy by Model",
)

# %%
for label, img in [
    ("All runs", graph_dir / "summary_accuracy_all_runs.png"),
    ("Average per model", graph_dir / "summary_accuracy_model_average.png"),
]:
    if img.exists():
        display(Markdown(f"### {label}"))
        display(Image(filename=str(img), width=900))

# %% [markdown]
# ## 4. Method Breakdown

# %%
plot_method_breakdown(
    all_results,
    output_dir=graph_dir,
    output_name="method_breakdown_all_runs.png",
    title="Detection Method Breakdown by Run",
)
plot_method_breakdown(
    model_average_results,
    output_dir=graph_dir,
    output_name="method_breakdown_model_average.png",
    title="Detection Method Breakdown Aggregated Across Runs",
)

# %%
for label, img in [
    ("All runs", graph_dir / "method_breakdown_all_runs.png"),
    ("Average per model", graph_dir / "method_breakdown_model_average.png"),
]:
    if img.exists():
        display(Markdown(f"### {label}"))
        display(Image(filename=str(img), width=900))

# %% [markdown]
# ## 5. Top-K Sensitivity Analysis
#
# Instead of always picking the single highest peak (top-1), what if we consider
# the top-K peaks per signal and check if the correct layer appears among them?
#
# This measures how "close" each signal gets even when the top-1 is wrong.

# %%
def topk_accuracy(all_results, k_values=[1, 2, 3, 5], trim=TRIM,
                  max_tests_per_run=MAX_TESTS_PER_RUN):
    """For each signal and each k, compute: in how many tests does the
    true target appear in the top-k peaks of that signal?"""
    signals = ["spectral_gap", "top1_energy", "norm_cv", "effective_rank",
               "row_alignment", "spectral_entropy"]
    transforms = {
        "raw": lambda vals: vals,
        "lz5": lambda vals: np.abs(local_zscore(vals, 5)),
        "lz7": lambda vals: np.abs(local_zscore(vals, 7)),
        "curv": lambda vals: _curvature(vals),
    }

    results = {}
    for sig in signals:
        for tname, tfn in transforms.items():
            key = f"{sig[:6]}_{tname}"
            results[key] = {k: 0 for k in k_values}
            results[key]["total"] = 0

    for r in all_results:
        path = Path(r["path"])
        target = r["target_layer"]
        with open(path) as f:
            data = json.load(f)

        used_tests = 0
        for t in data.get("tests", []):
            if t.get("error") or not t.get("rome", {}).get("success", True):
                continue
            if max_tests_per_run is not None and used_tests >= max_tests_per_run:
                break
            lf = t["blind_detection"]["layer_features"]
            layers = sorted(lf.keys(), key=int)
            n = len(layers)
            lo, hi = trim, n - trim
            if hi <= lo:
                continue
            eval_layers = [int(l) for l in layers[lo:hi]]

            for sig in signals:
                full = np.array([lf[l][sig] for l in layers])
                for tname, tfn in transforms.items():
                    vals = tfn(full)[lo:hi]
                    key = f"{sig[:6]}_{tname}"
                    results[key]["total"] += 1
                    # Top-k indices
                    ranked = np.argsort(vals)[::-1]
                    for k in k_values:
                        topk_layers = [eval_layers[i] for i in ranked[:k]]
                        if target in topk_layers:
                            results[key][k] += 1
            used_tests += 1

    return results

k_values = [1, 2, 3, 5]
composite_results = [r for r in all_results if r.get("detector_family") != "gpt"]
if RUN_TOPK_ANALYSIS:
    topk = topk_accuracy(composite_results, k_values)
else:
    topk = {}
    print("Top-K analysis skipped. Set RUN_TOPK_ANALYSIS = True to run it.")

# Display as table
if topk:
    print(f"{'Signal+Transform':<22s}", end="")
    for k in k_values:
        print(f"  {'top-'+str(k):>8s}", end="")
    print(f"  {'total':>6s}")
    print("-" * 60)

    for key in sorted(topk.keys()):
        total = topk[key]["total"]
        if total == 0:
            continue
        print(f"  {key:<20s}", end="")
        for k in k_values:
            acc = topk[key][k] / total
            print(f"  {acc:>7.1%}", end="")
        print(f"  {total:>6d}")

# %% [markdown]
# ### Top-K for GPT models only
#
# GPT-family models use `gpt_detector.py`, which votes on `norm_cv` transforms.
# The composite signal top-K analysis below is therefore intentionally skipped
# for GPT runs.

# %%
topk_gpt = {}
if RUN_TOPK_ANALYSIS:
    print("GPT top-K analysis skipped: GPT-family runs use a different detector.")

# %% [markdown]
# ### Top-K Accuracy Heatmap

# %%
if RUN_TOPK_ANALYSIS and (topk or topk_gpt):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (title, data) in zip(axes, [("All Models", topk),
                                          ("GPT Models Only", topk_gpt)]):
        if not data:
            ax.set_visible(False)
            continue

        keys = sorted(k for k in data.keys() if data[k].get("total", 0) > 0)
        if not keys:
            ax.set_visible(False)
            continue
        matrix = np.array([[data[k][kv] / data[k]["total"]
                            for kv in k_values] for k in keys])

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f"top-{k}" for k in k_values])
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_title(title, fontweight="bold")

        for i in range(len(keys)):
            for j in range(len(k_values)):
                ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center",
                        fontsize=7, color="black" if matrix[i,j] > 0.4 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    plt.tight_layout()
    graph_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(graph_dir / "topk_heatmap.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {graph_dir / 'topk_heatmap.png'}")
else:
    print("Top-K heatmap skipped.")

# %% [markdown]
# ## 6. Failure Analysis

# %%
failures = [(r, t) for r in all_results for t in r["results"] if not t["hit"]]
if failures:
    print(f"Total failures: {len(failures)}\n")
    for r, t in failures:
        short = r["model"].split("/")[-1]
        info = t.get("info", {})
        override = info.get("v5_override", {})
        override_str = f" (override from L{override['from']} {override.get('v5_method','')})" if override else ""
        print(f"  {short:<20s} {r['run_label']:<34s} test#{t['test_idx']:2d}  "
              f"target=L{t['target']:2d}  detected=L{t['detected']:2d}  "
              f"method={t['method']}{override_str}")
else:
    print("No failures!")

# %% [markdown]
# ## 7. Per-Model Detail Tables

# %%
for r in all_results:
    short = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
    if r["accuracy"] == 1.0:
        continue  # Skip perfect models
    print(f"\n{'='*60}")
    print(f"{short}  —  {r['run_label']}  —  {r['correct']}/{r['n_tests']} ({r['accuracy']:.0%})")
    print(f"{'='*60}")
    for t in r["results"]:
        mark = "✓" if t["hit"] else "✗"
        print(f"  {mark} test#{t['test_idx']:2d}  "
              f"target=L{t['target']:2d}  detected=L{t['detected']:2d}  "
              f"method={t['method']}")

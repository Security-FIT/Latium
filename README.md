# Latium Framework

## Quick Start

Use `pipeline.sh` for the standard end-to-end workflow:

```bash
# On the GPU host: ROME-only smoke benchmark on one model
bash pipeline.sh --models gpt2-medium --n 1 --compute-cov

# On the GPU host: structural benchmark + detector/new graph post-processing
bash pipeline.sh --structural --models gpt2-medium --n 1 --compute-cov

# Optional: orchestrate the same run over SSH from another machine
bash pipeline.sh --remote user@gpu-host --structural --models gpt2-medium --n 1

# Show all available pipeline options
bash pipeline.sh --help
```

`pipeline.sh` runs locally by default. The default mode runs the ROME-only
benchmark via `rome_benchmark.py`; `--structural` switches to the structural
benchmark plus detector and new graph post-processing. Add `--remote <host>` when
you want the same workflow launched over SSH/tmux from another machine.
Structural pipeline graphs are written under `pipeline_out/<run>/graphs/`.

## Running ROME

ROME (and related commands) is driven via the Hydra-based CLI in `src/cli.py`.

**Single intervention:**
```bash
python -m src.cli command=rome model=gpt2-medium
```

**Batch evaluation:**
```bash
python -m src.cli command=batch-rome model=gpt2-medium
```

**Compute second-moment statistics** (required before running ROME on a new model):
```bash
python -m src.cli command=second-moment model=gpt2-medium
```

**Cluster-local smoke on a GPU host:**
```bash
# First cold run: downloads model/datasets locally and builds second moments
python -m src.cli command=second-moment model=gpt2-medium

# Single local edit smoke test
python -m src.cli command=rome model=gpt2-medium

# Local ROME-only benchmark (same benchmark family used by pipeline.sh default mode)
python rome_benchmark.py --models gpt2-medium --n-tests 1 --start-idx 0 --output-dir ./analysis_out_local_rome
```

Notes:
- You can use other models, e.g. gpt2-xl, qwen3-4b etc (scroll down for full model list)
- `python -m src.cli command=rome ...` does **not** auto-compute missing second moments unless `ROME_ALLOW_SECOND_MOMENT_AUTOCOMPUTE=1` is set.
- By default, model downloads are cached under `../models`, dataset downloads under `../datasets`, and computed covariance files under `./second_moment_stats`.
- A true cold first run on a GPU host can stay quiet for several minutes while `command=second-moment` downloads assets and builds the covariance file.

The default config is at `src/config/config.yaml`. Override any value on the command line using Hydra syntax (e.g. `model=gpt2-large`).

Alternatively, use the console fallback (no Hydra overhead):
```bash
python -m src.cli --console rome --config src/config/config.yaml
```

---

## Running Causal Trace

```bash
python -m src.cli command=causal-trace model=gpt2-medium
```

To inspect the computed noise multiplier without running a full trace:
```bash
python -m src.cli command=compute-multiplier model=gpt2-medium
```

---

## Remote Covariance Pipeline

`covariance_a100_remote.sh` computes second-moment statistics on a remote GPU node (e.g. A100) and pulls the resulting artifacts back locally.

```bash
# Run with default models (deepseek-7b-base, granite4-micro, llama2-7b, mistral-7b-v0.1, mistral-7b-v0.3):
./covariance_a100_remote.sh user@gpu-host

# Override models:
MODEL_KEYS="gpt2-xl gpt-j-6b" ./covariance_a100_remote.sh user@gpu-host /path/to/Latium optim latium

# Arguments: <user@host> [remote_repo_path] [remote_branch] [conda_env]
```

The script syncs model configs and `src/rome/common.py` to the remote, runs covariance computation per model, and downloads the `.pt` artifacts into `second_moment_stats/`.

---

## Layer Selection Heuristic

`src/causal_trace/layer_heuristic.py` recommends the best MLP layer for ROME edits using multiple signals (causal trace, weight norms, spectral gap, architectural prior).

```bash
# CSV-only (no GPU needed):
python -m src.causal_trace.layer_heuristic \
    --csvs analysis_out/causal_trace_deepseek*.csv \
    --num-layers 30

# Full analysis (GPU + model):
python -m src.causal_trace.layer_heuristic \
    --model deepseek-ai/deepseek-llm-7b-base \
    --layer-template 'model.layers.{}.mlp.down_proj' \
    --num-layers 30 \
    --csvs analysis_out/causal_trace_deepseek*.csv
```

---

## Running the Structural Benchmark

`structural_benchmark.py` applies ROME edits across a dataset and evaluates all structural detectors (MSD, blind MSD, spectral, IPR) on the modified weights. Results are written as JSON to `analysis_out/`.

For the lightweight payload used by the post-hoc detector and
`paper_graphs.ipynb`, run `structural_benchmark.py --posthoc-only ...`
or `structural_benchmark.py --paper ...`
(`--analysis-profile paper` remains the underlying profile name).

```bash
python structural_benchmark.py \
    --model gpt2-large \
    --n-tests 30 \
    --start-idx 0 \
    --output-dir ./analysis_out \
    --spectral-top-k 50 \
    --trim-first-layers 2 \
    --trim-last-layers 2 \
    --spectral-neighbor-layers 1
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt2-large` | Model name (must match a config in `src/config/model/`) |
| `--n-tests` | `30` | Number of ROME edits to benchmark |
| `--start-idx` | `0` | Starting index in the facts dataset |
| `--output-dir` | `./analysis_out` | Directory for JSON result files |
| `--spectral-top-k` | `50` | Top-K singular values used by the spectral detector |
| `--trim-first-layers` | `2` | Layers to exclude from the head of the model |
| `--trim-last-layers` | `2` | Layers to exclude from the tail of the model |
| `--n-prompts` | auto | Number of ROME prefix prompts (scales with model size if omitted) |

---

## Detection Documentation

Detailed documentation for the detection methods is in the `docs/` directory:

- `docs/structural-docs.md` - structural detector metrics (L2 discrepancy, relative discrepancy, directional coherence, MSD, IPR, etc.)
- `docs/spectral-docs.md` - spectral detector signals and the mathematics behind singular-value z-scores and ratio scores

---

## Rebuild Final Paper Graphs

With a `final_n500_bundle/` artifact present at the repo root:

```bash
bash scripts/bundle_graphs/run_all_graphs.sh --bundle-root final_n500_bundle
```

From the directory containing a downloaded bundle:

```bash
bash final_n500_bundle/scripts_for_graphs/run_all_graphs.sh
```

The runner rebuilds per-model paper graphs, bundle summary graphs, windowed-detector reports, cohort figures, artifact grids, and refreshes the bundle indices.

---

## Models roadmap
---

| Supported Models  | Causal Trace       | Weight intervention | Mean ES (n=500) | Notes |
|-------------------|--------------------|---------------------|---------------------|-------|
| gpt2-medium       | :heavy_check_mark: | :heavy_check_mark:  | 0.988               | works |
| gpt2-large        | :heavy_check_mark: | :heavy_check_mark:  | 0.986               | works |
| gpt2-xl           | :heavy_check_mark: | :heavy_check_mark:  | 0.986               | works |
| gpt-j-6b          | :heavy_check_mark: | :heavy_check_mark:  | 0.996               | works |
| qwen3-0.6b        | :heavy_check_mark: | :heavy_check_mark:  |                     |       |
| qwen3-1.7b        | :heavy_check_mark: | :heavy_check_mark:  |                     |       |
| qwen3-4b          | :heavy_check_mark: | :heavy_check_mark:  | 0.992               |       |
| qwen3-8b          | :heavy_check_mark: | :heavy_check_mark:  | 1.000               |       |
| granite4-micro    | :heavy_check_mark: | :heavy_check_mark:  | 0.978               | Weird architecture |
| mistral-7b-v0.1   | :heavy_check_mark: | :heavy_check_mark:  | 0.948               |       |
| mistral-7b-v0.3   | :heavy_check_mark: | :heavy_check_mark:  | 0.934               |       |
| llama2-7b         | :heavy_check_mark: | :heavy_check_mark:  | 0.614               |Weird architecture|
| falcon-7b         | :heavy_check_mark: | :heavy_check_mark:  | 0.976               |       |
| opt-6.7b          | :heavy_check_mark: | :heavy_check_mark:  | 0.978               |       |
| deepseek-7b-base  | :heavy_check_mark: | :heavy_check_mark:  | 0.976               |       |
| llama3            |                    |                     |                     | planned |
| gpt-neo           |                    |                     |                     | planned |
| qwen2.5           |                    |                     |                     | planned |
| baichuan          |                    |                     |                     | planned |
| chatglm           |                    |                     |                     | planned |
| t5                |                    |                     |                     | planned |

---

## Pipeline Script

`pipeline.sh` runs either the ROME-only benchmark or the structural benchmark.
Run it directly on a GPU host after cloning the repo, or pass `--remote <host>`
to sync the repo and launch the selected mode over SSH/tmux.

```bash
# Local ROME-only smoke benchmark on one model
bash pipeline.sh --models gpt2-medium --n 1 --compute-cov

# Local structural run with detector/new graph processing
bash pipeline.sh --structural --models gpt2-medium --n 1 --compute-cov

# Local structural run, then rebuild final-bundle paper graphs if the bundle is present
bash pipeline.sh --structural --bundle-graphs --bundle-root final_n500_bundle

# Remote run with env setup
bash pipeline.sh --remote ubuntu@132.145.129.234 --setup-env

# Remote structural benchmark, N=1 smoke test
bash pipeline.sh --remote user@gpu-host --models gpt2-medium --n 1 --structural

# Compute covariance first, then benchmark
bash pipeline.sh --compute-cov --n 10

# Specific models only
bash pipeline.sh --models gpt2-xl mistral-7b-v0.1 --n 5
```

For structural runs, the current renderer outputs are under
`pipeline_out/<run>/graphs/`:

- `rome_success_metrics/` - stored ROME metric tables, heatmap, and bars
- `detector_stacked_variants/` - stacked SG/TE detector signal panels
- `detector_layer_window/` - strict and +/- window detector-layer scoring

| Flag | Default | Description |
|---|---|---|
| `--compute-cov` | off | Compute covariance matrices (otherwise uses existing) |
| `--n <int>` | 50 | Number of test edits per model |
| `--structural` | off | Run the structural benchmark and render the new per-run graph set under `pipeline_out/<run>/graphs/` |
| `--bundle-graphs` | off | After a structural run, rebuild graphs from `--bundle-root` |
| `--bundle-root <path>` | `final_n500_bundle` | Final bundle root used by `--bundle-graphs` |
| `--setup-env` | off | Set up conda env + deps on the remote host |
| `--remote <host>` | local current host | SSH target for remote execution |
| `--models <m1 ..>` | final paper model set | Override model list |
| `--output-dir <path>` | `./pipeline_out` | Output directory |

---

## Prefix/Template Spectral Variability Test

`prefixtest/experiment.py` measures how sensitive the spectral
detection pipeline is to the prefix/template used during the ROME edit.  A
single fact is edited 20 times under different prefix strategies (self-generated,
template-based, external) while all other parameters remain fixed.  The spectral
detector runs on each result, producing per-layer signal curves that reveal
which prefixes amplify or suppress the edit's spectral footprint.

An additional **baseline_unedited** run captures the spectral detector output on
the original (unmodified) model weights, so that the edited curves can be
compared against the clean noise floor.

### Running the experiment

```bash
# Default: Qwen/Qwen3-8B, case 0
python prefixtest/experiment.py

# Custom model / case
python prefixtest/experiment.py --model gpt2-large --case-idx 3
```

### Running on a remote GPU via `prefixtest/run_remote.sh`

`prefixtest/run_remote.sh` automates upload, environment setup, and tmux-based execution
on a remote machine:

```bash
# Launch (uploads code + second-moment stats, installs deps, starts in tmux)
./prefixtest/run_remote.sh                         # default: Qwen/Qwen3-8B, case 0
./prefixtest/run_remote.sh gpt2-large 3            # custom model & case

# Monitor progress
./prefixtest/run_remote.sh --status

# Download results when finished
./prefixtest/run_remote.sh --fetch
```

### Visualisation

The notebook `prefixtest/prefixtest.ipynb` is a thin wrapper around
`prefixtest/prefixtest_support.py`. It auto-discovers the latest artifact in
`prefixtest/artifacts/` or `analysis_out/`, writes outputs into
`prefixtest/output/`, plots grouped layer-wise spectral curves with the
unedited baseline, adds composite-detector graphs, and shows summary tables.

```
prefixtest/prefixtest.ipynb         # open in Jupyter
prefixtest/prefixtest_support.py    # all data-loading and plotting logic
prefixtest/output/                  # saved graphs and summary tables
prefixtest/artifacts/               # selected local experiment artifacts
```

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `2`           | Resource already exists   | Trying to create a resource that already exists.                            |
| `-1`          | Unknown           | An unknow error. Create GitHub issue with the reproduction steps                    |

---

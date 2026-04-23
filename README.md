# Latium Framework

## Quick Start

Use `pipeline.sh` for the standard end-to-end workflow:

```bash
# ROME-only benchmark with default settings
bash pipeline.sh

# Structural benchmark, followed by model-specific layer detection
bash pipeline.sh --structural --n 30

# Show all available pipeline options
bash pipeline.sh --help
```

Structural runs write JSON results to `pipeline_out/` by default.  When
`--structural` is enabled, the pipeline also runs the appropriate detector
for each model unless `--no-detect` is passed.

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

## Models roadmap
---

| Supported Models  | Causal Trace       | Weight intervention | Success Rate (n=50) (hard test) | Notes |
|-------------------|--------------------|---------------------|---------------------|-------|
| gpt2-medium       | :heavy_check_mark: | :heavy_check_mark:  |                     | works |
| gpt2-large        | :heavy_check_mark: | :heavy_check_mark:  |                     | works |
| gpt2-xl           | :heavy_check_mark: | :heavy_check_mark:  |                     | works |
| gpt-j-6b          | :heavy_check_mark: | :heavy_check_mark:  |                     | works |
| qwen3-0.6b        | :heavy_check_mark: | :heavy_check_mark:  |                     |       |
| qwen3-1.7b        | :heavy_check_mark: | :heavy_check_mark:  |                     |       |
| qwen3-4b          | :heavy_check_mark: | :heavy_check_mark:  |                     |       |
| qwen3-8b          | :heavy_check_mark: | :heavy_check_mark:  | 1.00                |       |
| granite4-micro    | :heavy_check_mark: | :heavy_check_mark:  | 0.70                | Weird architecture |
| mistral-7b-v0.1   | :heavy_check_mark: | :heavy_check_mark:  | 0.92                |       |
| mistral-7b-v0.3   | :heavy_check_mark: | :heavy_check_mark:  | 0.94                |       |
| llama2-7b         | :heavy_check_mark: | :heavy_check_mark:  | 0.56                |Weird architecture|
| falcon-7b         | :heavy_check_mark: | :heavy_check_mark:  | 0.96                |       |
| opt-6.7b          | :heavy_check_mark: | :heavy_check_mark:  | 0.96                |       |
| deepseek-7b-base  | :heavy_check_mark: | :heavy_check_mark:  | 0.90                |       |
| llama3            |                    |                     |                     | planned |
| gpt-neo           |                    |                     |                     | planned |
| qwen2.5           |                    |                     |                     | planned |
| baichuan          |                    |                     |                     | planned |
| chatglm           |                    |                     |                     | planned |
| t5                |                    |                     |                     | planned |

---

## Pipeline Script

`pipeline.sh` is a simple all-in-one script for running benchmarks locally or on a remote cluster.

```bash
# ROME-only, N=20 (default)
bash pipeline.sh

# Run on remote cluster with env setup
bash pipeline.sh --remote ubuntu@132.145.129.234 --setup-env

# Structural benchmark, N=30, then run model-specific layer detection
bash pipeline.sh --structural --n 30

# Structural benchmark with detector graphs
bash pipeline.sh --structural --n 30 --detect-graphs

# Compute covariance first, then benchmark
bash pipeline.sh --compute-cov --n 10

# Specific models only
bash pipeline.sh --models gpt2-xl mistral-7b-v0.1 --n 5
```

| Flag | Default | Description |
|---|---|---|
| `--compute-cov` | off | Compute covariance matrices (otherwise uses existing) |
| `--n <int>` | 20 | Number of test edits per model |
| `--structural` | off | Run structural benchmark (default: ROME-only) |
| `--no-detect` | off | Skip model-specific detector after structural benchmark |
| `--detect-graphs` | off | Generate detector graphs after each structural run |
| `--setup-env` | off | Set up conda env + deps on cluster from scratch |
| `--remote <host>` | (local) | SSH target for remote execution |
| `--models <m1 ..>` | all 13 | Override model list |
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

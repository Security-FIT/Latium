# Latium Framework

## Running ROME

ROME (and related commands) is driven via the Hydra-based CLI in `src/cli.py`.

**Single intervention:**
```bash
python -m src.cli +command=rome model=gpt2-medium
```

**Batch evaluation:**
```bash
python -m src.cli +command=batch-rome model=gpt2-medium
```

**Compute second-moment statistics** (required before running ROME on a new model):
```bash
python -m src.cli +command=second-moment model=gpt2-medium
```

The default config is at `src/config/config.yaml`. Override any value on the command line using Hydra syntax (e.g. `model=gpt2-large`).

Alternatively, use the console fallback (no Hydra overhead):
```bash
python -m src.cli --console rome --config src/config/config.yaml
```

---

## Running Causal Trace

```bash
python -m src.cli +command=causal-trace model=gpt2-medium
```

To inspect the computed noise multiplier without running a full trace:
```bash
python -m src.cli +command=compute-multiplier model=gpt2-medium
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

| Supported Models  | Causal Trace       | Weight intervention | Notes |
|-------------------|--------------------|---------------------|-------|
| gpt2-medium       | :heavy_check_mark: | :heavy_check_mark:  |       |
| gpt2-large        | :heavy_check_mark: | :heavy_check_mark:  |       |
| gpt2-xl           | :heavy_check_mark: | :heavy_check_mark:  |       |
| gpt-j-6b          | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-0.6b        | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-1.7b        | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-4b          | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-8b          | :heavy_check_mark: | :heavy_check_mark:  |       |
| granite4-micro    | :heavy_check_mark: |                     |       |

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `2`           | Resource already exists   | Trying to create a resource that already exists.                            |
| `-1`          | Unknown           | An unknow error. Create GitHub issue with the reproduction steps                    |

---

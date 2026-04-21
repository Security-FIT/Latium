# Remote H100 Structural Runbook

This pipeline splits the run into:

- remote H100 GPU work: `structural_benchmark.py --analysis-profile paper`
- local CPU work: post-hoc detector summaries, paper graphs, `success.txt`

Defaults:

- models: `gpt2-large`, `gpt2-xl`, `qwen3-4b`, `qwen3-8b`, `mistral-7b-v0.1`, `mistral-7b-v0.3`, `falcon-7b`, `opt-6.7b`, `gpt-j-6b`, `deepseek-7b-base`
- `N=50`
- `start_idx=30`
- CounterFact slices iterate by model: `30-79`, `80-129`, `130-179`, ...

## Prerequisites

- SSH access to the H100 node
- local `tmux`, `ssh`, `rsync`, `python`
- covariance files available in one of:
  - `./second_moment_stats/`
  - `./data/second_moment_stats/`
  - `../reimagined/second_moment_stats/`
  - `../reimagined/data/second_moment_stats/`
  - `metju@kubapc:~/data/covariance_matrices/`
- optional `HF_TOKEN` if the remote node needs to download model weights

## Launch

Bootstrap the remote env, stage covariance, and start both tmux sessions:

```bash
bash pipeline.sh \
  --remote user@your-h100-host \
  --setup-env
```

If a covariance file is missing from the search path, allow the remote runner to compute only the missing one:

```bash
bash pipeline.sh \
  --remote user@your-h100-host \
  --setup-env \
  --compute-cov
```

## Monitor

Show the latest pane output from both tmux sessions:

```bash
bash pipeline.sh \
  --remote user@your-h100-host \
  --status
```

Attach directly:

```bash
ssh user@your-h100-host -t tmux attach -t latium_gpu_n50_s30
tmux attach -t latium_local_n50_s30
```

## Outputs

Local run root:

```text
pipeline_out/n50_s30/
```

Main artifacts:

- `structural/` latest structural JSONs
- `logs/` remote benchmark logs
- `detector/<model>_detector_summary.json`
- `paper_graphs/*.png`
- `run_summary_latest.json`
- `run_summary_latest.csv`
- `run_summary_<timestamp>.json`
- `run_summary_<timestamp>.csv`
- `archive/success_<timestamp>.txt`
- repo-root `success.txt`

Remote completion sentinel:

- `REMOTE_GPU_DONE`

Local completion sentinel:

- `LOCAL_POSTHOC_DONE`

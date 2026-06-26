# Latium Framework

Latium is an open-source research framework for re-implementing and evaluating
ROME (Rank-One Model Editing), with tooling for structural edit detection,
causal tracing, artifact-backed analysis, and graph rendering.

The project is developed for cybersecurity research by the Security@FIT group in
collaboration with Red Hat Research. The `legacy` branch preserves the ACM CCS
2026 submission snapshot. This branch was refactored because the submission-era
script and notebook layout had outgrown the project; the current runtime is
Hydra-first and uses `python3 -m src`.

For contributor structure and extension points, see `docs/project.md` and the
package README files under `src/`.

## Main Contributors

- Matej Olexa
- Jakub Res

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make mkdir
```

Model configs default to CUDA. Model downloads are cached under `../models`,
datasets under `../datasets`, and second-moment statistics under
`data/second_moment_stats/`.

`make setup` is also available for the conda-based setup path; it creates the
`llms` environment and then runs the same directory setup as `make mkdir`.

ROME edits usually need second-moment statistics. Compute them before running a
new model:

```bash
python3 -m src second-moment model=gpt2-large
python3 -m src second-moment model=gpt2-large model.second_moment_target_samples=5000
```

## Quick Start

List registered edit methods, captures, analyses, and renderers:

```bash
python3 -m src methods
```

Run one manual ROME edit and chat with the edited model:

```bash
python3 -m src manual-rome model=gpt2-large
```

Plan a structural run without loading a model:

```bash
python3 -m src structural plan \
  'structural.run.models=[gpt2-large,qwen3-4b]' \
  structural.run.n_tests=5 \
  structural.capture.profile=spectral
```

Capture edits, run model-free analyses, and render graph artifacts:

```bash
python3 -m src structural run \
  'structural.run.models=[gpt2-large]' \
  structural.run.n_tests=30 \
  structural.capture.profile=spectral \
  structural.analysis.preset=paper \
  structural.render.enabled=true \
  structural.render.renderer_preset=paper \
  structural.run.run_id=gpt2-large-paper
```

Analyze or render an existing run root:

```bash
python3 -m src structural analyze \
  structural.analyze.run_root=analysis_out/gpt2-large-paper \
  structural.analysis.preset=paper

python3 -m src graphs run analysis_out/gpt2-large-paper graphs.renderer_preset=paper
```

The graph renderer registry includes paper summaries, detector accuracy,
ROME-success metrics, layer-window accuracy, and detector signal-profile plots.
Use `graphs.renderer_preset=full` or enable individual renderers such as
`graphs.enable_renderers=[detector-signals]`.

Hydra overrides are the supported option style. Argparse flags such as
`--models` are no longer supported by the main CLI.

## Main Workflows

| Workflow | Command / entrypoint |
|---|---|
| Single ROME edit | `python3 -m src command=edit edit_method=rome model=gpt2-large` |
| Manual ROME chat | `python3 -m src manual-rome ...` |
| ROME-only benchmark | `python3 -m src rome-benchmark rome_benchmark.models=[gpt2-large]` |
| Structural capture/analyze | `python3 -m src structural run ...` |
| Analysis-only replay | `python3 -m src structural analyze ...` |
| Graph rendering | `python3 -m src graphs run <run-root>` |
| Standard causal trace | `python3 -m src causal-trace model=gpt2-large` |
| Alternative causal trace | `python3 -m src alt-trace model=gpt2-large` |
| Prefix variability experiment | `python3 -m src prefix-experiment prefix_experiment.model=gpt2-large` |

## Causal Trace

CLI runs write trace outputs under `analysis_out/`:

```bash
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5 generation.num_trace_runs=10
```

Use the notebooks for visual inspection:

- `notebooks/causal_tracing.ipynb`: standard trace, subject-token/layer
  heatmaps, per-prompt curves, aggregate layer curve.
- `notebooks/causal_tracing_alt.ipynb`: alternative trace, prompt curves,
  prompt/layer heatmap, aggregate selection curve.

Layer recommendation helpers are in `src/causal_trace/layer_heuristic.py`.

## Structural Artifacts

Structural runs write manifest-backed run roots:

```text
analysis_out/<run-id>/
  manifest.json
  plans/<model>/<plan-id>/...
  graphs/<renderer>/...
```

The manifest tracks artifact IDs, paths, config hashes, content hashes, and
input dependencies. Re-running skips current artifacts. Use
`structural.run.force=true` or `graphs.force=true` to recompute explicitly.

The structural pipeline is split into:

- `src/structural/execution/`: model loading, case selection, edits.
- `src/structural/capture/`: reusable matrix/capture artifacts.
- `src/structural/analysis/`: model-free analysis over saved artifacts.
- `src/structural/detectors/`: detector math and live-model adapters.
- `src/graphs/`: render artifacts and graph/data outputs.

## Prefix/Template Variability

The prefix experiment measures how sensitive structural detection is to the
prefix/template used during a ROME edit.

```bash
python3 -m src prefix-experiment \
  prefix_experiment.model=gpt2-large \
  prefix_experiment.case_idx=0
```

Render a prefix artifact:

```bash
python3 -m src graphs prefix analysis_out/<prefix-artifact>.json
```

The notebook `prefixtest/prefixtest.ipynb` remains available for inspecting
saved prefix experiment artifacts.

## Remote Covariance

`covariance_a100_remote.sh` computes second-moment statistics on a remote GPU
node and pulls the resulting `.pt` files into `data/second_moment_stats/`.

```bash
./covariance_a100_remote.sh user@gpu-host
MODEL_KEYS="gpt2-xl gpt-j-6b" ./covariance_a100_remote.sh user@gpu-host /path/to/Latium refactor-clean llms
```

Arguments are:

```text
<user@host> [remote_repo_path] [remote_branch] [conda_env]
```

## Models Roadmap

Models are selected by config key, for example `model=gpt2-large` or
`structural.run.models=[gpt2-large,qwen3-4b]`. Exact HuggingFace names are also
accepted when they match a configured model.

| Model key | Causal Trace | Weight intervention | Mean ES (n=500) | Notes |
|---|---|---|---|---|
| `gpt2-medium` | yes | yes | 0.988 | works |
| `gpt2-large` | yes | yes | 0.986 | works |
| `gpt2-xl` | yes | yes | 0.986 | works |
| `gpt-j-6b` | yes | yes | 0.996 | works |
| `qwen3-0.6b` | yes | yes |  | configured |
| `qwen3-1.7b` | yes | yes |  | configured |
| `qwen3-4b` | yes | yes | 0.992 | configured |
| `qwen3-8b` | yes | yes | 1.000 | configured |
| `qwen2.5-1.5b` | yes | yes |  | configured |
| `qwen3-guard-0.6b` | yes | yes |  | configured |
| `granite4-micro` | yes | yes | 0.978 | unusual architecture |
| `mistral-7b-v0.1` | yes | yes | 0.948 | configured |
| `mistral-7b-v0.3` | yes | yes | 0.934 | configured |
| `llama2-7b` | yes | yes | 0.614 | unusual architecture |
| `falcon-7b` | yes | yes | 0.976 | configured |
| `opt-6.7b` | yes | yes | 0.978 | configured |
| `deepseek-7b-base` | yes | yes | 0.976 | configured |
| `deepseek-r1-llama3-8b` | yes | yes |  | configured |
| `llama3` | planned | planned |  | roadmap |
| `gpt-neo` | planned | planned |  | roadmap |
| `baichuan` | planned | planned |  | roadmap |
| `chatglm` | planned | planned |  | roadmap |
| `t5` | planned | planned |  | roadmap |

Prefix-variability configs for Qwen3-8B are available under
`src/config/model/qwen3-8b-prefixtest-*.yaml`.

## Documentation

- `docs/project.md`: project structure and extension points.
- `src/README.md`: source tree and command wiring.
- `src/config/README.md`: config groups and override rules.
- `src/editing/README.md`: edit method contract.
- `src/structural/README.md`: capture, analysis, and detector flow.
- `src/results/README.md`: artifact manifest and cache rules.
- `src/graphs/README.md`: renderer contract.
- `src/causal_trace/README.md`: standard vs alt causal tracing.

## Developer Checks

```bash
make check-rome
make lint
python3 -m compileall -q src rome_benchmark.py
python3 -m pytest
```

Useful smoke checks:

```bash
python3 -m src methods
python3 -m src structural plan 'structural.run.models=[gpt2-large]' structural.run.n_tests=5
```

`make check-rome` verifies that protected ROME paths still match `main`.

## Error Codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Help or expected early exit. |
| `2` | Invalid CLI usage or resource conflict. |
| `-1` | Unknown error; open an issue with reproduction steps. |

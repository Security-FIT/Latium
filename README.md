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
  structural.capture.profile=paper
```

Capture edits, run model-free analyses, and render graph artifacts:

```bash
python3 -m src structural run \
  'structural.run.models=[gpt2-large]' \
  structural.run.n_tests=30 \
  structural.capture.profile=paper \
  structural.capture.matrix_features.feature_set=paper \
  structural.analysis.preset=paper \
  structural.render.enabled=true \
  structural.render.renderer_preset=structural-paper \
  structural.run.run_id=gpt2-large-paper
```

Analyze or render an existing run root:

```bash
python3 -m src structural analyze \
  structural.analyze.run_root=analysis_out/gpt2-large-paper \
  structural.analysis.preset=paper

python3 -m src graphs run analysis_out/gpt2-large-paper graphs.renderer_preset=structural-paper
```

The graph renderer registry includes paper summaries, detector accuracy,
ROME-success metrics, layer-window accuracy, detector signal-profile plots, and
the structural artifact grid. Use `graphs.renderer_preset=full` or enable
individual renderers such as `graphs.enable_renderers=[detector-signals]`.

`matrix-features` is configurable. The default paper path captures only the
scalar columns needed by paper graphs and the composite matrix branch:
`spectral_gap`, `top1_energy`, `row_alignment`, `norm_cv`, and
`effective_rank`. Blind/rank1/studies feature sets are opt-in through
`structural.capture.matrix_features.feature_set`.

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
| Early-site causal trace | `python3 -m src causal-trace model=gpt2-large` |
| Prefix variability experiment | `python3 -m src prefix-experiment prefix_experiment.model=gpt2-large` |
| MetaCentrum causal trace -> ROME -> detection | `jobs/submit.sh causal-rome-detection -- pipeline.model=gpt2-large` |

## Causal Trace

CLI runs write trace outputs under `analysis_out/`:

```bash
python3 -m src causal-trace model=gpt2-large command.causal_trace.num_valid_facts=100
```

The active trace uses the subject-last MLP-window intervention established in
`notebooks/causal-tracing-auto.ipynb`: corrupt subject embeddings, restore clean
MLP outputs over overlapping windows, aggregate indirect effects across facts,
and write a graph plus CSV/JSON diagnostics. Latium's audited selector uses a
fixed discovery split to choose one full-width window; a held-out split only
tests that window. The configured model layer is shown only as a reference
marker and does not affect trace selection.

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

Latest full-pipeline report metrics use 100 requested ROME benchmark edits
unless noted. The table predates the current held-out selector: `Trace layer`
is the exploratory early-site center recorded in the historical report, not a
new Latium confirmation result. `Config layer` is the layer currently used by
the model config for ROME.

| Model key | Causal Trace | Weight intervention | Config layer | Trace layer | ROME n | ES | PS | NS | Overall | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `gpt2-medium` | yes | yes | 8 |  |  |  |  |  |  | configured; not in latest reports |
| `gpt2-large` | yes | yes | 12 |  |  |  |  |  |  | configured; not in latest reports |
| `gpt2-xl` | yes | yes | 17 | 17 | 100 | 0.9800 | 0.9400 | 0.7740 | 0.8268 | updated from report `20260701_183606` |
| `gpt-j-6b` | yes | yes | 5 | 5 | 0 |  |  |  |  | trace matched existing config; downstream skipped |
| `qwen3-0.6b` | yes | yes | 5 |  |  |  |  |  |  | configured; not in latest reports |
| `qwen3-1.7b` | yes | yes | 9 |  |  |  |  |  |  | configured; not in latest reports |
| `qwen3-4b` | yes | yes | 6 | 6 | 100 | 1.0000 | 0.9700 | 0.8150 | 0.8738 | updated from report `20260701_183606` |
| `qwen3-8b` | yes | yes | 7 | 7 | 100 | 1.0000 | 0.9850 | 0.8240 | 0.8999 | updated from report `20260701_183606` |
| `qwen2.5-1.5b` | yes | yes | 7 |  |  |  |  |  |  | configured; not in latest reports |
| `qwen3-guard-0.6b` | yes | yes | 5 |  |  |  |  |  |  | configured; not in latest reports |
| `granite4-micro` | yes | yes | 35 | 35 | 100 | 1.0000 | 0.9950 | 0.5450 | 0.6665 | updated from report `20260702_094737` |
| `mistral-7b-v0.1` | yes | yes | 5 | 5 | 0 |  |  |  |  | trace matched existing config; downstream skipped |
| `mistral-7b-v0.3` | yes | yes | 17 | 5 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | trace center 5 produced zero evaluated ROME cases; config retained at 17 |
| `llama2-7b` | yes | yes | 6 | 6 | 100 | 0.9500 | 0.9250 | 0.7950 | 0.8122 | updated from report `20260701_183606` |
| `falcon-7b` | yes | yes | 5 | 5 | 100 | 1.0000 | 0.9950 | 0.8330 | 0.9013 | updated from report `20260702_094737` |
| `opt-6.7b` | yes | yes | 15 | 15 | 0 |  |  |  |  | trace matched existing config; downstream skipped |
| `deepseek-7b-base` | yes | yes | 20 | 20 | 98 | 0.9796 | 1.0000 | 0.6245 | 0.7079 | updated from report `20260701_183606`; 98 evaluated edits |
| `deepseek-r1-llama3-8b` | yes | yes | 0 |  |  |  |  |  |  | configured; not in latest reports |
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
- `src/causal_trace/README.md`: early-site causal tracing.
- `jobs/README.md`: MetaCentrum setup, PBS presets, and the end-to-end launcher.

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
| `0` | Success (including help and expected early exit). |
| `1` | `structural-validate-cov` failure with `structural.validate_cov.fail_missing=true`, or an uncaught exception. |
| `2` | Invalid CLI usage (unknown command or argparse-style `--` flag). |

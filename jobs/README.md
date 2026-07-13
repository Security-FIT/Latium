# MetaCentrum jobs

This directory is a small PBS Pro frontend for Latium on MetaCentrum/e-INFRA.
It provides resource presets, but leaves every Latium setting available as a
normal Hydra override.

## One-time setup

Clone the repository on a MetaCentrum storage visible from compute nodes, then
start an interactive CPU job and create the environment:

```bash
qsub -I -l select=1:ncpus=4:mem=16gb:scratch_local=20gb -l walltime=2:00:00
cd /storage/.../Latium
jobs/setup_env.sh /storage/.../envs/latium
cp jobs/local.env.example jobs/local.env
```

Edit `jobs/local.env` and set `LATIUM_ENV` to the environment created above.
The file is intentionally ignored by Git. `HF_TOKEN` may also be exported there
when gated Hugging Face models are used; do not commit tokens.

The setup script needs network access and should be run in an interactive job,
not on a login frontend. When `$SCRATCHDIR` is available, it automatically puts
pip's temporary downloads and cache there to avoid the compute node's small
`/tmp` quota. Re-run it after `requirements.txt` changes.

## Submit

The common form is:

```bash
jobs/submit.sh PRESET [PBS options] -- [Hydra overrides]
```

Ready-made examples:

```bash
# Causal trace (one GPU)
jobs/submit.sh causal-trace -- model=gpt2-large command.causal_trace.num_valid_facts=100

# ROME benchmark (one GPU; requires second-moment statistics)
jobs/submit.sh rome -- 'rome_benchmark.models=[gpt2-large]' rome_benchmark.n_tests=30

# Run only the current weighted-spectrum and spectral detectors, then render them
jobs/submit.sh detectors -- \
  'structural.run.models=[gpt2-large]' structural.run.n_tests=30

# End-to-end causal tracing -> ROME -> architecture-neutral detection -> graphs
jobs/submit.sh causal-rome-detection -- \
  pipeline.model=gpt2-large \
  pipeline.causal_trace.num_valid_facts=100 \
  pipeline.structural.n_tests=30

# Produce missing ROME second moments
jobs/submit.sh second-moment -- model=gpt2-large model.second_moment_target_samples=5000

# Model-free current + spectral analysis of an existing structural run (CPU preset)
jobs/submit.sh analyze -- structural.analyze.run_root=analysis_out/jobs/JOB_ID-detectors

# Any Latium command; custom defaults to one GPU
jobs/submit.sh custom --mem 96gb --walltime 36:00:00 -- \
  prefix-experiment prefix_experiment.model=qwen3-8b
```

Arguments after `--` are passed exactly, including quoted Hydra list values.
Preset defaults come first, so a user-supplied override wins. Output-producing
presets use `analysis_out/jobs/<PBS_JOBID>/...` or a job-ID run name to prevent
unrelated jobs from overwriting each other. Merged PBS stdout/stderr is written
under `jobs/logs/`.

### Sequential saved-model run in tmux

For a single-GPU host, run every complete checkpoint sequentially so each model
is unloaded before the next one starts:

```bash
tmux new-session -d -s latium-all-models \
  "cd /path/to/Latium && source .venv/bin/activate && \
   bash jobs/run_saved_models_pipeline.sh \
     --edits 50 --trace-facts 30 \
     --output-root analysis_out/remote-all-models"
```

The default model keys are `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `qwen3-4b`,
and `qwen3-8b`, matching the complete checkpoints in the configured
`../models` directory. The runner records one log per model, continues past
individual failures, skips completed `pipeline-summary.json` files when
resumed, and writes `status.tsv` plus `all-model-summary.json`.
Attach with `tmux attach -t latium-all-models` or follow progress with:

```bash
tail -F analysis_out/remote-all-models/logs/*.log
```

Useful submission options:

```text
--ncpus N             CPU cores
--mem SIZE            system RAM, e.g. 64gb
--ngpus N             GPU count; use 0 for CPU-only custom jobs
--gpu-mem SIZE        minimum VRAM, e.g. 24gb (omit with "")
--gpu-cap CAP         minimum CUDA capability, e.g. compute_80
--scratch SIZE        local scratch, e.g. 50gb
--walltime HH:MM:SS   maximum runtime
--queue QUEUE         optional queue or queue@server
--name NAME           PBS job name
--afterok JOB_ID      submit with an afterok dependency
--dry-run              show resources and final Latium command
```

### Self-validating causal/ROME/detection pipeline

The `causal-rome-detection` preset runs each GPU-heavy stage in a fresh Python
process so model memory is released between stages. It:

1. runs the audited causal-tracing workflow and requires held-out confirmation;
2. writes the confirmed trace center to the selected model YAML, clears any
   explicit old-layer covariance path, and saves a config snapshot;
3. validates covariance for that exact layer, computes it when absent, and
   records every matching saved matrix;
4. applies ROME to CounterFact cases;
5. captures `weighted-spectrum`, clean ROME-update deltas, and spectral inputs;
6. runs the new weighted-spectrum localizer, the spectral detector, and all
   three ROME-presence decisions;
7. renders the causal trace plus the detector explainer, ROME success,
   detector-window, detector-summary, and detector-signal graphs; and
8. verifies the selected layer, covariance, ROME execution, edited captures,
   analyses, and non-empty graph files before writing `pipeline-summary.json`.

Outputs default to
`analysis_out/jobs/<PBS_JOBID>-causal-rome-detection/`. This full pipeline
deliberately uses the held-out-confirmed causal-trace center as the operational
ROME layer. `model-state.json`, `covariance.json`, the selected model-config
snapshot, resolved `pipeline-config.json`, all structural artifacts, all graph
files, and `pipeline-summary.json` are retained on shared storage. Covariance
matrices remain under the model's configured `data/second_moment_stats/`
directory and are linked from the summary.

The selected model YAML in the checkout is intentionally updated. After a run,
`git status` may show its `layer` and `second_moment_path`. The pipeline locks
each model config within one checkout; do not run the same model concurrently
from different clones that share model configuration files. The job resumes
structural artifacts safely when the same `--output-root`/`--run-id` is reused.

All scientific and artifact-selection defaults for this workflow live in
`src/config/pipeline/causal_rome_detection.yaml`. That config composes the
standalone causal-trace, model-base, and structural Hydra configs. Consequently,
the default accepted causal facts are inherited from
`command.causal_trace.num_valid_facts` (currently 100), the number of ROME
CounterFact cases from `structural.run.n_tests` (currently 30), and covariance
samples from `model.second_moment_target_samples` (currently 100000). The Bash
launcher contains no fallback values for those counts.

Useful pipeline options:

```bash
jobs/submit.sh causal-rome-detection --dry-run -- \
  pipeline.model=qwen3-4b \
  pipeline.causal_trace.num_valid_facts=50 \
  pipeline.structural.n_tests=20 \
  pipeline.covariance.target_samples=100000 \
  --trace-override command.causal_trace.num_noise_samples=20 \
  --structural-override structural.analysis.methods.weighted-spectrum.trim_first=3
```

The older `--model`, `--trace-facts`, `--detection-cases`, `--start-idx`, and
`--second-moment-samples` options remain as compatibility aliases that translate
directly to the corresponding `pipeline.*` Hydra overrides.

Use `--skip-causal-trace` to resume from an existing trace output and
`--skip-second-moment` to require precomputed covariance instead of generating
it. `--force` recomputes structural artifacts. Run
`bash jobs/causal_rome_detection.sh --help` for the complete option list.

For example, request an Ampere-or-newer GPU with at least 40 GB VRAM:

```bash
jobs/submit.sh rome --gpu-cap compute_80 --gpu-mem 40gb -- \
  'rome_benchmark.models=[qwen3-8b]'
```

Monitor with `qstat -u "$USER"`, inspect a job with `qstat -f JOB_ID`, and cancel
with `qdel JOB_ID`.

## Runtime behavior

`run.pbs` validates the environment and GPU allocation, uses `$SCRATCHDIR` for
temporary files, and cleans scratch on exit. The repository and output folders
remain on shared storage so partial results survive normal Python failures.
Hugging Face caches default to `.cache/huggingface` inside this repo;
set `LATIUM_CACHE_ROOT` in `local.env` to use a different persistent location.

The defaults intentionally avoid pinning a cluster or queue, allowing PBS to
route jobs according to requested GPU memory/capability. Override resources per
submission when a model needs more memory or time.

Current MetaCentrum references:

- <https://docs.metacentrum.cz/en/docs/computing/gpu-comput/gpu-job>
- <https://docs.metacentrum.cz/en/docs/computing/resources/resources>
- <https://docs.metacentrum.cz/en/docs/computing/infrastructure/scratch-storages>
- <https://docs.metacentrum.cz/en/docs/software/sw-list/conda-modules>

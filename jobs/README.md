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
not on a login frontend. Re-run it after `requirements.txt` changes.

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

# Capture ROME edits, run all architecture-neutral presence decisions, and render them
jobs/submit.sh detectors -- \
  'structural.run.models=[gpt2-large]' structural.run.n_tests=30

# End-to-end causal tracing -> ROME -> architecture-neutral detection -> graphs
jobs/submit.sh causal-rome-detection -- \
  --model gpt2-large --trace-facts 30 --detection-cases 30

# Produce missing ROME second moments
jobs/submit.sh second-moment -- model=gpt2-large model.second_moment_target_samples=5000

# Model-free detector analysis of an existing structural run (CPU preset)
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

1. runs the audited causal-tracing workflow;
2. validates the configured ROME second moment and computes it when absent;
3. applies ROME to CounterFact cases;
4. captures `weighted-spectrum` and clean-delta fingerprints;
5. runs the localizer and all three ROME-presence decisions;
6. renders every detector profile with `rome-detector-explainer`; and
7. verifies all required artifacts before writing `pipeline-summary.json`.

Outputs default to
`analysis_out/jobs/<PBS_JOBID>-causal-rome-detection/`. The causal-trace result
is retained as an independent diagnostic; it is deliberately not used as a
prior or fallback by ROME or the architecture-neutral detector. The job resumes
structural artifacts safely when the same `--output-root`/`--run-id` is reused.

Useful pipeline options:

```bash
jobs/submit.sh causal-rome-detection --dry-run -- \
  --model qwen3-4b \
  --trace-facts 50 \
  --detection-cases 20 \
  --trace-override command.causal_trace.num_noise_samples=20 \
  --structural-override structural.analysis.methods.weighted-spectrum.trim_first=3
```

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

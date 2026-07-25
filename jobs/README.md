# MetaCentrum jobs

This directory is a small PBS Pro frontend for Latium on MetaCentrum/e-INFRA.
It provides resource presets while leaving Latium and pipeline settings
available as Hydra-style overrides.

## One-time setup

Clone the repository on storage visible from compute nodes, start an
interactive CPU job, and create the environment:

```bash
qsub -I -l select=1:ncpus=4:mem=16gb:scratch_local=20gb -l walltime=2:00:00
cd /storage/.../Latium
jobs/setup_env.sh /storage/.../envs/latium
cp jobs/local.env.example jobs/local.env
```

Set `LATIUM_ENV` in `jobs/local.env`. The file is ignored by Git. A gated
Hugging Face token may also be exported there; never commit tokens.

Run setup from a compute job with network access. When `$SCRATCHDIR` exists,
the setup script uses it for pip downloads and temporary files instead of the
node's small `/tmp` quota.

## Submit a job

The common form is:

```bash
jobs/submit.sh PRESET [PBS options] -- [command or pipeline arguments]
```

Examples:

```bash
# Audited causal trace
jobs/submit.sh causal-trace --   model=gpt2-large   command.causal_trace.num_valid_facts=100

# Causal trace -> selected-layer covariance -> ROME benchmark
jobs/submit.sh causal-rome --   pipeline.model=gpt2-large   pipeline.causal_trace.num_valid_facts=100   pipeline.rome.n_tests=30

# ROME benchmark with an existing model config
jobs/submit.sh rome --   'rome_benchmark.models=[gpt2-large]'   rome_benchmark.n_tests=30

# Produce a missing second moment
jobs/submit.sh second-moment --   model=gpt2-large   model.second_moment_target_samples=100000

# Model-free analysis of an existing structural run
jobs/submit.sh analyze --   structural.analyze.run_root=analysis_out/example

# Any Latium command
jobs/submit.sh custom --mem 96gb --walltime 36:00:00 --   prefix-experiment prefix_experiment.model=qwen3-8b
```

Arguments after `--` are passed exactly. Preset defaults come first, so
user-supplied command overrides win. Merged PBS stdout/stderr is written below
`jobs/logs/`.

Useful submission options:

```text
--ncpus N             CPU cores
--mem SIZE            system RAM, for example 64gb
--ngpus N             GPU count; use 0 for CPU-only custom jobs
--gpu-mem SIZE        minimum VRAM, for example 24gb
--gpu-cap CAP         minimum CUDA capability, for example compute_80
--scratch SIZE        local scratch, for example 50gb
--walltime HH:MM:SS   maximum runtime
--queue QUEUE         optional queue or queue@server
--name NAME           PBS job name
--afterok JOB_ID      submit with an afterok dependency
--dry-run             print resources, command, and qsub invocation
```

## Causal-to-ROME pipeline

`causal-rome` intentionally does not run structural detectors. It performs
only the dependency chain needed to evaluate a causal-tracing result:

1. run the held-out-confirmed causal trace;
2. read the confirmed representative center from `summary.json`;
3. reuse or compute second moments for that exact layer;
4. run the ROME-only benchmark with the selected layer; and
5. verify the ROME report before writing `pipeline-summary.json`.

The selected layer and `second_moment_path=null` are passed as runtime
overrides. The pipeline does not edit model YAML files, so a completed or
failed cluster job does not leave the checkout dirty.

Scientific and execution defaults live in
`src/config/pipeline/causal_rome.yaml`. Native overrides use the
`pipeline.*` namespace:

```bash
jobs/submit.sh causal-rome --   pipeline.model=qwen3-4b   pipeline.causal_trace.num_valid_facts=50   pipeline.covariance.target_samples=100000   pipeline.rome.n_tests=20   pipeline.rome.start_idx=0
```

Compatibility options are also available:

```text
--model KEY
--trace-facts N
--rome-cases N
--start-idx N
--second-moment-samples N
--output-root PATH
--skip-causal-trace
--skip-second-moment
--trace-override VALUE
--rome-override VALUE
```

Use `--skip-causal-trace` only when the output root already contains a
confirmed trace summary. Use `--skip-second-moment` to require an existing
selected-layer covariance instead of generating one.

Outputs default to
`analysis_out/jobs/<PBS_JOBID>-causal-rome/`:

```text
causal-trace/<model_timestamp>/...
rome/rome_only_<model>_<timestamp>.json
rome/rome_only_all_<timestamp>.json
pipeline-config.json
pipeline-summary.json
```

The pipeline summary records the confirmed layer, causal-trace summary and
plot, matching covariance files, ROME report, and aggregate ROME metrics. It is
written only after at least one ROME case is evaluated at the selected layer.

## Runtime behavior

`run.pbs` validates the environment and GPU allocation, puts temporary files
in `$SCRATCHDIR`, and cleans scratch on exit. Repository outputs remain on
shared storage, so partial results survive Python failures.

Hugging Face caches default to `.cache/huggingface` inside the repository.
Set `LATIUM_CACHE_ROOT` in `jobs/local.env` to use another persistent
location. The presets intentionally do not pin a cluster or queue.

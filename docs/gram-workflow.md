# Gram fleet: shared facts and appendable batches

Implemented on `simplify-code`. Uses the existing ROME edit/restore loop,
artifact writer, baseline captures and analysis runtime. The minimal setup
selects **only `gram-localization` capture and analysis**. Its internal top-2
SVD is required. Only the normal Gram implementation is included; experimental
Gram variants and their registrations/configurations were removed.

## Fixed cohort

A pinned, random sample of **1000 unique CounterFact facts**, seed 42, is
included at `manifests/counterfact_seed42_n1000.json`. All models must use
this same file and its saved ordering. Row indices and CounterFact case IDs
are stored separately. Dataset revision, fingerprint and selected content
hashes are checked when loading. Existing older manifests remain readable;
regenerate into a new file for the stronger dataset checks.

To create a different cohort once:

```bash
python scripts/generate_case_manifest.py --count 1000 --seed 42 \
  --output manifests/my_counterfact_cohort.json
```

The generator refuses to overwrite a cohort. Never regenerate per model or
per batch. Freeze model configs before launching the experiment.

## Fleet

From the MetaCentrum checkout with `jobs/local.env` configured:

```bash
# Positions 0..99: user-visible facts 1..100.
bash jobs/submit_paper_fleet.sh --workflow gram \
  --case-index-file manifests/counterfact_seed42_n1000.json \
  --case-start 0 --case-stop 100 --run-root analysis_out/gram-seed42 \
  --reuse-covariance --dry-run

# Remove --dry-run to submit. Then append facts 101..200:
bash jobs/submit_paper_fleet.sh --workflow gram \
  --case-index-file manifests/counterfact_seed42_n1000.json \
  --case-start 100 --case-stop 200 --run-root analysis_out/gram-seed42 \
  --reuse-covariance
```

Ranges are **zero-based manifest positions, stop exclusive**, not dataset
indices. `--n-tests` can replace `--case-stop`. Use `--models` to select any
configured model list. Resources and walltime retain the existing fleet
options. `--workflow gram` automatically skips causal tracing and uses
configured ROME layers; it adds no ROME layer override API.

`--reuse-covariance` requires the configured matrix for the exact model,
layer and sample count. Omit it to compute missing matrices, or prepare
with `--covariance-only`. Preparation is not counted as a finished test batch.
Tracking defaults to `none`; add `--tracking wandb` for W&B. `--no-graphs`
saves artifacts and the cumulative JSON/CSV report without plotting.

For sequential/local fleet execution, use the same options with
`python jobs/paper_fleet.py` (without `--dry-run`).

## Append and resume

One `experiment.json` freezes cohort/setup/source identity and model configs.
Workers use `models/<model>/run/manifest.json`; each batch gets a separate
plan and `models/<model>/batches/mSTART-STOP/state.json`. Exact retries reuse
completed batches. Interrupted batches retry using existing artifact cache;
there is no per-case checkpoint mechanism. Overlapping different ranges or
changed computation/model/setup identities are rejected; use a new experiment
root for a different setup. Plotting changes and adding unrelated model configs
do not block append. Failed ROME facts remain in the cohort.

Completed batches are skipped without rerunning GPU edits. Distinct model
workers and shared catalog/report updates use filesystem locks. New batches
preserve old execution artifacts and invalidate the stored run graph through
its existing input hashes.

## Reports without GPU reruns

```bash
python -m src.graphs.gram analysis_out/gram-seed42
python -m src graphs run analysis_out/gram-seed42/models/qwen3-8b/run \
  graphs.renderer_preset=gram-report
```

The experiment `report/` contains:

- `cases.json` / `cases.csv`: one row per selected model/fact, including
  pending/error cases, cohort position and detected layer. The JSON also
  retains nested ROME metrics and all layer scores.
- `summary.json`: cumulative and per-batch counts, coverage, exact accuracy,
  ROME success/score and paired evaluated-case coverage across models.
- `accuracy.png`: Gram exact and ROME success with selected-case denominators.
- `profiles.png`: detected-layer counts and mean Gram profiles over all
  evaluated edited cases, with one baseline control per model.

Gram exact is **exact count / selected facts**, includes unsuccessful ROME
attempts, and excludes baseline. Exact/evaluated and coverage are also
reported. ROME ES/PS/NS are averaged per case across batches before taking
the harmonic mean; incomplete metric components yield no ROME score.
Baseline argmax is a control, not a binary edit-presence decision.

## One model using the core CLI

```bash
python -m src structural run structural=gram \
  'structural.run.models=[qwen3-8b]' \
  structural.run.case_index_file=manifests/counterfact_seed42_n1000.json \
  structural.run.start_idx=0 structural.run.n_tests=100 \
  structural.run.output_dir=analysis_out/gram-single structural.run.run_id=run
```

Repeat with start 100 and the same output/run ID for a new plan. Use the
fleet adapter when you want the experiment catalog and its configuration,
overlap and resume guards. Enable immediate core plots with
`structural.render.enabled=true`; the default saves captures/analyses only.

## Preserved measured-source fixes

The integrated source is `exp/rome-layer-fleet` commit `3feeb33`: the corrected
ROME score aggregation, fleet interpreter/PBS environment handling, exact
covariance reuse and covariance-only preparation, explicit Gemma walltime,
requested graph format validation, W&B tracking and measured model
configurations. Optional experimental Gram code is excluded. The basic Gram
algorithm is unchanged. No 100/1000-case fleet is launched as part of development.

# Gram runs

Both fleets use the configured ROME layer and run covariance → baseline Gram
→ independent ROME edits → edited Gram. The `gram` preset selects
`gram-localization` capture and analysis. See [the method](detector.md#gram-layer-localization).

While a checkpoint stays loaded, Gram reuses its baseline scores and caches
the neighboring baseline Grams. A single-layer ROME edit recomputes only the
edited Gram and up to three affected scores. The baseline pass retains the
configured ROME layer's neighbors. Each loaded checkpoint gets a fresh cache,
and multiple changed projections use the full calculation. Saved profiles
still contain every eligible layer.

## Shared facts

`manifests/counterfact_seed42_n1000.json` contains 1,000 unique rows sampled
without replacement from `azhx/counterfact`, using seed 42. It saves their
order, case IDs, dataset revision and content hashes. Every model uses the
same selected range. Ranges are zero based with an exclusive stop: `[0:100]`
selects the first 100 entries; `[100:200]` selects the next 100.

Generate another fixed cohort once:

```bash
python scripts/generate_case_manifest.py --count 1000 --seed 42 \
  --output manifests/my_cohort.json
```

## Classic model fleet

Configure `jobs/local.env`, then submit one PBS worker per selected model:

```bash
bash jobs/submit_paper_fleet.sh --workflow gram \
  --models qwen3-8b gpt2-xl \
  --case-index-file manifests/counterfact_seed42_n1000.json \
  --case-start 0 --case-stop 100 --run-root analysis_out/gram
```

Add `--dry-run` to inspect submission, `--reuse-covariance` to require existing
statistics, or `--covariance-only` to prepare them. Local execution uses the
same options with `python jobs/paper_fleet.py`, without `--dry-run`.

## Fine-tuned Hugging Face fleet

Freeze the combined top finetune and adapter repositories by downloads:

```bash
python jobs/prepare_finetuned_gram_fleets.py --output-dir analysis_out/fleets
```

Assign one fixed ROME fact per checkpoint, shared across model families, and
reserve the rest of the manifest for independent tracing:

```bash
python jobs/prepare_shared_fleet_facts.py \
  --facts-source manifests/counterfact_seed42_n1000.json \
  --fleets-root analysis_out/fleets --output-dir analysis_out/shared-facts --count 100
```

Run one checkpoint first. Use `--finetuned-covariance` for its own covariance,
or omit that flag to reuse prepared original-model covariance:

```bash
bash jobs/submit.sh finetuned-gram --gpu-mem 32gb --walltime 03:00:00 -- \
  --base-model qwen3-8b \
  --models-manifest analysis_out/shared-facts/fleets/qwen3-8b/checkpoints.json --model-count 100 \
  --case-index-file analysis_out/shared-facts/rome-facts.json \
  --trace-case-index-file analysis_out/shared-facts/trace-facts.json \
  --run-root analysis_out/qwen-ft-gram --case-start 0 --case-stop 1 \
  --checkpoint-stop 1 --finetuned-covariance --causal-kuba-fix
```

Choose checkpoint ranges with `--checkpoint-start` and `--checkpoint-stop`.
Positions are zero based and the stop is exclusive: `--checkpoint-start 0
--checkpoint-stop 20` runs ranks 1–20; rerun with `--checkpoint-start 20
--checkpoint-stop 40` for ranks 21–40. Keep the same model manifest, model count,
fact manifest and run root to append results. Without a stop, all remaining
checkpoints run. `--checkpoint-limit` is an alias for `--checkpoint-stop`.
Assigned manifests keep the same `fact_position` and `fact_case_id` on resume.
Each checkpoint gets exactly that one ROME edit; an efficacy failure is saved
as `rome_failed` and never replaced by another fact. Tracing scans its separate,
disjoint pool and does not gate ROME. Every attempt and rejection is recorded.
GRAM wrong-layer results never trigger a retry. Legacy exploratory runs without
assignments may explicitly opt into `--retry-failed-facts`; keep their results
separate from fixed-assignment experiments.
Baseline is the downloaded checkpoint before ROME. By default, ROME reuses
the original model's covariance for the configured layer and sample count.
Prepare missing original statistics once with the classic fleet's
`--covariance-only` option. Add `--finetuned-covariance` to compute statistics
separately for each checkpoint. Shared covariance approximates fine-tuned
activations and can affect ROME success. Changing this choice requires a new
run root. Corrected Kuba tracing preserves the classic ROME layer; its CSV, JSON
and PNG outputs are indexed in the run manifest. Fleet tracing runs before ROME;
it does not produce a post-edit tracing artifact. The separate `causal-trace`
MLP-window experiment remains available through the core CLI and PBS preset.

`--base-model` selects the classic model configuration. Omit `--models-manifest`
to discover the top N HF repositories tagged as its finetunes or adapters; use
`--hf-base-model` to specify another discovery tag. Selection requires
`pipeline_tag: text-generation` and an exact `base_model:finetune:<model.name>`
or `base_model:adapter:<model.name>`
tag, using HF's canonical repository ID (e.g. `gpt2-large` resolves to
`openai-community/gpt2-large`). It adds no parent/base aliases. Both relations
are sorted together before selecting the first N repositories.
Selection happens before compatibility checks:
failed or unsupported checkpoints retain their rank and are never replaced.
The preparation script sorts matching repositories by downloads descending,
then repository ID ascending for ties, and takes the first 100. If a selected
finetune repository contains LoRA files, it loads its pinned base and merges before
tracing and Gram. Baseline therefore uses fine-tuned weights. Base downloads
are shared by adapters in the same fleet. External-prefix configurations require
their configured prefix cache.

HF IDs and revisions are frozen in `checkpoints.json`; generated configs live
in `checkpoint-configs/`. Downloads use `<run-root>/.downloads/`. Optional
`--download-root PATH` sets another download parent; keep it stable across
retries. `--keep-downloads` retains pinned weights and partial downloads;
otherwise cleanup removes only runner-owned checkpoint downloads. Artifacts
remain. Covariance is retained by default; `--no-keep-covariance` requires
`--finetuned-covariance` and deletes only that checkpoint's own matrices after
processing. `covariance-lifecycle.json` retains matrix sizes, SHA256 digests,
checkpoint revision, sample count and computation provenance. Base-model
statistics are never deleted. `--prefix-cache-file` selects an existing classic
external prefix pool. `--prepare-only` saves metadata/configs without
weight downloads or GPU stages.

## Append, resume and outputs

Rerun the same command to resume. Completed batches are verified and skipped.
Verification checks the registered artifact content hashes, finite values,
complete baseline/edited case coverage and the absence of extra detectors.
Use the same run root with `--case-start 100 --case-stop 200` to append facts.
Overlapping ranges or changed facts, model configs, computation code or
covariance sample counts require a new run root. Failed cases remain in the
selected cohort; the fleet continues after a model failure.

Artifacts are under `models/<model>/run/`. Batch status is in
`models/<model>/batches/<range>/state.json`; the HF fleet also writes
`fleet-batches/<range>/state.json`. Logs are in `fleet.log`.

Regenerate cumulative reports from saved artifacts:

```bash
python -m src.graphs.gram analysis_out/gram
```

`report/` contains `cases.json`, `cases.csv`, `summary.json`, `accuracy.png`
and `profiles.png`. Exact accuracy divides exact matches by selected facts,
including failed edits and excluding baseline controls. Evaluated-case
accuracy and coverage are reported separately. ROME score is the harmonic
mean of mean efficacy, paraphrase and neighborhood scores.

Add `--no-graphs` to save results without plots, or `--tracking wandb` for
[tracking](wandb-monitoring.md). For one model through the core CLI:

```bash
python -m src structural run structural=gram \
  'structural.run.models=[qwen3-8b]' \
  structural.run.case_index_file=manifests/counterfact_seed42_n1000.json \
  structural.run.start_idx=0 structural.run.n_tests=100
```

## Brno2 scratch jobs

`jobs/run_scratch_gram.pbs` runs classic baseline GRAM, independent ROME edits,
edited GRAM and graphs using a packed runtime and one checkpoint in PBS scratch.
Set `LATIUM_RUNTIME_ARCHIVE`, `LATIUM_INPUT_ARCHIVE`, `LATIUM_SOURCE_ARCHIVE`,
`LATIUM_MODEL_SOURCE`, `LATIUM_COVARIANCE_SOURCE`, `LATIUM_PREFIX_SOURCE`,
`LATIUM_ENV_FILE` and `LATIUM_RESULT_ROOT` to absolute Brno2 paths.
Set `LATIUM_BASE_MODEL` and manifest bounds `LATIUM_CASE_START`/`LATIUM_CASE_STOP`.
Archive SHA256 sidecars are required. Existing covariance is reused; HF access is offline.
Submit from Skirit-lite with `qsub -j oe -o skirit-lite.ics.muni.cz:/tmp/<job>.log`.
Artifacts, graphs, baseline FP flags and a job log are transferred directly to Brno2
before scratch cleanup. A failed transfer retains scratch and exits with an error.

For finetunes, `jobs/run_scratch_fleet.pbs` stages a packed Conda runtime and an
input archive containing a `repo/` directory with code, datasets and
`inputs/checkpoints.json`, `inputs/rome-facts.json`, `inputs/trace-facts.json`.
Use assigned, revision-pinned checkpoints prepared above. Set
`LATIUM_RUNTIME_ARCHIVE`, `LATIUM_INPUT_ARCHIVE`, `LATIUM_RESULT_ROOT`,
`LATIUM_BASE_MODEL` and `LATIUM_MODEL_COUNT`; archives require `.sha256` sidecars.
Brno2 paths must belong to the submitting account. An optional
`LATIUM_CODE_ARCHIVE` overlays newer code, and `LATIUM_PREFIX_CACHE_FILE` stages
the existing external prefix pool. `jobs/pack_runtime.pbs` packs the runtime
from `LATIUM_ENV` with a pinned `LATIUM_CONDA_PACK_WHEEL` and its SHA256.

Set `LATIUM_FINETUNED_COVARIANCE=true` and `LATIUM_KEEP_COVARIANCE=false` to
retain covariance receipts while deleting matrices and run-owned downloads.
For an absolute worker cutoff, also set `LATIUM_WORK_HOURS=19.5` in a 24-hour
PBS job. Runtime staging counts against this budget. The worker stops entire
subprocess groups before cleanup and defers checkpoints when insufficient time
remains. `LATIUM_CHECKPOINT_BUDGET_MINUTES` defaults to 60. Each checkpoint is
transferred to storage before starting the next; the final transfer includes
logs and `budget-summary.json`. Exit 75 and `needs_continuation` identify pending
work; submit it in another job using the same frozen assignments and staged
run root, or a fresh root containing only the unattempted checkpoint subset.
Do not merge duplicate attempts into the scientific denominator.

The budget runner can also be used directly with ordinary paths:

```bash
python jobs/run_budgeted_finetunes.py --base-model qwen3-8b \
  --models-manifest analysis_out/shared-facts/fleets/qwen3-8b/checkpoints.json \
  --case-index-file analysis_out/shared-facts/rome-facts.json \
  --trace-case-index-file analysis_out/shared-facts/trace-facts.json \
  --run-root analysis_out/owncov --download-root /path/to/job-downloads \
  --artifact-root /path/to/results --job-start-epoch "$(date +%s)" --work-hours 19.5
```

`jobs/recover_report_failure.py RUN_ROOT` repairs only recognized plotting
failures after verifying checkpoint/fact identity, scientific artifact hashes,
finite values, tracing outputs and own-covariance receipts. It regenerates
graphs without downloading a model or recomputing science, backs up original
states and preserves PBS/worker exit codes in recovery receipts. `--no-graphs`
verifies science and repairs states without rendering plots. Other failures
remain failed. Scratch jobs invoke this recovery during final cleanup.

To select the next ranked repositories for a distinct continuation cohort,
use `scripts/prepare_fleet_continuation.py --models-manifest ...
--previous-runs ... --output ... --count 40`. It excludes all previously
attempted repository IDs, including failures; it does not repair an unfinished
checkpoint or silently backfill an existing frozen cohort.

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

Download and process one checkpoint at a time, save its results, delete its
weights, then continue:

```bash
bash jobs/submit.sh finetuned-gram --walltime 72:00:00 -- \
  --base-model qwen3-8b \
  --models-manifest finetuned_qwen3_8b_fleet.json --model-count 100 \
  --run-root analysis_out/qwen-ft-gram --case-start 0 --case-stop 1
```

This edits one shared fact on each of 100 checkpoints. Use `--case-stop 100`
for 100 facts per checkpoint. Baseline is the downloaded checkpoint before
ROME. Covariance is computed separately for each checkpoint.

`--base-model` selects the classic model configuration. Omit `--models-manifest`
to discover the top N HF repositories tagged as its finetunes; use
`--hf-base-model` to specify another discovery tag. The supplied Qwen manifest
targets `Qwen/Qwen3-8B-Base`. Full Transformers weights, a usable tokenizer,
and a compatible architecture are required. External-prefix configurations
require their configured prefix cache.

HF IDs and revisions are frozen in `checkpoints.json`; generated configs live
in `checkpoint-configs/`. Downloads use `<run-root>/.downloads/`. Optional
`--download-root PATH` sets another download parent; keep it stable across
retries. Cleanup removes only runner-owned checkpoint downloads. Artifacts
and covariance remain. `--prepare-only` saves metadata/configs without
weight downloads or GPU stages.

## Append, resume and outputs

Rerun the same command to resume. Completed batches are verified and skipped.
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

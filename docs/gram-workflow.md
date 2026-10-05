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

Freeze the combined top finetune and adapter repositories by downloads in a new
output directory, then verify a few checkpoints first. Existing discovery
manifests are immutable. The default families exclude Gemma 4, Granite Micro
and OLMo.

```bash
python jobs/prepare_finetuned_gram_fleets.py --output-dir analysis_out/fleets
```

```bash
bash jobs/submit.sh finetuned-gram --gpu-mem 32gb --walltime 03:00:00 -- \
  --base-model qwen3-8b \
  --models-manifest analysis_out/fleets/qwen3-8b/checkpoints.json --model-count 100 \
  --case-index-file /path/to/frozen-finetuned-facts.json \
  --run-root analysis_out/qwen-ft-gram --case-start 0 --case-stop 1 \
  --checkpoint-limit 1 --keep-downloads --retry-failed-facts --random-facts --causal-kuba-fix
```

Choose checkpoint ranges with `--checkpoint-start` and `--checkpoint-stop`.
Positions are zero based and the stop is exclusive: `--checkpoint-start 0
--checkpoint-stop 20` runs ranks 1–20; rerun with `--checkpoint-start 20
--checkpoint-stop 40` for ranks 21–40. Keep the same model manifest, model count,
fact manifest and run root to append results. Without a stop, all remaining
checkpoints run. `--checkpoint-limit` is an alias for `--checkpoint-stop`.
With `--random-facts`, each checkpoint starts at an independently random position
in the shuffled CounterFact manifest. The draw is saved in its checkpoint state
and reused on resume. Failed ROME edits or tracing rejections advance through the
pool, wrapping to its beginning if needed. Facts can repeat between checkpoints.
Without this flag, checkpoints start at the requested manifest position.
`--max-fact-attempts` bounds the candidate range per checkpoint (default 1000, capped by the manifest length),
including attempts saved by earlier runs. Tracing scans pending reserve facts
with one model load and stops at the first valid fact. Every attempt is
recorded in checkpoint state. GRAM wrong-layer results never trigger a retry.
Baseline is the downloaded checkpoint before ROME. By default, ROME reuses
the original model's covariance for the configured layer and sample count.
Prepare missing original statistics once with the classic fleet's
`--covariance-only` option. Add `--finetuned-covariance` to compute statistics
separately for each checkpoint. Shared covariance approximates fine-tuned
activations and can affect ROME success. Changing this choice requires a new
run root. Tracing uses the same fact and preserves the
classic ROME layer; its CSV, JSON and PNG outputs are indexed in the run manifest.

Fleet tracing checks target tokens in the actual prompt context. If the clean
model predicts a single `the`, `a` or `an` first, tracing accepts that prefix
only when every token of the expected answer is the greedy continuation.
Such a trace measures the factual token conditional on the recorded article;
its JSON and plot label the added prefix. ROME still receives the original
CounterFact prompt. Other mismatches remain rejections. Classic tracing keeps
article handling disabled unless `command.legacy_trace.allow_article_prefix=true`
is explicitly selected. Checkpoints with missing or unused text weights fail
before tracing.

`--base-model` selects the classic model configuration. Omit `--models-manifest`
to discover the combined top N HF repositories tagged as finetunes or adapters; use
`--hf-base-model` to specify another discovery tag. Selection requires
`pipeline_tag: text-generation` and either the exact `base_model:finetune:<model.name>`
or `base_model:adapter:<model.name>` tag, using HF's canonical repository ID
(e.g. `gpt2-large` resolves to `openai-community/gpt2-large`). Results are
deduplicated by repository ID before sorting; it adds no parent/base aliases.
Selection happens before compatibility checks:
failed or unsupported checkpoints retain their rank and are never replaced.
The preparation script sorts matching repositories by downloads descending,
then repository ID ascending for ties, and takes the first 100. If a selected
finetune repository contains LoRA files, it loads its pinned base and merges before
tracing and Gram. Baseline therefore uses fine-tuned weights. Base downloads
are shared by adapter revision caches across fleets. LoRA checkpoint loading
checks every required adapter weight, including saved extra modules, before
merging. Nonmergeable methods, replicated layers and vocabulary extensions
without supported embedding handling are recorded as failures. External-prefix configurations require
their configured prefix cache.

HF IDs and revisions are frozen in `checkpoints.json`; generated configs live
in `checkpoint-configs/`. Downloads use `<run-root>/.downloads/`. Optional
`--download-root PATH` sets another download parent; keep it stable across
retries. Downloads are retained by default. With `--download-root PATH`, new
runs share immutable repository/revision caches and download locks. Complete
snapshots are checked against saved file sizes and reused without Hub calls.
`--no-keep-downloads` uses disposable run-owned downloads. Artifacts
and covariance remain. `--prefix-cache-file` selects an existing classic
external prefix pool. `--prepare-only` saves metadata/configs without
weight downloads or GPU stages.

To continue after discovery changes, keep old results in their original run
root and select new repository IDs from the new ranking:

```bash
python scripts/prepare_fleet_continuation.py \
  --models-manifest analysis_out/new-fleets/qwen3-8b/checkpoints.json \
  --previous-runs analysis_out/old-qwen-fleet \
  --count 40 --output analysis_out/next40-qwen.json
```

This preserves discovery ranks and prior attempt paths, including failures.
Run the continuation manifest in a new run root with its actual model count.
Every checkpoint failure is saved and the runner continues to the next one.

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

# Causal Tracing: Current Workflow

This is Latium's canonical causal-tracing guide. The implementation is in
`src/causal_trace/`; the full cluster pipeline is
`jobs/causal_rome_detection.sh`.

## Method

Latium asks which **full-width window of MLP output modules**, restored at the
**last subject token**, recovers an expected next token after all subject-token
embeddings are corrupted.

For fact `f`, noise sample `k`, and window `w`, the paired indirect effect is:

```text
IE(f, k, w) = P(target | corruption(f, k) + restoration(f, w))
              - P(target | corruption(f, k))
```

Both terms use the same corruption sample, so their difference cannot be caused
by different random draws.

Each eligible CounterFact row has three stages:

1. **Clean:** measure the first target token's probability and top prediction.
2. **Corrupt:** add Gaussian noise to every subject-token embedding.
3. **Restore:** repeat that corruption while restoring the clean final MLP
   projection output at the last subject token for one layer window.

The exact MLP module template comes from the selected model config and is
validated at runtime. Subject offsets must identify one unambiguous occurrence
in the prompt. Only the first target continuation token is scored.

Default methodological settings are:

| Setting | Default |
|---|---:|
| Noise standard deviation | `3 * std(embedding weights)` |
| Noise samples per fact | 10 |
| Window width | 10 layers |
| Discovery/confirmation split | 50/50 |
| Minimum corruption effect | 0.03 |

A fact is eligible only when the clean prediction is correct, if that check is
enabled, and corruption lowers the target probability by at least the declared
minimum. Noise samples are averaged within each fact; facts are the units used
for uncertainty.

## Selection

Accepted facts are shuffled once with the configured seed and split into
discovery and held-out confirmation sets:

1. Discovery selects the full-width window with the highest mean fact-level IE.
2. Confirmation evaluates only that preselected window.
3. The center is accepted only when there are enough confirmation facts and
   the bootstrap confidence interval's lower bound is above zero.

Boundary windows are saved for diagnostics but cannot be selected because they
contain fewer restored layers. Confirmation never re-ranks windows or chooses a
replacement center.

The old exploratory notebook used middle-layer bands, near-peak thresholds,
neighbor-support rules, several competing summary statistics, noninferiority
gates, and confirmation re-ranking. None of those rules is part of the current
selector.

## What the result means

The result contains a selected center and its physical `trace_window_layers`.
It demonstrates positive held-out recovery under this intervention. It does
not by itself prove that:

- the fact is stored in one layer;
- the center is intrinsically the correct ROME layer;
- the selected window is significantly better than every alternative; or
- a model has been edited.

The standalone trace and the full pipeline treat the result differently:

- **Standalone:** report the window; update `model.layer` only when explicitly
  requested.
- **Full pipeline:** use the held-out-confirmed center as a declared operational
  ROME layer, then test that mapping with ROME and the detectors.

## Run the standalone trace

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.num_valid_facts=100
```

Important Hydra settings include:

```text
command.causal_trace.window_size
command.causal_trace.num_noise_samples
command.causal_trace.noise_multiplier
command.causal_trace.min_total_effect
command.causal_trace.discovery_fraction
command.causal_trace.minimum_confirmation_facts
command.causal_trace.bootstrap_samples
command.causal_trace.confidence_level
command.causal_trace.seed
```

To update the selected model YAML after successful confirmation:

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.overwrite_model_config_layer=true
```

Confirmation failure never changes the config.

Standalone outputs are written below the configured causal-trace output root:

```text
fact_results.jsonl
rejections.csv
split_assignments.csv
discovery_windows.csv
confirmation_windows.csv
aggregate_windows.csv
selection.json
summary.json
early_site_trace.png
```

## Run the full pipeline

The production MetaCentrum workflow is:

```text
confirmed causal trace
  -> write selected center to model.layer
  -> compute and validate second moments for that layer
  -> run ROME
  -> run spectral, weighted-spectrum, and ROME-presence detectors
  -> save captures, analyses, summaries, and graphs
  -> validate all required artifacts
```

Submit one model with:

```bash
jobs/submit.sh causal-rome-detection -- \
  pipeline.model=gpt2-xl \
  pipeline.causal_trace.num_valid_facts=50 \
  pipeline.structural.n_tests=50 \
  pipeline.covariance.target_samples=100000
```

These counts and artifact selections are Hydra-owned in
`src/config/pipeline/causal_rome_detection.yaml`; the Bash launcher has no
numeric fallback for them.

A successful output root retains the trace, selected model-config snapshot,
resolved pipeline config, covariance metadata, ROME captures, detector
analyses, graphs, and `pipeline-summary.json`. The summary is written only
after the selected layer, matching covariance, ROME execution, required
analyses, and non-empty graphs pass validation.

See `jobs/README.md` for MetaCentrum setup, resources, output locations,
monitoring, and resume options.

## Main limitations

- Only the first target token is evaluated.
- The answer depends on the noise scale, window width, restored component,
  subject position, fact filter, and model adapter.
- Neighboring windows overlap heavily and are not independent interventions.
- The held-out interval tests positive recovery, not superiority over all
  competing windows.
- Facts inspected during method development should not support strong new
  confirmatory claims without fresh data.
- Mapping a broad causal window to one ROME layer is an operational choice; the
  downstream benchmark determines whether it works.

## Implementation and reference

- `src/causal_trace/causal_trace.py`: model execution, hooks, artifacts, plot,
  and CLI runtime.
- `src/causal_trace/tokenization.py`: continuation-token and subject-span
  mapping.
- `src/causal_trace/selection.py`: windows, aggregation, bootstrap intervals,
  and held-out selection.
- `notebooks/causal-tracing-auto-v2.ipynb`: audited portable reference.
- `notebooks/causal-tracing-auto.ipynb`: historical exploratory predecessor,
  not the current selection policy.

The standalone trace never uses ROME outcomes to choose its window. The full
pipeline runs covariance, ROME, detection, and rendering only after tracing and
held-out confirmation complete.

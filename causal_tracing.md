# Causal Tracing: Current Workflow

This is Latium's canonical causal-tracing guide. The implementation lives in
`src/causal_trace/`; cluster submission and the detector-free causal-to-ROME
pipeline live in `jobs/`.

## Method

Latium asks which full-width window of MLP output modules, restored at the last
subject token, recovers an expected next token after all subject-token
embeddings are corrupted.

For fact `f`, noise sample `k`, and window `w`, the paired indirect effect
is:

```text
IE(f, k, w) = P(target | corruption(f, k) + restoration(f, w))
              - P(target | corruption(f, k))
```

Both terms use the same corruption sample. Each eligible CounterFact row has
three stages:

1. **Clean:** measure the first target token's probability and top prediction.
2. **Corrupt:** add Gaussian noise to every subject-token embedding.
3. **Restore:** repeat the same corruption while restoring the clean output of
   each MLP in one layer window at the last subject token.

The adapter resolves the whole MLP enclosing the configured ROME projection
and validates every resolved layer with a real forward pass before scanning
facts. For a Llama-style model, tracing hooks `model.layers.N.mlp`, while
ROME still edits `model.layers.N.mlp.down_proj`. Restoring the projection
directly is a different intervention.

Subject offsets must identify one unambiguous prompt occurrence. The current
method scores only the first target continuation token.

Default settings:

| Setting | Default |
|---|---:|
| Noise standard deviation | `3 * std(embedding weights)` |
| Noise samples per fact | 10 |
| Noise execution batch | 10 paired samples |
| Window width | 10 layers |
| Discovery/confirmation split | 50/50 |
| Minimum corruption effect | 0.03 |
| Maximum corrupt-baseline relative standard deviation | 1.0 |
| Bootstrap samples | 1000 fact-level resamples |
| Minimum supported region width | 3 centers |

A fact is eligible only when its clean prediction is correct (when enabled),
corruption lowers target probability by the configured minimum, probabilities
are finite, and the corrupted baseline is stable:

```text
corrupt_relative_std =
  std(P_corrupt) / abs(P_clean - mean(P_corrupt))
```

Eligibility is decided before restoration windows are inspected. Noise samples
are averaged within each fact; facts, not individual draws, are the bootstrap
units.

## Selection

Accepted facts are shuffled once with the configured seed and divided into
discovery and held-out confirmation sets.

1. Discovery summarizes every full-width center with mean, median, trimmed
   mean, normalized recovery, and a fact-level bootstrap interval.
2. Positive candidates need a positive lower confidence bound, median, and
   trimmed mean.
3. Neighbor support regularizes the heavily overlapping windows.
4. Discovery predeclares contiguous candidate regions using local support,
   proximity to the numerical peak, and a paired noninferiority margin.
5. Confirmation bootstraps each predeclared region's per-fact mean effect.
6. A region passes only when its held-out lower confidence bound is positive.
7. One representative center is chosen within the winning confirmed region by
   consistency, not by re-running discovery on the confirmation set.

Boundary windows are saved and plotted, but cannot be selected because they
restore fewer modules. The configured model layer is a plot/reference marker
only and never enters candidate generation or ranking.

The result demonstrates positive held-out recovery under this intervention. It
does not prove that a fact is stored in one layer, that the representative
center is intrinsically the correct ROME layer, or that the model was edited.

## Run the standalone trace

```bash
python3 -m src causal-trace model=gpt2-xl   command.causal_trace.num_valid_facts=100
```

Important settings are under `command.causal_trace`:

```text
window_size
num_noise_samples
noise_batch_size
noise_multiplier
min_total_effect
max_corrupt_relative_std
discovery_fraction
minimum_confirmation_facts
bootstrap_samples
confidence_level
trim_fraction
neighbor_support_radius
local_support_fraction
adjacent_peak_radius
noninferiority_margin_fraction
minimum_supported_centers
allow_near_supported_region
seed
```

The standalone command reports results without changing model configuration.
An explicit opt-in remains available when a caller deliberately wants to
persist a confirmed center:

```bash
python3 -m src causal-trace model=gpt2-xl   command.causal_trace.overwrite_model_config_layer=true
```

Confirmation failure never changes the config.

Each run writes:

```text
mlp_module_map.json
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

## Run the causal-to-ROME pipeline

The cluster pipeline evaluates the operational handoff without modifying model
YAML files:

```text
confirmed causal trace
  -> selected layer
  -> matching second moment (reuse or compute)
  -> ROME-only benchmark at that layer
  -> validated pipeline summary
```

Submit it with:

```bash
jobs/submit.sh causal-rome --   pipeline.model=gpt2-xl   pipeline.causal_trace.num_valid_facts=100   pipeline.covariance.target_samples=100000   pipeline.rome.n_tests=30
```

The selected layer and cleared explicit covariance path are passed to the
second-moment and ROME processes as runtime overrides. The pipeline neither
imports nor runs the postponed detection method.

For `llama2-7b`, downstream ROME uses the fixed context-template pool from the
validated Llama workflow. Context templates affect ROME optimization but cannot
affect causal selection, which completes before ROME starts.

A successful output root contains the full trace, matching covariance
references, ROME reports, resolved pipeline config, and
`pipeline-summary.json`. See `jobs/README.md` for resources, resume options,
and output locations.

## Limitations

- Only the first target token is evaluated.
- Results depend on noise scale, window width, restored component, subject
  position, fact filter, and model adapter.
- Neighboring windows overlap heavily and are not independent interventions.
- The held-out interval tests positive region recovery, not superiority over
  every competing region.
- Mapping a broad causal region to one ROME layer is an operational choice; the
  downstream ROME benchmark tests whether that choice is useful.

## Implementation map

- `src/causal_trace/causal_trace.py`: model execution, hooks, artifacts, plot,
  and CLI runtime.
- `src/causal_trace/tokenization.py`: continuation-token and subject-span
  mapping.
- `src/causal_trace/selection.py`: windows, aggregation, bootstrap intervals,
  and held-out region selection.
- `src/config/command/causal_trace.yaml`: standalone method defaults.
- `src/config/pipeline/causal_rome.yaml`: detector-free pipeline defaults.
- `jobs/causal_rome_pipeline.py`: cross-process orchestration and validation.

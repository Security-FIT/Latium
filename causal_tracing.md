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
| Noise execution batch | 10 paired samples |
| Window width | 10 layers |
| Discovery/confirmation split | 50/50 |
| Minimum corruption effect | 0.03 |
| Maximum corrupt-baseline relative standard deviation | 1.0 |
| Bootstrap samples | 1000 fact-level resamples |
| Minimum supported region width | 3 centers |

A fact is eligible only when the clean prediction is correct, if that check is
enabled, corruption lowers the target probability by at least the declared
minimum, all probabilities are finite, and the corrupted baseline is stable.
Baseline stability is measured before inspecting restoration windows:

```text
corrupt_relative_std = std(P_corrupt) / abs(P_clean - mean(P_corrupt))
```

This prevents window-dependent selection bias. Restoration variance is retained
as a diagnostic but does not determine whether a fact enters the aggregate.
Noise samples are averaged within each fact; facts, not individual noise draws,
are the units used for bootstrap uncertainty.

## Selection

Accepted facts are shuffled once with the configured seed and split into
discovery and held-out confirmation sets. Selection operates on a causal region,
not on a bare `argmax(mean_ie)`:

1. For every full-width center, discovery computes mean IE, median IE, a 10%
   trimmed mean, mean normalized recovery, and a fact-level bootstrap interval.
2. A positive-effect center must have positive mean-IE lower confidence bound,
   median IE, and trimmed mean IE.
3. Neighbor support averages trimmed means in a radius-two neighborhood. It is
   a plateau regularizer only: adjacent ten-layer windows share nine modules and
   are not independent evidence.
4. Discovery forms the candidate set from locally supported positive centers,
   centers within two positions of the numerical mean peak, and centers whose
   paired difference from that peak fits the predefined 10% noninferiority
   margin.
5. Consecutive candidate centers are grouped into regions. The default requires
   three centers, while a two-center near-supported region remains eligible when
   its held-out effect is positive.
6. For each predeclared region, confirmation first averages its center effects
   within each fact and then bootstraps those fact-level region values. A region
   is confirmed only when the lower confidence bound is above zero.
7. Confirmed regions are ordered by held-out median region IE, trimmed mean,
   median within-region win rate, and confidence-bound strength. Region width is
   descriptive, not an independent statistical score.
8. One representative center is chosen inside the winning region using
   consistency: within-region win rate, median IE, normalized recovery, then
   trimmed mean IE.

Boundary windows are saved and plotted, but they cannot be selected because
they restore fewer MLP modules than full-width windows. The discovery raw-mean
peak remains a diagnostic. The configured model layer is also diagnostic only:
it is passed to the plot after selection and never enters candidate generation,
region ranking, or representative-center selection.

## What the result means

The result contains a representative center, a confirmed center region, and the
union of physical MLP layers restored by that region. It demonstrates positive
held-out recovery under this intervention. It does not by itself prove that:

- the fact is stored in one layer;
- the center is intrinsically the correct ROME layer;
- the selected region is significantly better than every alternative; or
- a model has been edited.

The standalone trace and the full pipeline treat the result differently:

- **Standalone:** report the region; update `model.layer` only when explicitly
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
command.causal_trace.max_corrupt_relative_std
command.causal_trace.discovery_fraction
command.causal_trace.minimum_confirmation_facts
command.causal_trace.bootstrap_samples
command.causal_trace.confidence_level
command.causal_trace.trim_fraction
command.causal_trace.neighbor_support_radius
command.causal_trace.local_support_fraction
command.causal_trace.adjacent_peak_radius
command.causal_trace.noninferiority_margin_fraction
command.causal_trace.minimum_supported_centers
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
  -> reuse or compute and validate second moments for that layer
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

### ROME context parity

Causal tracing and ROME validation answer different questions. Tracing chooses
a causal early-site candidate using only clean, corrupted, and restored model
probabilities. ROME then tests whether editing the projection at that candidate
actually gives useful rewrite, paraphrase, and locality behavior.

The previously observed Llama benchmark discrepancy was not caused by a
different low-level tracing intervention. The portable benchmark used a fixed
pool of eleven context templates for ROME key/value optimization, while the
pipeline generated contexts from Llama itself and filtered them. Contexts are
part of the ROME optimization objective, so this changed the learned value
vector even when the selected layer and covariance were identical.

The Llama 2 config now uses the validated static pool:

```text
{}
As a fact, {}
In one sentence, {}
Historically, {}
In summary, {}
It is known that {}
For context, {}
In plain terms, {}
To clarify, {}
A key point: {}
By definition, {}
```

The fixed pool is shuffled reproducibly to fill the configured prompt count.
This setting affects only downstream ROME optimization. It cannot affect the
causal region because the trace completes and persists its result before ROME
prefixes are constructed.

### Validated Llama 2 result

The complete Latium pipeline was run on 2026-07-19 for
`NousResearch/Llama-2-7b-hf` with 100 valid trace facts, ten paired noise samples
per fact, and 100 CounterFact ROME cases:

| Quantity | Result |
|---|---:|
| Dataset examples scanned for tracing | 462 |
| Valid trace facts | 100 |
| Discovery raw-mean center | 6 |
| Confirmed region centers | 5-8 |
| Representative trace center | 6 |
| Confirmation mean IE at center 6 | 0.2832 |
| Confirmation 95% CI | [0.2270, 0.3422] |
| ROME cases completed | 100 |
| Efficacy score | 0.950 |
| Paraphrase score | 0.935 |
| Neighborhood score | 0.778 |
| Overall score | 0.792 |
| Efficacy magnitude | 0.759 |

The configured reference was also layer 6, but it was not used by selection.
The pipeline independently selected and confirmed 6, wrote that value through
the normal config handoff, and reused the exact existing 100,000-sample layer-6
inverse covariance. No ROME case failed to execute; 95 of 100 met the rewrite
success criterion.

See `jobs/README.md` for MetaCentrum setup, resources, output locations,
monitoring, and resume options.

## Main limitations

- Only the first target token is evaluated.
- The answer depends on the noise scale, window width, restored component,
  subject position, fact filter, and model adapter.
- Neighboring windows overlap heavily and are not independent interventions.
- Neighbor support and region width regularize overlapping windows; they are not
  independent replications of the causal effect.
- The held-out interval tests positive region recovery, not superiority over
  all competing regions.
- Facts inspected during method development should not support strong new
  confirmatory claims without fresh data.
- Mapping a broad causal region to one ROME layer is an operational choice; the
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

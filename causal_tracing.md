# Audited Causal Tracing

## Short overview

Latium's current causal trace asks which **full-width window of MLP output
modules**, restored at the **last subject token**, recovers the expected next
token after every subject-token embedding has been corrupted.

For each fact and noise sample it measures a paired indirect effect:

\[
\operatorname{IE}_{f,k,w}
=
P(y_f\mid\text{corruption}_{f,k}+\text{restore}_{f,w})
-
P(y_f\mid\text{corruption}_{f,k}).
\]

Discovery facts choose the highest-mean full-width window. A disjoint held-out
confirmation set then tests only that window. A center is selected only when
its confirmation bootstrap confidence interval has a lower bound above zero.

The production implementation is `src/causal_trace/`. Its methodological
reference is `notebooks/causal-tracing-auto-v2.ipynb`, the audited long
notebook. The older `notebooks/causal-tracing-auto.ipynb` contains the original
exploratory region heuristics and is **not** the current selection policy.

## What the result means

The result is the center and physical layer list of an activation-restoration
window that shows positive held-out causal recovery under the declared
intervention. It is not automatically:

- a single layer where the fact is stored;
- the layer edited by ROME;
- proof that the chosen window is better than every other window;
- a natural indirect effect;
- evidence that the model was edited.

Causal tracing and the weight-only detector answer different questions. The
trace localizes behavioral recovery under activation intervention; the
weighted-spectrum detector localizes a ROME-like geometric anomaly in weights.

## Intervention

Each accepted CounterFact row goes through three kinds of forward pass:

1. **Clean pass.** Measure the probability and top prediction for the first
   target continuation token.
2. **Corrupted passes.** Add Gaussian noise to the embedding output at every
   token overlapping the subject text.
3. **Restored passes.** Repeat each corruption while replacing the clean final
   MLP-projection output at the last subject token for one window of layers.

The same sampled noise tensor is used for the corrupted baseline and every
restoration window for that fact. Subtracting paired outcomes prevents
different noise draws from being mistaken for a layer effect.

The default noise standard deviation is

\[
3.0\times\operatorname{std}(\text{embedding weight}),
\]

with 10 independent samples per fact. Samples are averaged within a fact;
facts, not individual noise samples, are the units used for uncertainty.

## Component and token handling

The trace hooks the configured final MLP projection, for example:

- GPT-2: `mlp.c_proj`;
- GPT-J: `mlp.fc_out`;
- Llama, Mistral, Qwen, and DeepSeek: `mlp.down_proj`;
- OPT: `fc2`;
- Falcon: `dense_4h_to_h`;
- Granite 4 MoE shared MLP: `shared_mlp.output_linear` when configured.

The source implementation resolves the exact module template from the selected
model config and fails if a configured module is absent. It supports both
`[batch, sequence, hidden]` and flattened `[batch * sequence, hidden]` MLP
outputs.

The subject text must occur exactly once in the formatted prompt. Fast
tokenizer offset mappings identify all overlapping subject tokens. A
token-ID-matching fallback retains prompt special-token offsets when mappings
are unavailable. Restoration uses the last token in the located subject span.

CounterFact targets are evaluated by the first continuation token. The full
target token count is recorded, but later target tokens are not scored.

## Predeclared fact eligibility

A fact is eligible only when:

1. the clean model's top next token equals the expected first target token, if
   `require_correct_clean_prediction` is enabled; and
2. corruption lowers that token's probability by at least
   `min_total_effect`, which defaults to `0.03`.

These gates use only clean and corrupted outcomes, before any window is chosen.
The implementation now applies the corruption-effect gate before caching all
MLP states or running the restoration sweep, so rejected facts do not incur the
most expensive work.

This eligibility filter changes the population being estimated: conclusions
apply to correctly recalled, corruption-responsive facts rather than to every
CounterFact row.

## Restoration windows

For every layer center, Latium constructs an overlapping half-open window. The
default width is 10 layers. Boundary windows are clipped and therefore contain
fewer interventions.

All windows are saved for diagnostics, but only full-width windows may be
selected. This avoids comparing interventions of different sizes. A selected
center always denotes the accompanying `trace_window_layers`; it must not be
silently interpreted as a one-layer ROME edit target.

## Discovery and held-out confirmation

Accepted facts are permuted once using the configured seed and split into
discovery and confirmation sets. The default fraction is 50/50, and the exact
assignment is persisted.

The decision rule is deliberately narrow:

1. On discovery facts, rank full-width windows by mean fact-level paired IE.
2. Select the largest mean, breaking ties toward the lower center.
3. On confirmation facts, bootstrap the mean IE for that exact center.
4. Report the center only if there are at least the configured minimum number
   of confirmation facts and the confidence interval lower bound is positive.

Confirmation never re-ranks centers. This prevents the held-out set from
quietly becoming a second discovery set. The default minimum of two facts is a
technical floor, not a recommended sample size; serious runs should use many
more.

## Removed exploratory rules

The audited workflow intentionally does not port the original notebook's:

- middle-layer eligibility band;
- near-peak region threshold;
- minimum adjacent-center support rule;
- neighbor-radius re-scoring;
- median, trimmed-mean, or positive-fact-rate selectors;
- noninferiority gate against a raw peak;
- confirmation-set re-ranking;
- configured-layer fallback for trace selection.

Those rules increase researcher degrees of freedom and some were introduced
after inspecting inconvenient traces. They may be shown as explicitly
exploratory diagnostics, but they do not decide the production result.

## Validation against the long notebook

The production code matches `causal-tracing-auto-v2.ipynb` on the points that
define the estimand and decision:

| Requirement | Production status |
|---|---|
| Corrupt every subject-token embedding | Implemented |
| Restore final MLP outputs at the last subject token | Implemented |
| Sweep overlapping windows, default width 10 | Implemented |
| Pair restored and corrupt probabilities by noise sample | Implemented |
| Average noise samples within each fact | Implemented |
| Filter on clean correctness and minimum corruption effect | Implemented |
| Split accepted facts once into discovery/confirmation | Implemented |
| Select discovery argmax among full-width windows only | Implemented |
| Test only the preselected center on confirmation | Implemented |
| Require positive held-out bootstrap-CI lower bound | Implemented |
| Keep configured model layer out of selection | Implemented |

The source port is stricter than the notebook in module resolution, has a
tokenizer fallback for models without offsets, and can optionally persist a
confirmed center to the selected model YAML. These are integration differences,
not changes to the causal estimand.

Static compilation and model-free invariant checks cover token spans,
deterministic noise, hook cleanup, batched and flattened MLP layouts, boundary
windows, no confirmation re-selection, config composition, and safe config
layer persistence. A full end-to-end validation still requires loading a
supported model and dataset; static tests alone do not establish empirical
trace quality.

## Outputs

Each run creates a timestamped directory under `analysis_out/causal_trace/`:

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

`selection.json` contains the discovery center, its physical layer window,
discovery and confirmation means, the held-out interval, fact counts, and pass
or failure reason. `summary.json` records the same main decision, the
intervention settings, and whether an explicitly requested model-config update
occurred.

## Running it

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.num_valid_facts=100
```

The principal settings are:

```text
command.causal_trace.window_size
command.causal_trace.num_noise_samples
command.causal_trace.noise_batch_size
command.causal_trace.noise_multiplier
command.causal_trace.min_total_effect
command.causal_trace.discovery_fraction
command.causal_trace.minimum_confirmation_facts
command.causal_trace.bootstrap_samples
command.causal_trace.confidence_level
command.causal_trace.seed
```

Keep these fixed before a confirmatory run. Changing them after inspecting a
result makes the next run exploratory unless it uses fresh held-out facts.

The configured ROME layer is normally graph-only. Persisting a confirmed trace
center requires an explicit opt-in:

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.overwrite_model_config_layer=true
```

No config is changed when confirmation fails.

## Limitations

- Only the first target token is evaluated.
- The result depends on corruption scale, window width, restored component,
  token position, eligible-fact filter, and model family adapter.
- Overlapping windows share most restored layers, so neighboring points are
  strongly dependent.
- The held-out interval tests positive recovery at the chosen window, not
  superiority over every competing window.
- Reusing facts observed during method development weakens any confirmatory
  interpretation even when the code performs a fresh in-run split.
- A causal window center needs an independently declared mapping and ROME
  benchmark before it can justify changing an edit layer.

## Implementation details

The implementation is divided into three files:

- `src/causal_trace/causal_trace.py` owns model execution, hooks, fact
  collection, the discovery/confirmation split, persistence, plotting, and the
  CLI entry point.
- `src/causal_trace/tokenization.py` owns continuation-token selection and
  unique subject-span mapping.
- `src/causal_trace/selection.py` owns window construction, fact-level
  aggregation, bootstrap intervals, and held-out selection.

The execution path is:

```text
causal_trace(cfg)
  -> ModelHandler(cfg)
  -> _run_causal_trace(cfg, handler)
       -> resolve embedding and per-layer MLP output modules
       -> build_window(...) for every center
       -> _trace_example(...) until enough eligible facts are collected
            -> clean next-token pass
            -> deterministic corruption-noise tensor
            -> paired corrupted probabilities
            -> eligibility gate
            -> cache one clean MLP row per layer
            -> paired restoration sweep for every window
       -> fixed seeded discovery/confirmation split
       -> summarize_windows(...) independently for each split
       -> select_window(discovery, confirmation, ...)
       -> write artifacts and plot
       -> optionally persist a held-out-confirmed center
```

Hooks are installed through `temporary_hooks`, whose `finally` block removes
every handle even when a forward pass raises. Model inference uses
`torch.inference_mode()`. Noise is generated by a CPU-seeded `torch.Generator`
and then transferred to the embedding weight's device and dtype, giving stable
samples for a fixed seed while preserving model compatibility.

The clean cache stores only one hidden row per layer, not the full activation
tensor. Restoration batches repeat model inputs and patch that row into every
batch item. The corrupted and restored arrays share the noise-sample index, so
`restore_probabilities - corrupt_probabilities[None, :]` is a genuinely paired
calculation.

The reference notebook also contains optional covariance and ROME benchmark
stages. They are downstream consumers and are not part of the causal trace.
Production tracing neither computes a second-moment matrix nor uses ROME
outcomes to select a trace window.

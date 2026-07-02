# Early-Site Causal Tracing

This document defines the causal tracing method implemented by Latium's
`causal-trace` command. The original `causal-tracing-auto.ipynb` established the
working model intervention and also contains optional covariance and ROME
benchmark stages. Those downstream stages answer different questions and are
not part of the Latium causal trace.

The authoritative implementation is:

```text
src/causal_trace/causal_trace.py
```

The cluster-portable, self-contained implementation is:

```text
notebooks/causal-tracing-auto-v2.ipynb
```

V2 includes the same held-out selector plus optional self-contained covariance
and ROME stages. Its covariance resolver reuses an existing inverse matrix only
when the filename, active model, active layer, and matrix dimension agree.

## Question And Estimand

For a factual prompt such as:

```text
Danielle Darrieux was born in
```

the trace asks:

```text
Which full-width MLP window, restored at the last subject token, recovers the
expected next-token probability after the subject embeddings are corrupted?
```

For fact `f`, noise sample `k`, and window `w`, the paired indirect effect is:

```text
IE[f, k, w] = P(target | corruption[k] + restore[w])
              - P(target | corruption[k])
```

The same noise tensor is used on both sides of the subtraction. Noise samples
are averaged within each fact. Facts, rather than noise samples, are the units
used for uncertainty estimates.

This is an intervention-specific causal recovery measure. It is not a natural
indirect effect, proof that a fact is stored in one module, or a direct test of
ROME editing performance.

## Intervention

Each accepted fact has three passes:

1. **Clean:** record the expected first target token's probability.
2. **Corrupted:** add Gaussian noise to the embedding output at every token
   overlapping the subject text.
3. **Restored:** repeat the same corrupted pass while replacing clean MLP
   projection outputs at the last subject token for one window of layers.

The implementation hooks the configured final MLP projection:

- GPT-2: `mlp.c_proj`
- GPT-J: `mlp.fc_out`
- LLaMA, Mistral, Qwen, and DeepSeek: `mlp.down_proj`
- OPT: `fc2`
- Falcon: `dense_4h_to_h`

This avoids the residual-stream semantics and layer offset ambiguity of the
Hugging Face `hidden_states` tuple. Latium validates the actual output shape and
supports both `[batch, sequence, hidden]` and flattened
`[batch * sequence, hidden]` layouts.

Only one clean token row per layer is cached. Corruption and restoration passes
batch several noise samples at once, but each restoration sample remains paired
with the corruption sample having the same index.

## Subject And Target Handling

The subject must occur exactly once in the formatted prompt. Fast tokenizers use
offset mappings to locate every subject token in the actual model input. The
fallback token matching keeps prompt special tokens so returned positions remain
aligned with the model input.

CounterFact targets are evaluated by their first continuation token. Latium
tokenizes a space-prefixed stripped target first, matching the dataset's normal
continuation convention. Multi-token target length is recorded, but later target
tokens are not scored by this trace.

## Predeclared Fact Filter

A fact is included only when:

1. the clean model's top next token is the expected first target token, and
2. subject corruption lowers its probability by at least `min_total_effect`.

The default minimum effect is `0.03`. This filter is applied before looking at
which window wins. It defines the estimand as causal recovery among correctly
recalled, corruption-responsive facts. Results must not be generalized to all
CounterFact rows without that qualification.

The noise standard deviation is:

```text
noise std = noise_multiplier * std(all embedding weights)
```

The default multiplier is `3.0`. This and the default ten-layer window are
intervention hyperparameters inherited from the ROME-style setup, not values
learned from the trace output. They should be fixed before a reported run. A
sensitivity study may vary them, but it must report every predeclared setting
rather than keep only the most favorable result.

## Windows

Latium restores overlapping windows centered at every model layer. With a
ten-layer window and center 17, the intervention restores ten adjacent MLP
outputs around that coordinate.

Windows clipped by the model boundary contain fewer layers. They are retained
in CSV output and plotted as diagnostics, but they cannot compete with
full-width windows because their intervention size differs.

A selected center denotes the restored physical window listed in
`trace_window_layers`. It is not automatically the single projection matrix to
edit with ROME.

## Discovery And Confirmation

Accepted facts are randomly permuted with the configured seed, then split once
into discovery and confirmation sets. The default split is 50/50.

The exact assignment is saved to `split_assignments.csv`. Within one run this
prevents selection from leaking into confirmation. It cannot retroactively make
previously inspected CounterFact rows confirmatory: because this method was
developed after looking at earlier CounterFact traces, a strong confirmatory
claim requires fact indices or a dataset that were not used during method
development.

The decision rule has only two steps:

1. On discovery facts, choose the full-width window with the largest mean
   fact-level indirect effect. Ties choose the lower center deterministically.
2. On confirmation facts, compute a percentile bootstrap confidence interval
   for that exact window. Select it only when the lower bound is above zero.

Confirmation never chooses another center, changes a region boundary, or ranks
discovery candidates. This makes the reported interval a held-out check of a
decision made without confirmation outcomes. At least two confirmation facts
are required; meaningful production runs should use far more than two.

The confirmation test establishes evidence for positive recovery at the chosen
window. It does not establish that this window is significantly better than all
other windows.

## Post-Hoc Audit

The notebook's earlier selection policy was not suitable as a confirmatory
procedure. In particular, the middle-layer restriction was introduced after a
Mistral v0.3 trace selected center 5. That is a post-hoc exclusion, even if it
was motivated by a valid warning about mapping a broad trace window to one ROME
edit layer.

Latium does not port the following selection heuristics:

| Removed item | Reason |
|---|---|
| Middle 25-75% center band | Added after observing an inconvenient early result; it can exclude legitimate early causal sites. |
| 90% near-peak threshold | Arbitrary scale-dependent region boundary. |
| Minimum three adjacent centers | Adjacent windows share nearly all intervened layers and are not independent support. |
| Neighbor-radius scores | Re-smooth the same overlapping curve without new evidence. |
| Median, trimmed mean, and positive-fact-rate ranking | Multiple interchangeable rankings increase researcher degrees of freedom. |
| Noninferiority gates against the raw peak | Adds a second bootstrap decision without changing the causal estimand. |
| Confirmation-set re-ranking | Uses held-out outcomes to change the selected result, defeating confirmation. |
| Normalized recovery for selection | Unstable when the corruption effect denominator is small; raw paired probability recovery is the estimand. |
| Config-layer fallback in trace selection | Would bias an intended independent check toward the existing configuration. |

Median effects, local plateaus, normalized recovery, and configured ROME layers
may still be useful for explicitly labeled exploratory plots. They do not decide
the Latium result.

## Config Layer And ROME

`model.layer` appears only as a reference line and summary value. It is never
read by the selection function.

Causal tracing and ROME evaluation remain separate:

- causal tracing tests probability recovery under an activation intervention;
- a second-moment matrix is an input to ROME at a particular edit layer;
- a ROME benchmark measures efficacy, paraphrase generalization, and locality.

Therefore a trace window center must not silently replace `model.layer`. Any
trace-to-edit mapping requires its own declared rule and editing benchmark.

## Outputs

Each run writes a timestamped directory below
`analysis_out/causal_trace/` containing:

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

`selection.json` records the discovery center, its physical layer window, both
split sizes, discovery and confirmation means, the held-out confidence interval,
and whether confirmation passed. `summary.json` repeats the main decision and
states that the configured layer was not used for selection.

## Command

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.num_valid_facts=100
```

Important pre-run settings are:

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

Changing these after inspecting a result makes the next run exploratory unless
the new settings are evaluated on fresh held-out facts.

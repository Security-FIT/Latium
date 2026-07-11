# Early-Site Causal Tracing

Latium uses one causal tracing method: paired subject-last MLP-window
restoration with a discovery/confirmation split. The standalone reference is
[`notebooks/causal-tracing.ipynb`](../notebooks/causal-tracing.ipynb); the same
method is exposed by `python3 -m src causal-trace` for repeatable batch runs.

This method is derived from causal tracing in ROME, but it is not a copy of the
original notebook. It adds architecture-specific MLP-output adapters, paired
multi-sample corruption, equal-width window eligibility, deterministic
selection, and held-out confirmation. Old Latium/Aquin variants and the former
post-hoc multi-heuristic selector are not supported.

## Scientific question

For a factual prompt such as:

```text
Danielle Darrieux was born in
```

the trace asks:

> Which fixed-width set of MLP outputs, restored at the last token of the
> subject, recovers probability assigned to the expected next token after the
> subject representation is corrupted?

For fact (f), corruption sample (k), and MLP window (w), Latium measures
the paired indirect effect

```text
IE[f,k,w] = P(target | corrupt[f,k] + restore-clean[f,w])
              - P(target | corrupt[f,k]).
```

The exact same noise tensor is used in both terms. Noise samples are averaged
inside a fact, producing one value `IE[f,w]` per fact and window. Facts—not
noise draws—are therefore the independent units used for aggregation and
uncertainty.

This quantity is intervention-specific probability recovery. It is not a
natural indirect effect, proof that a fact is stored in one matrix, or an
editing-success measurement.

## Three executions per fact

1. **Clean execution.** Run the unmodified prompt and record the expected first
   target token's probability and the model's top next token.
2. **Corrupted execution.** Add Gaussian noise to the embedding output at every
   model-input token overlapping the subject.
3. **Restored executions.** Repeat the corrupted execution while replacing the
   clean final-MLP-projection output at the last subject token for every layer in
   one sliding window.

The clean MLP value is cached once per layer. Corrupted and restored executions
batch several noise samples, but sample `k` in every restored batch is compared
only with corrupted sample `k`.

## Why the last subject token and MLP windows?

ROME's causal analysis found an early factual-recall site in middle-layer MLPs
at the last subject token. A subject such as “Space Needle” is integrated over
multiple tokens; its last token is the point where the complete subject is
available before subsequent prompt tokens consume that representation.

Restoring a single component can underestimate a computation distributed over
neighboring layers. Latium therefore sweeps overlapping windows (ten layers by
default), matching the component-window analysis used by ROME. A center denotes
the complete physical layer list saved in `trace_window_layers`; the center is
only a coordinate and does not imply that one layer alone caused the effect.

Boundary windows are clipped and contain fewer interventions. They are saved
and plotted, but cannot win selection because comparing a smaller boundary
intervention against full-width interior interventions is not like-for-like.

## What is restored

Latium hooks the final projection of each feed-forward/MLP block:

| Architecture family | Restored module output |
|---|---|
| GPT-2 | `mlp.c_proj` |
| GPT-J | `mlp.fc_out` |
| LLaMA, Mistral, Qwen, DeepSeek | `mlp.down_proj` |
| OPT | `fc2` |
| Falcon | `dense_4h_to_h` |
| Granite 4 | `shared_mlp.output_linear` |

This identifies the MLP contribution directly and avoids ambiguity in the
layer offset and residual-stream semantics of a model's `hidden_states` tuple.
Adapters accept both `[batch, sequence, hidden]` and flattened
`[batch * sequence, hidden]` module outputs. Unsupported shapes fail rather than
silently patching the wrong axis.

## Subject and target tokens

The subject string must occur exactly once in the formatted prompt. Fast
tokenizers use character-offset mappings to locate all overlapping subject
tokens in the actual model input. A token-ID fallback retains prompt special
tokens so indices remain aligned when offset mappings are unavailable.

CounterFact targets use continuation tokenization: Latium first tries a
space-prefixed target without special tokens, then explicit fallbacks. The trace
scores the first target token, as in the original next-token causal-tracing
estimand. The full target and token count are recorded so multi-token cases are
not mistaken for full-string evaluation.

## Corruption scale

For all candidate examples scanned in a model run, Latium collects the
embedding rows of the subject tokens at their actual prompt positions. Before
any clean, corrupt, or restoration outcome is inspected, it fixes

```text
sigma_subject = std(candidate subject-token embedding scalars)
noise_std     = noise_multiplier * sigma_subject
```

The default multiplier is `3.0`. Repeated subject tokens retain their
multiplicity. Weighted moments compute the same sample standard deviation
without retaining all embedding vectors in memory.

This follows the ROME protocol's subject-embedding calibration more closely
than taking the standard deviation of the entire vocabulary matrix. The
calibration source, subject count, token count, number of unique token IDs,
embedding standard deviation, multiplier, and final noise standard deviation
are saved in the run summary. Calibration is fixed before observing layer
scores, so it cannot adapt to a favorable window.

Ten independently generated noise tensors are used per fact by default. They
reduce Monte Carlo noise; they do not turn one factual prompt into ten
independent observations.

## Predeclared fact population

A candidate fact enters selection only if:

1. the model's clean top next token equals the expected first target token; and
2. corruption reduces that token's probability by at least `min_total_effect`
   (default `0.03`).

This filter is evaluated without consulting the winning restoration window. It
defines the estimand as recovery among correctly recalled,
corruption-responsive facts. A result must not be generalized to all
CounterFact rows without this qualification. Rejection reasons are saved.

## Discovery and held-out confirmation

Accepted facts are permuted once with the configured seed and split into
discovery and confirmation sets (50/50 by default). The exact assignment is
saved.

The decision rule is deliberately short:

1. Among full-width windows, discovery selects the largest mean fact-level
   indirect effect. Exact ties select the lower center.
2. Confirmation evaluates that exact center. A percentile bootstrap resamples
   confirmation facts and forms a confidence interval for mean indirect effect.
   The center is reported as selected only when the lower interval bound is
   strictly positive and the minimum confirmation count is met.

Confirmation never re-ranks centers, expands a region, substitutes a neighbor,
or falls back to the configured edit layer. Consequently, a failed confirmation
produces no selected trace center; the discovery center remains diagnostic.

The interval supports the claim “the preselected window has positive recovery
on these held-out facts.” It does not prove that this window is better than
every alternative. The configured minimum of two confirmation facts is a
technical floor for small tests, not a recommended scientific sample size.

## Why the previous selector was rejected

The older exploratory notebook combined a middle-layer restriction, near-peak
thresholds, adjacency requirements, neighbor smoothing, alternative summary
statistics, noninferiority gates, and confirmation-set adaptation. These rules
created multiple interchangeable ways to obtain a preferred layer. In
particular, a middle-band restriction was introduced after inspecting an early
Mistral result.

The current selector removes those degrees of freedom. Median effects,
normalized recovery, and local plateaus may still appear in explicitly labeled
diagnostics, but none decides the reported result.

## Relationship to ROME editing

`model.layer` is displayed only as a graph reference and is never an input to
selection. Causal tracing asks whether an activation intervention restores a
prediction; ROME changes one weight matrix and is judged by efficacy,
paraphrase generalization, and locality. A broad trace window's center must not
silently replace an edit layer.

This separation is important because localization and editability are related
but not equivalent; empirical work has found that causal-localization scores do
not reliably predict the best editing location. Any trace-to-edit mapping must
be declared separately and validated with an editing benchmark.

## Reproducibility and outputs

The following settings must be fixed before a reported run:

```text
num_valid_facts
max_dataset_examples_to_scan
num_noise_samples
noise_batch_size
noise_multiplier
window_size
require_correct_clean_prediction
min_total_effect
discovery_fraction
minimum_confirmation_facts
bootstrap_samples
confidence_level
seed
```

Changing them after inspecting a result makes the next run exploratory unless
it is evaluated on fresh held-out facts.

The CLI writes a timestamped directory under `analysis_out/causal_trace/`:

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

`fact_results.jsonl` retains paired corruption and restoration probabilities so
the indirect effects can be audited. `selection.json` records the one discovery
decision and its held-out result. `summary.json` records the method, physical
window, filtering counts, calibration, and the fact that the configured layer
was not used.

## Run it

Standalone notebook:

```text
notebooks/causal-tracing.ipynb
```

Pipeline:

```bash
python3 -m src causal-trace \
  model=gpt2-xl \
  command.causal_trace.num_valid_facts=100
```

The notebook defaults to one model and is intentionally editable. It contains
all model loading, token alignment, hooks, statistics, plots, and persistence;
it does not import Latium. The pipeline is preferred for scheduled runs and the
notebook for direct inspection and experiments.

## Limits of the claim

- The trace evaluates first-token probability, not full object-string
  likelihood.
- Gaussian embedding corruption is an artificial intervention and can move the
  model off its normal activation distribution.
- CounterFact rows inspected during method development are not genuinely fresh
  confirmation data, even if re-split by code. Strong confirmatory claims need
  untouched facts or another dataset.
- Overlapping windows share most layers, so their curve points are highly
  dependent.
- Results are conditional on model version, tokenizer, precision, window size,
  corruption calibration, and accepted-fact filter.
- A selected window localizes recovery under this intervention; it does not
  establish unique storage or prescribe a ROME edit layer.

## Primary references

- Meng et al., [Locating and Editing Factual Associations in
  GPT](https://arxiv.org/abs/2202.05262), NeurIPS 2022.
- The authors' [ROME causal tracing implementation and
  notebook](https://github.com/kmeng01/rome).
- Hase et al., [Does Localization Inform Editing? Surprising Differences in
  Causality-Based Localization vs. Knowledge Editing in Language
  Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/3927bbdcf0e8d1fa8aa23c26f358a281-Paper-Conference.pdf),
  NeurIPS 2023.

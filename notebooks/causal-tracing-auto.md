# causal-tracing-auto.ipynb

This notebook is the self-contained causal tracing workflow for early-site ROME analysis. It does not read the project model config files at runtime. Model aliases, Hugging Face IDs, trace settings, graph-only reference layers, covariance settings, and ROME benchmark settings are defined inside the notebook so it can be moved to another machine or Colab-style runtime.

The notebook answers three separate questions:

1. Where does an early MLP-window intervention causally restore a factual next-token prediction?
2. If that trace-selected region differs from the existing configured ROME layer, what second-moment matrix is needed for the newly selected layer?
3. How does ROME actually perform when evaluated on edits at the selected layer?

The first answer is a causal-tracing result. The third answer is an editing result. They are intentionally not treated as the same thing.

## Core Idea

The trace follows the ROME early-site hypothesis:

```text
subject last token
+ MLP output
+ middle-layer window
```

For a factual prompt such as:

```text
Danielle Darrieux was born in
```

the model predicts a next token. The trace measures how much the expected target-token probability can be recovered when the subject representation has been damaged and a clean internal MLP computation is restored.

The intervention has three passes:

1. Clean pass:
   Run the normal prompt and record the target-token probability.

2. Corrupted pass:
   Add Gaussian noise to the embedding vectors for all tokens belonging to the subject span. The rest of the prompt is left unchanged.

3. Restored pass:
   Reuse the same corruption noise, but patch one clean MLP-window activation back into the corrupted run at the last subject token.

The causal quantity is the paired recovery:

```text
indirect effect = P(target | corrupted + restored) - P(target | corrupted)
```

Because the same noise sample is used for the corrupted and restored run, this is a paired comparison. The paired design matters: it reduces variance and prevents a restoration run from being compared against an unrelated corruption sample.

## What Gets Corrupted

Only the subject-token embeddings are corrupted.

The notebook first identifies the full subject-token span in the actual tokenized prompt. This matters because subjects can be split into multiple subword tokens:

```text
Autonomous University of Madrid
```

may occupy several token positions. The corruption is applied to every token in that span, not only the first or last token.

The noise scale is tied to the model embedding scale:

```text
noise std = 3 * embedding std
```

This avoids an arbitrary absolute noise value such as `0.1`, which can mean very different things across model families.

## What Gets Restored

The notebook restores the output of the model's MLP projection module, not a hidden state from the Hugging Face `hidden_states` tuple.

This distinction is important:

- `hidden_states` are usually residual-stream states.
- ROME early-site tracing is about MLP-mediated factual recall at the subject token.
- Hugging Face hidden-state indexing often includes the embedding state, which can introduce an off-by-one layer mistake.

The notebook hooks the same MLP output module during the clean pass and during the restoration pass. For GPT-2-style models this is the final MLP projection. For LLaMA/Mistral/Qwen-style models this is the down projection. For OPT/Falcon-style models the final feed-forward projection can have different tensor layout, so the notebook validates and patches the correct token row for that architecture.

Only the last subject-token position is restored:

```text
subject_last
```

The final prompt token is not used for early-site selection. Final-token restoration often finds late-layer output recovery, which is useful diagnostically but should not be interpreted as the ROME early MLP site.

## Why Windows Are Used

The notebook restores windows of adjacent MLP layers instead of one MLP layer at a time.

The original ROME causal tracing analysis used MLP windows because individual component effects can be weak and distributed. A window asks:

```text
If I restore this local block of MLP computations at the subject token,
how much target probability comes back?
```

For a 10-layer window centered at layer 17, the physical intervention covers a range of layers around 17. The center is a coordinate for the graph. It is not automatically the exact layer to edit with ROME.

Boundary windows are still plotted, but strict layer selection is restricted to a middle band of full-width windows. This prevents the first full-width window from being selected just because it is the earliest comparable window.

## Fact-Level Aggregation

The trace is not trusted from one prompt.

For each valid factual prompt:

1. the clean model must predict the expected first target token,
2. the subject corruption must reduce the target probability enough to be meaningful,
3. the notebook runs multiple fixed noise samples,
4. the indirect effect is averaged within that fact.

The main uncertainty unit is the fact, not the individual noise sample. This is important because the ten noise samples for the same fact are highly correlated. The notebook averages noise samples inside each fact, then bootstraps across facts.

## Discovery and Confirmation

The notebook avoids choosing a center from the same data and then over-interpreting that choice as statistically confirmed.

It splits valid facts into:

```text
discovery facts
confirmation facts
```

The discovery split finds plausible high-scoring regions. The confirmation split checks whether those preselected regions still have a positive causal effect.

This reduces the failure mode where a noisy local maximum becomes a precise-looking layer recommendation.

## Region Selection

The notebook does not simply take:

```text
argmax(mean indirect effect)
```

The raw maximum is still reported, but it is treated as a diagnostic. A robust region is preferred when neighboring centers show consistent effects and the confirmation split supports the result.

The region decision prioritizes:

1. positive confirmation-set effect,
2. median indirect effect,
3. trimmed mean indirect effect,
4. positive fact rate,
5. neighborhood consistency.

Neighbor support is treated as smoothing, not as independent evidence. Adjacent windows share most of their physical layers, so a plateau is useful but not equivalent to independent repeated confirmation.

## Config Layer Is Only a Reference

The existing configured ROME layer is not used to choose the trace result.

It appears only as:

- a reference line in the graph,
- a comparison value in the summary,
- a downstream skip condition when the graph-selected layer is already the configured layer.

This is deliberate. If the configured layer influenced the trace decision, the graph would not be an independent check of the config.

## Why Mistral v0.3 Previously Looked Weird

Mistral v0.3 previously produced a strict trace center near 5, while the configured ROME layer was 17. The issue was not that causal tracing and ROME necessarily had to agree. The deeper problem was interpretive:

```text
window center 5 with window size 10
```

does not mean:

```text
edit MLP layer 5 with ROME
```

For a 32-layer model, a center-5 window restores roughly the first ten MLP modules. It is a broad window intervention. ROME, however, edits one actual projection matrix. Mapping the window center directly to one edit layer is too crude, especially near the first full-width window.

The notebook now handles this more conservatively:

- it uses more valid facts for Mistral rather than the earlier small cap,
- strict selection must lie in the middle early-site band,
- the graph-selected trace center is kept separate from the edit layer,
- if a selected trace layer evaluates zero ROME cases, the notebook retries the configured reference layer for benchmarking only.

The retry does not change causal-trace selection. It only prevents the final benchmark table from ending with zero evaluated ROME edits when the trace-to-edit mapping fails.

## Covariance / Second Moment

ROME needs a second-moment matrix for the layer being edited. The notebook computes that matrix only when needed.

If the trace-selected layer differs from the existing configured layer, the notebook checks whether the exact second-moment file for the selected model and selected layer already exists. If it exists, it is reused. If it does not exist, the notebook computes and saves:

- the inverse second-moment matrix used by ROME,
- the raw covariance matrix for audit/reuse.

The notebook creates the output directories automatically:

```text
data/second_moment_stats/
data/second_moment_stats/raw_covariance/
```

The covariance is tied to the active edit layer. A covariance file for layer 17 is not reused for layer 5.

## ROME Benchmark

The final benchmark runs actual ROME edits and measures editing behavior.

The reported metrics include:

- efficacy score,
- efficacy magnitude,
- paraphrase score,
- neighborhood score,
- specificity/locality-style score,
- overall score,
- skipped/error counts.

The benchmark tries to collect the requested number of evaluated edits, not merely the requested number of attempted dataset rows. This matters when some edits fail during optimization or evaluation.

If the strict graph-selected layer is the same as the configured reference layer, the expensive covariance recomputation is skipped. If the trace-selected layer differs, the notebook computes/reuses the selected-layer covariance and runs the ROME benchmark for that layer.

## Outputs

Outputs are written under:

```text
analysis_out/causal_tracing_auto/
```

Each model gets a timestamped directory with:

```text
early_site_causal_region_with_config_reference.png
summary_facts.csv
rejections.csv
discovery_windows.csv
confirmation_windows.csv
confirmation_regions.csv
final_selection.json
selection_diagnostics.json
fact_results.jsonl
raw_window_probabilities.npz
rome_layer_validation.csv
```

The batch-level outputs include:

```text
causal_trace_batch_summary_<timestamp>.csv
rome_benchmark_summary_<timestamp>.csv
rome_benchmark_all_<timestamp>.json
full_pipeline_report_<timestamp>.md
```

## Interpretation

The correct interpretation is:

```text
Restoring this subject-last MLP-window region causally recovers part of the
first target-token probability under subject corruption.
```

The incorrect interpretation is:

```text
This center is definitely the exact ROME edit layer.
```

The causal trace localizes a causal contribution under one intervention. The ROME benchmark separately tests whether editing a layer gives good rewrite, paraphrase, and locality behavior.

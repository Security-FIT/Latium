# Causal tracing

Latium measures whether restoring one clean MLP output can recover the first
true-target token after subject embeddings are corrupted. A positive result is
evidence for a causal effect of that intervention. It does not establish where
a fact is stored or which layer is best for ROME editing.

## Fixed experiment

For every accepted CounterFact row, the clean model's most likely next token
must be the first true-target token. Independent Gaussian noise is added to
every subject-token embedding. The standard deviation is three times the
embedding-table standard deviation, computed once per run. A row is accepted
only if corruption lowers the mean probability of the target token. The noise
multiplier may be overridden, but it is fixed for every row in a run.

The same noise draws are used in the corrupt baseline and each restoration:

    IE(f, k, w) = P(target | corrupt(f, k) + restore(f, w))
                  - P(target | corrupt(f, k))

The clean outputs of the enclosing MLP are restored at the last subject token.
For example, the Llama trace hooks `model.layers.N.mlp`, while ROME edits
`model.layers.N.mlp.down_proj`. Ten noise draws are averaged within a fact;
confidence intervals resample facts.

The default window width is **one layer**, so a selected center is a directly
tested MLP layer. `command.causal_trace.window_size` may be increased to study
intervals, but a window center must not be interpreted as an individually
localized layer. Multi-layer windows cannot overwrite a model YAML layer and
cannot feed the automatic ROME pipeline.

## Discovery and confirmation

The run requests 100 accepted facts and scans at most 10,000 rows. Accepted
facts are shuffled with seed 42, then split equally into discovery and
confirmation sets. If fewer than the requested number are accepted, no layer
is selected.

Only discovery facts are swept across all layers. The layer with the largest
mean paired effect is frozen, with the lower layer breaking ties. Confirmation
facts are evaluated **only at this frozen layer**. Selection succeeds when
its 95% fact-bootstrap confidence interval has a positive lower bound. The
default interval uses 1,000 bootstrap resamples.

The configured model layer is a plot reference only. It does not enter
selection. Model YAML changes only with
`command.causal_trace.overwrite_model_config_layer=true` after confirmation.

## Run and outputs

    python3 -m src causal-trace model=gpt2-xl

Each successful run writes `resolved_config.yaml`, `mlp_module_map.json`,
`fact_results.jsonl`, `rejections.csv`, `split_assignments.csv`,
`discovery_windows.csv`, `confirmation_windows.csv`, `selection.json`,
`summary.json`, and `early_site_trace.png`.

Each fact row records the fixed noise standard deviation and seed, its split,
and exactly the window centers evaluated for that split. The summary includes
model and tokenizer revisions when available, the Git commit, model dtype,
the complete hook map, and the selected interval. Only the first target token
is measured.

The notebook `notebooks/causal_tracing.ipynb` invokes this command and reads
its output. `notebooks/causal_tracing_reference.ipynb` is historical reference
material.

## Causal trace to ROME

    jobs/submit.sh causal-rome -- pipeline.model=gpt2-xl

The job requires a directly tested, confirmed single layer. It computes or
finds matching second moments and runs the ROME benchmark at that layer. ROME
case enumeration starts after all CounterFact rows scanned by the trace, so
trace construction and edit evaluation use disjoint facts. The job does not
modify model YAML.

To determine whether the traced layer is the best editing layer, compare it
against neighboring and configured layers on an independent ROME validation
set. Evaluate the chosen editing layer once more on a separate test set. This
editing comparison is a separate experiment from causal tracing:

    python3 jobs/validate_rome_layers.py \
      --model gpt2-xl \
      --trace-summary analysis_out/causal_trace/<run>/summary.json \
      --output-root analysis_out/rome-layer-validation

The validator compares the traced layer, adjacent layers, and the configured
layer on identical CounterFact validation rows. It ranks by the ROME overall
score, reports efficacy, paraphrase and neighborhood metrics, and evaluates
the chosen layer on new test rows. Both sets start after the rows scanned by
causal tracing. Every candidate must evaluate the same requested number of
cases; otherwise the comparison stops.

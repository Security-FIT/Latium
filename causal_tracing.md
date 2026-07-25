# Causal tracing

Latium uses causal tracing to choose a layer for a later ROME run. Production code lives in src/causal_trace/. The notebook calls that code and plots saved artifacts.

## Measurement

For each accepted CounterFact example, Latium measures the probability of the first true-target token in three conditions:

1. Clean: no intervention.
2. Corrupt: add Gaussian noise to every subject-token embedding.
3. Restore: repeat the same corruption and restore clean MLP outputs at the last subject token for one overlapping layer window.

For fact f, noise draw k, and window w:

    IE(f, k, w) = P(target | corrupt(f, k) + restore(f, w))
                  - P(target | corrupt(f, k))

The two terms use the same noise draw. Draws are averaged within each fact. Confidence intervals resample facts, not individual draws.

The hook restores the whole MLP output. On Llama-style models, tracing hooks model.layers.N.mlp; ROME still edits model.layers.N.mlp.down_proj. A forward pass validates all MLP hooks before the dataset scan.

## Default experiment

| Quantity | Default |
|---|---:|
| Accepted facts requested | 100 |
| Maximum rows scanned | 10,000 |
| Noise draws per fact | 10 |
| Noise batch size | 10 |
| Noise standard deviation | 3.0 times embedding-weight standard deviation |
| Restored window width | 10 layers |
| Discovery / confirmation split | 50 / 50 |
| Minimum confirmation facts | 50 |
| Minimum clean-to-corrupt probability drop | 0.03 |
| Maximum corrupt relative standard deviation | 1.0 |
| Bootstrap resamples | 1,000 |
| Confidence level | 95% |
| Trimmed-mean fraction | 10% per tail |
| Neighbor and adjacent-peak radius | 2 centers |
| Required local support | 90% |
| Noninferiority margin | 10% of discovery peak |
| Minimum supported region | 3 centers |
| Seed | 42 |

The clean top prediction must equal the first true-target token. Corruption must reduce its probability by at least 0.03, and:

    std(P_corrupt) / abs(P_clean - mean(P_corrupt)) <= 1.0

These checks happen before restoration windows are evaluated.

## Selection

The accepted facts are shuffled once with seed 42. With 100 facts, discovery and confirmation each receive 50 facts.

1. Evaluate one centered window at every model layer. Boundary windows shorter than 10 layers are saved and plotted but cannot be selected.
2. On discovery facts, calculate the mean, median, 10% trimmed mean, normalized recovery, and a 95% fact-bootstrap interval.
3. Require a positive lower confidence bound, median, and trimmed mean.
4. Form contiguous regions using radius-2 neighborhoods, 90% local support, the radius-2 peak rule, and a 10% noninferiority margin.
5. Test only those regions on confirmation facts. A region passes when its held-out 95% lower confidence bound is above zero.
6. Choose one consistency-ranked center inside the winning confirmed region.

The configured model layer is only a plot marker. It does not affect selection. Model YAML stays unchanged unless command.causal_trace.overwrite_model_config_layer=true is explicitly set. Confirmation failure never changes it.

## Run the trace

    python3 -m src causal-trace model=gpt2-xl command.causal_trace.num_valid_facts=100

Each run writes:

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

summary.json records split counts, the discovery center, confirmed region, selected center, hook semantics, and confirmation result.

## Causal trace to ROME

The detector-free job runs:

    causal trace -> confirmed layer -> matching second moment -> ROME benchmark

The default job uses 100 trace facts and 30 ROME cases:

    jobs/submit.sh causal-rome -- pipeline.model=gpt2-xl

Useful overrides include pipeline.causal_trace.num_valid_facts, pipeline.covariance.target_samples, and pipeline.rome.n_tests.

When pipeline.covariance.target_samples is null, the job reads second_moment_target_samples from the selected pipeline model. If that model has no value, it uses the second-moment command fallback of 100,000 samples. An explicit pipeline override wins.

pipeline.rome.n_tests remains 30 because it is this job's evaluation size. The structural.run.n_tests reference in new-detection-clean belongs to a structural detector run. This job does not compose or run that pipeline, so linking to the setting would add an unrelated dependency and would not be model-adaptive.

The selected layer is passed to covariance and ROME as a runtime override. The job requires a passing trace, a non-empty plot, covariance for the selected layer, and at least one evaluated ROME case before writing pipeline-summary.json.

For llama2-7b, ROME uses the validated fixed context-template pool. It runs after causal selection and cannot affect the selected layer.

## Limits

- Only the first target token is scored.
- Overlapping windows are correlated.
- Results depend on noise scale, window width, fact filtering, restored component, subject position, and model adapter.
- Positive held-out recovery does not prove that a fact is stored in one layer.
- The representative center is an operational ROME layer; the later ROME benchmark tests whether it is useful.

## Files

- src/causal_trace/causal_trace.py: execution and artifacts.
- src/causal_trace/model_adapter.py: whole-MLP hooks.
- src/causal_trace/tokenization.py: subject and target token mapping.
- src/causal_trace/selection.py: aggregation, intervals, and selection.
- src/config/command/causal_trace.yaml: method defaults.
- src/config/pipeline/causal_rome.yaml: job defaults.
- jobs/causal_rome_pipeline.py: trace, covariance, and ROME orchestration.
- notebooks/causal_tracing.ipynb: production runner and artifact plots.
- notebooks/causal_tracing_legacy.ipynb: unchanged origin/causal-trace analysis notebook.

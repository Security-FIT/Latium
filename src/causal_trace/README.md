# Causal Trace

This package has one causal tracing workflow: `causal-trace`.

It corrupts every subject-token embedding, restores clean MLP projection
outputs at the last subject token over overlapping layer windows, and measures
the paired change in first-target-token probability. Noise samples are paired
within facts; uncertainty is estimated across facts.

Selection uses a fixed random discovery/confirmation split. Discovery chooses
the highest-mean full-width window once. Confirmation tests only that window and
does not choose another center. A center is reported as selected only when its
held-out bootstrap confidence interval is above zero.

The result identifies a causal window intervention. Its center is not an exact
ROME edit layer and ROME performance must be evaluated separately.

## CLI

```bash
python3 -m src causal-trace model=gpt2-xl command.causal_trace.num_valid_facts=100
```

The full method, audit decisions, outputs, and limitations are documented in
`notebooks/causal-tracing-auto.md`. The portable cluster notebook is
`notebooks/causal-tracing-auto-v2.ipynb`.

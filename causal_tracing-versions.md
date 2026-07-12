# Causal Tracing Version Status

## Current

- Production: `src/causal_trace/`
- Portable reference: `notebooks/causal-tracing-auto-v2.ipynb`
- Full standalone documentation: `causal_tracing.md`
- Notebook-oriented audit notes: `notebooks/causal-tracing-auto.md`

The current workflow corrupts all subject-token embeddings, restores final MLP
outputs at the last subject token over overlapping windows, uses paired noise
samples, selects one full-width window on discovery facts, and tests only that
window on held-out confirmation facts.

## Historical

`notebooks/causal-tracing-auto.ipynb` established the original self-contained
experiment but retains exploratory selection heuristics. Earlier source
“standard” and “alternative” workflows used single-token or all-subject-token
curves and did not implement the current held-out selection rule.

Historical outputs are not directly comparable to current selections unless
the intervention, window width, noise settings, fact filter, target-token rule,
and selector are all matched.

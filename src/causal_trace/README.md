# Causal Trace

This package exposes one active causal tracing workflow for ROME early-site
diagnostics.

| Command | Workflow |
|---|---|
| `causal-trace` | Subject-last MLP-window trace with paired noise and fact-level aggregation. |

The trace corrupts all subject-token embeddings, restores clean MLP output at
the final subject token over overlapping layer windows, and measures the paired
indirect effect on the first target token. It reports a causal trace region,
not a validated ROME edit layer.

Visual notebook:

- `notebooks/causal-tracing-auto.ipynb`

## CLI

```bash
python3 -m src causal-trace model=gpt2-xl command.causal_trace.num_valid_facts=100
```

Useful overrides:

```bash
python3 -m src causal-trace model=mistral-7b-v0.3 command.causal_trace.window_size=10 command.causal_trace.num_noise_samples=10
```

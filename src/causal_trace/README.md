# Causal Trace

This package has two causal tracing workflows.

| File | Workflow |
|---|---|
| `causal_trace.py` | Standard trace: corrupt subject embeddings, restore each subject token at each layer, write CSV rows. |
| `alt_trace.py` | Alternative trace: restore the last subject token, average repeated noise runs, rank layers with middle-third fallback. |
| `layer_heuristic.py` | Layer scoring helpers that can consume causal trace CSVs. |

Visual notebooks:

- `notebooks/causal_tracing.ipynb` for the standard workflow.
- `notebooks/causal_tracing_alt.ipynb` for the alternative workflow.

Use matching `MODEL_CONFIG`, prompt counts, and dataset settings when comparing
the two notebooks.

## CLI

```bash
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5 generation.num_trace_runs=10
```

## Adding A Variant

Keep shared prompt preprocessing in `causal_trace.py`. Put variant-specific
tracing and persistence in a new module, add a command handler in
`command_handlers/operations.py`, and add a command config under
`src/config/command/`.

# Causal Trace

This package keeps the two existing causal tracing workflows and adds three
prototype workflows through `prototype.py`.

| Command | Workflow |
|---|---|
| `causal-trace` | Existing standard-style trace: restore subject-token positions across layers. |
| `alt-trace` | Existing alternative-style trace: one restore position, ranking, optional middle-third fallback. |
| `aquin-trace` | Pragmatic ROME candidate trace with optional ROME validation. |
| `canonical-trace` | Token-by-layer ROME-style causal tracing, including residual/MLP/attention modes. |
| `fast-trace` | Short paired-noise trace, one restore position, no middle-third fallback. |

All five modes use paired fixed noise samples, explicit first-token target
probabilities, safe hook cleanup, and structured outputs under
`analysis_out/causal_trace/`.

Visual notebooks:

- `notebooks/causal_trace_standard.ipynb`
- `notebooks/causal_trace_alt.ipynb`
- `notebooks/causal_trace_aquin.ipynb`
- `notebooks/causal_trace_canonical.ipynb`
- `notebooks/causal_trace_fast.ipynb`

Legacy notebooks `notebooks/causal_tracing.ipynb` and
`notebooks/causal_tracing_alt.ipynb` are still present for comparison.

## CLI

```bash
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5
python3 -m src aquin-trace model=gpt2-large generation.num_of_runs=5
python3 -m src canonical-trace model=gpt2-large generation.num_of_runs=5
python3 -m src fast-trace model=gpt2-large generation.num_of_runs=5
```

Useful overrides:

```bash
python3 -m src canonical-trace model=gpt2-large tracing.component=mlp tracing.position_scope=subject_last tracing.window_size=10
python3 -m src canonical-trace model=gpt2-large tracing.component=attention tracing.position_scope=prompt_last tracing.window_size=10
python3 -m src fast-trace model=gpt2-large tracing.restore_position=subject_last
python3 -m src aquin-trace model=gpt2-large tracing.validate_with_rome=false
```

# Causal Tracing Implementations: Standard vs Alternative

This document summarizes the differences between the two causal tracing
workflows in `src/causal_trace/`.

| Aspect | Standard (`causal_trace.py`) | Alternative (`alt_trace.py`) |
|---|---|---|
| Main entry point | `causal_trace(cfg)` | `run_alt_trace(cfg)` |
| Hydra command | `causal-trace` | `alt-trace` |
| Per-prompt restoration target | **All** subject token positions | **Only the last** subject token position |
| Noise samples per prompt | Single corruption sample | `num_trace_runs` independent samples (default 10) |
| Per-layer probability | Single-run value | Mean across `num_trace_runs` noise samples |
| Noise multiplier | Uses `handler._noise_multiplier` computed by `compute_embedding_std()` | Auto-computes `3x` embedding std; falls back to `0.1`; respects configured multiplier unless `auto_compute_multiplier` is true |
| Built-in layer ranking | No — produces raw CSV rows | Yes — `select_layers()` ranks all layers and falls back to the middle third when the signal is noisy |
| Output artifacts | One timestamped CSV per run | `analysis_out/<model>_alt_<timestamp>.csv` + `analysis_out/<model>_alt_selection_<timestamp>.json` |
| Return value | `None` | `Optional[LayerSelection]` with the recommended layer and ranked candidates |
| Data structures | Ad-hoc tuples/lists | `TraceResult`, `LayerCandidate`, `LayerSelection` dataclasses |
| Prompt failure handling | Returns `1` and increments a failure counter | Returns `None` and skips the prompt |

---

## 1. Restoration target tokens

- **Standard**: For every subject token position, it restores the clean
  hidden-state activation at each layer. This produces a restoration curve
  per subject token.
- **Alternative**: It restores **only** the last subject token at each layer,
  reducing the output to a single per-layer curve per prompt.

## 2. Noise sampling and averaging

- **Standard**: One corrupted run is performed per prompt.
- **Alternative**: `num_trace_runs` (default 10, configurable via
  `generation.num_trace_runs`) fresh Gaussian noise samples are drawn per
  prompt, and the resulting per-layer restoration probabilities are averaged.

## 3. Noise scale / multiplier computation

- **Standard**: The noise multiplier comes from the model handler
  (`handler._noise_multiplier`), usually computed by `compute_embedding_std()`.
  The `compute_multiplier` helper exposes this value as a standalone command.
- **Alternative**: The docstring states noise is applied at `3.0 x embedding
  std`. The helper `_ensure_noise_multiplier()` auto-computes the multiplier
  if it is missing or if `cfg.model.auto_compute_multiplier` is set; if no
  data is available, it falls back to `0.1`.

## 4. Output format

- **Standard CSV columns** (written by `save_results_to_csv`):
  `run_number`, `prompt_num`, `clean`, `corrupted`, `restored_token`,
  `restored`. The `restored` column stores a Python literal list of
  `(decoded_token, probability)` tuples for each layer.
- **Alternative CSV columns**: `prompt_idx`, `subject`, `target`,
  `clean_prob`, `corrupt_prob`, then one column per layer
  (`layer_0` ... `layer_N`) with the averaged restoration probability.
- **Alternative JSON output**: `LayerSelection` serialized to JSON, including
  `best_layer`, `signal_quality`, `used_middle_third_fallback`, and the full
  ranked candidate list.

## 5. Layer ranking and signal quality

- **Standard**: No ranking is performed; downstream tools or notebooks must
  parse the CSV and decide which layer to use.
- **Alternative**: Implements `select_layers()` which:
  1. Ranks layers by average restoration probability.
  2. Detects noisy/flat signals using coefficient of variation, peak-to-mean
     ratio, and peak-layer position.
  3. Falls back to the **middle third** of layers when the signal is noisy,
     ranking only within that band.

The thresholds are:

```python
_NOISE_CV_THRESHOLD = 0.15
_NOISE_PEAK_RATIO_THRESHOLD = 0.3
```

## 6. Code reuse

The alternative workflow reuses the standard workflow's prompt preprocessing:

```python
from src.causal_trace.causal_trace import filter_dataset, preprocess_prompt
```

Everything else in `alt_trace.py` is variant-specific (tracing, ranking,
persistence).

## 7. CLI usage examples

```bash
# Standard
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5

# Alternative
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5 generation.num_trace_runs=10
```

Both commands share the same model/dataset defaults defined in their
respective `src/config/command/*.yaml` files.

## 8. Notebooks

- `notebooks/causal_tracing.ipynb` — visualizes the **standard** workflow.
- `notebooks/causal_tracing_alt.ipynb` — visualizes the **alternative**
  workflow.

Use matching `MODEL_CONFIG` and prompt counts when comparing them.

# Causal Tracing Implementations: Standard vs Alternative

Comparison of the two causal tracing workflows in `src/causal_trace/`,
covering both the Python drivers and their companion notebooks.

- **Standard**: `src/causal_trace/causal_trace.py` + `notebooks/causal_tracing.ipynb`
- **Alternative**: `src/causal_trace/alt_trace.py` + `notebooks/causal_tracing_alt.ipynb`

## Summary Table

| Aspect | Standard (`causal_trace.py`) | Alternative (`alt_trace.py`) |
|---|---|---|
| Top-level driver | `causal_trace(cfg)` | `run_alt_trace(cfg)` |
| Hydra command | `causal-trace` | `alt-trace` |
| Per-prompt restoration target | **All** subject token positions | **Only the last** subject token position |
| Noise samples per prompt | Single corruption sample | `num_trace_runs` independent samples (default 10) |
| Per-layer probability | Single-run value | Mean across `num_trace_runs` noise samples |
| Clean hidden-state lookup | Re-read from `outputs_clean` inside the layer loop | Pre-cached into a list before the run loop |
| Corrupted-run timing | Before restoration runs | After all restoration runs |
| Noise multiplier | Uses `handler._noise_multiplier` from `compute_embedding_std()` | `_ensure_noise_multiplier()`: auto-computes 3x embedding std, respects configured value, falls back to `0.1` |
| Built-in layer ranking | No | Yes — `select_layers()` with middle-third fallback |
| Signal quality detection | No | Yes — `_signal_is_noisy()` (CV, peak ratio, peak position) |
| Output artifacts | One timestamped CSV (nested tuples) | Wide CSV (one column per layer) + selection JSON |
| Return value | `int` (0 success / 1 clean-token mismatch) | `Optional[LayerSelection]` with ranked candidates |
| Data structures | Ad-hoc tuples/lists | `TraceResult`, `LayerCandidate`, `LayerSelection` dataclasses |
| Prompt failure handling | Returns `1`, increments failure counter | Returns `None`, skips prompt |

---

## 1. Restoration Target Tokens

- **Standard**: Iterates over every subject token position. For each
  position it restores the clean hidden state at every layer, producing a
  2D restoration matrix of shape `(num_subject_tokens, num_layers)`.
- **Alternative**: Restores only `subject_positions[-1]` (the last subject
  token) at every layer, producing a single 1D per-layer curve per prompt.

## 2. Noise Sampling And Averaging

- **Standard**: One corrupted run per prompt; the restoration probability
  at each layer comes from a single noise draw.
- **Alternative**: `num_trace_runs` (default 10, configurable via
  `generation.num_trace_runs`) fresh Gaussian noise samples are drawn per
  prompt. Each run performs a full corrupted-and-restored sweep, and the
  per-layer probabilities are averaged with `np.mean(all_run_probs, axis=0)`.

## 3. Clean Hidden-State Caching

- **Standard**: Reads the restoration source inside the layer loop:
  `outputs_clean["hidden_states"][restore_layer + 1][0][restore_token_idx, :]`.
- **Alternative**: Pre-computes the cache once before the run loop:
  `clean_hidden_at_last = [outputs_clean["hidden_states"][l + 1][0][last_subject_token, :] for l in range(num_layers)]`,
  then indexes the list inside the loop. This avoids re-walking the
  hidden-states tuple on every layer of every noise run.

## 4. Corrupted-Run Timing

- **Standard**: Runs the corrupted forward pass **before** the restoration
  sweeps, so `corrupt_prob` and `corrupt_token` are available up front.
- **Alternative**: Runs the corrupted forward pass **after** all restoration
  runs (solely to record `corrupt_prob`). The restoration sweeps set up
  their own corruption hooks independently per run.

## 5. Hook Management

- **Standard**: The corrupt hook is set once per outer subject-token loop;
  the restore hook is set and unset for each layer; corrupt hooks are
  removed at the end of the token loop. The notebook version wraps each
  pass in `try/finally` to guarantee hook cleanup.
- **Alternative**: Per noise run, the corrupt hook is set once, then the
  restore hook is set/unset per layer, then the corrupt hook is removed.
  No `try/finally` guard is used in `trace_prompt`.

## 6. Gradient Context

- **Standard notebook** wraps the trace in `with torch.no_grad():`.
- **Alternative** (`alt_trace.py` and its notebook) does not use an explicit
  `torch.no_grad()` context; it relies on the model being in eval mode.

## 7. Noise Multiplier Computation

- **Standard**: The multiplier comes from `handler._noise_multiplier`,
  typically computed by `compute_embedding_std()`. A standalone
  `compute_multiplier(cfg)` helper exposes this value as a CLI command.
- **Alternative**: `_ensure_noise_multiplier(handler, cfg, df_dataset)`
  - Respects a configured `corruption_noise_multiplier` unless
    `cfg.model.auto_compute_multiplier` is set.
  - Auto-computes `3x` embedding std via `compute_embedding_std()`.
  - Falls back to `0.1` if no data is available.

## 8. Layer Ranking And Signal Quality (Alt Only)

`select_layers(per_layer_probs, num_layers)`:

1. Calls `_signal_is_noisy(probs)`, which returns `True` when any hold:
   - Coefficient of variation `< 0.15` (flat curve).
   - Peak-to-mean ratio `< 0.3` (weak peak).
   - Peak layer outside the middle third `[num_layers // 3, 2 * num_layers // 3]`.
2. If noisy, restricts the candidate pool to the **middle third** and ranks
   within that band (`used_middle_third_fallback = True`).
3. If clean, ranks all layers.
4. Returns a `LayerSelection` with `best_layer`, ranked `LayerCandidate`s,
   `signal_quality` (`"clean"`/`"noisy"`), and `avg_per_layer_probs`.

Threshold constants:

```python
_NOISE_CV_THRESHOLD = 0.15
_NOISE_PEAK_RATIO_THRESHOLD = 0.3
```

The standard workflow performs no ranking; downstream tools or notebooks
must parse the CSV and pick a layer themselves.

## 9. Output Format

- **Standard CSV** (written by `save_results_to_csv`):
  Columns `run_number`, `prompt_num`, `clean`, `corrupted`,
  `restored_token`, `restored`. The `restored` column stores a Python
  literal list of `(decoded_token, probability)` tuples, one per layer.
  Filename is `<model>_<timestamp>.csv`.
- **Alternative CSV** (written by `_save_results`):
  Columns `prompt_idx`, `subject`, `target`, `clean_prob`,
  `corrupt_prob`, then one column per layer (`layer_0` ... `layer_N`)
  with the averaged restoration probability. Wide format.
  Filename is `analysis_out/<model>_alt_<timestamp>.csv`.
- **Alternative JSON**: `analysis_out/<model>_alt_selection_<timestamp>.json`
  serializes the `LayerSelection` — `best_layer`, `signal_quality`,
  `used_middle_third_fallback`, ranked candidates, and
  `avg_per_layer_probs`.

## 10. Data Structures

- **Standard**: Returns ad-hoc tuples appended to a list and written to
  CSV; no dedicated result dataclass.
- **Alternative**: Typed dataclasses:
  - `TraceResult` — `prompt_idx`, `subject`, `target`, `clean_prob`,
    `corrupt_prob`, `per_layer_probs`. Has `to_dict()`.
  - `LayerCandidate` — `layer`, `rank`, `restoration_prob`,
    `in_middle_third`.
  - `LayerSelection` — `best_layer`, `candidates`,
    `used_middle_third_fallback`, `signal_quality`,
    `avg_per_layer_probs`. Has `summary()` and `to_dict()`.

## 11. Failure Handling

- **Standard**: `causal_trace_single_run` returns `1` when the clean run
  does not produce the target token; the caller increments a failure
  counter and continues.
- **Alternative**: `trace_prompt` returns `None` on the same condition;
  the caller skips the prompt.

## 12. Code Reuse

The alternative workflow reuses the standard workflow's prompt
preprocessing and dataset filtering:

```python
from src.causal_trace.causal_trace import filter_dataset, preprocess_prompt
```

Everything else in `alt_trace.py` is variant-specific (tracing, ranking,
persistence, noise-multiplier handling).

---

# Notebook Differences

Both notebooks share the same setup cell (project-root discovery, imports,
Hydra config composition) and the same `MODEL_CONFIG` / `NUM_PROMPTS`
knobs. They diverge as follows.

## N1. Trace Helper

- **Standard notebook** (`causal_tracing.ipynb`) defines an inline
  `trace_prompt_standard()` helper plus a `StandardTraceResult` dataclass.
  It replicates `causal_trace_single_run` logic but returns in-memory
  matrices instead of appending CSV rows, and wraps everything in
  `torch.no_grad()` with `try/finally` hook cleanup.
- **Alt notebook** (`causal_tracing_alt.ipynb`) imports the production
  functions directly:
  `trace_prompt`, `select_layers`, `_ensure_noise_multiplier`,
  `_save_results` from `src.causal_trace.alt_trace`. No inline tracing
  logic.

## N2. Config Overrides

- Standard: `command=causal_trace`, `generation.num_of_runs`.
- Alt: `command=alt_trace`, `generation.num_of_runs`,
  `generation.num_trace_runs`.

## N3. Noise Multiplier Resolution

- Standard notebook only prints
  `cfg.model.get("corruption_noise_multiplier", "auto")`; it does not call
  `_ensure_noise_multiplier`, so the handler's default path applies.
- Alt notebook explicitly calls
  `_ensure_noise_multiplier(handler, cfg, df_dataset)` and prints the
  resolved `handler._noise_multiplier`.

## N4. Result Shape

- Standard: `StandardTraceResult.restoration` is a
  `dict[int, np.ndarray]` keyed by subject token position. The `matrix`
  property stacks these into a 2D array; `layer_mean` averages over
  subject tokens.
- Alt: `TraceResult.per_layer_probs` is a 1D `np.ndarray` (one value per
  layer), already averaged over noise runs.

## N5. Prompt Gallery Visualization

- Standard: one row per prompt with **two** columns — a token-position x
  layer heatmap (`viridis`) with a white dashed peak line, and a mean
  layer curve with clean/corrupt reference lines and a crimson dashed
  peak line. Y-axis ticks show subject token positions.
- Alt: a grid of single plots (no per-prompt heatmap). Each plot is the
  per-layer probability curve with clean/corrupt reference lines and a
  peak marker. Title includes the `NUM_TRACE_RUNS` count.

## N6. Run-Level Graphs

- Standard: prompt x layer heatmap (`magma`) with white `x` peak markers,
  plus an average curve with `1 std` fill band and a crimson dashed peak
  line.
- Alt: same heatmap + average curve, **plus**:
  - An orange `axvspan` marking the middle-third band.
  - A darkgreen dash-dot line at `selection.best_layer`.
  - A title reporting `quality` and `used_middle_third_fallback`.
  - A `print(selection.summary())` call beneath the figure.

## N7. Summary Tables

- Standard summary columns: `prompt_idx`, `subject`, `target`,
  `clean_prob`, `corrupt_prob`, `peak_layer`, `peak_prob`,
  `subject_positions`.
- Alt summary columns: `prompt_idx`, `subject`, `target`, `clean_prob`,
  `corrupt_prob`, `peak_layer`, `peak_prob` (no `subject_positions`
  since only the last token is traced). Additionally produces a
  `candidates` table with `rank`, `layer`, `restoration_prob`,
  `in_middle_third` for the top 10 ranked layers.

## N8. Save Behavior

- Standard: writes `summary` CSV and an `avg_curve` CSV to
  `analysis_out/causal_trace_notebook/<model>_standard_*.csv`.
- Alt: calls `_save_results(results, avg_probs, selection, cfg)`, writing
  the per-prompt CSV and the selection JSON to `analysis_out/`.

## N9. Section Layout

- Standard (7 sections): Setup; Load Config, Model, And Dataset; Standard
  Trace Helper; Run Traces; Prompt Gallery; Run-Level Graphs; Summary
  And Optional Save.
- Alt (6 sections): Setup; Load Config, Model, And Dataset; Run Alt
  Traces; Prompt Gallery; Run-Level Graphs And Layer Selection;
  Candidate Table And Optional Save.

---

# CLI Usage

```bash
# Standard
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5

# Alternative
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5 generation.num_trace_runs=10
```

Both commands share the same model/dataset defaults. Use matching
`MODEL_CONFIG` and prompt counts when comparing the notebooks.

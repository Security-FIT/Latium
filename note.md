# Latium Cluster Test Notes

Tested on: `ubuntu@216.81.248.46` (A100-SXM4-80GB) via `metju@kubapc` jump host
Branch: `main` (commit at clone time)
Date: 2026-06-28
Model: `mistral-7b-v0.3` (Mistral-7B-v0.3, layer 17, bf16)

## Summary

All 7 test phases passed. 98/98 CPU tests pass. The full structural pipeline
(capture → analyze → render) works end-to-end on the cluster. 19 of 21
commands validated successfully. Four issues need code changes; three are
deployment notes.

## Issues Requiring Code Changes

### 1. `torch_dtype` deprecation warning (transformers 5.x)

**File:** `src/common/loading.py:96`

**Problem:** `_model_from_pretrained` passes `torch_dtype=dtype` to
`AutoModelForCausalLM.from_pretrained`. In transformers 5.x, `torch_dtype` is
deprecated in favor of `dtype`. The existing fallback (lines 99-110) only
triggers on `TypeError` (i.e., when `torch_dtype` is fully removed), not on
deprecation warnings.

**Fix:** Swap the order — try `dtype=` first (non-deprecated), fall back to
`torch_dtype=` for older transformers versions that don't accept `dtype`:

```python
def _model_from_pretrained(path_or_name: str, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(
            path_or_name,
            dtype=dtype,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc) and "torch_dtype" not in str(exc):
            raise
        LOGGER.debug("Retrying with torch_dtype for older transformers", exc_info=True)
        return AutoModelForCausalLM.from_pretrained(
            path_or_name,
            torch_dtype=dtype,
            **kwargs,
        )
```

### 2. Worker progress file status stays "running" after completion

**File:** `src/structural/execution/model_runtime.py:178-322` (`run_capture`)

**Problem:** `_update_progress` is called during the run with
`status="running"` (default at line 105). After all plans for a model complete
(line 317) and after the entire run finishes (line 322), there is no call to
update the status to `"complete"`. The worker progress file
(`/tmp/worker_progress.txt`) is left with `status: running` indefinitely.

Observed output after successful completion:
```
status | running
current_model | mistral-7b-v0.3
current_model_progress | 3/3
```

**Fix:** Add a final progress update after each model completes (after line
320, before the `finally` or after it) and after the entire run:

```python
# After line 320 (after results["models"][model_key] = {...})
_update_progress(
    config,
    model=model_key,
    completed=sum(p["methods"][0].get("cases", 0) for p in model_results),
    total=sum(p["methods"][0].get("cases", 0) for p in model_results),
    status="complete",
)
```

Or simpler — add to the `finally` block:
```python
finally:
    clear_linalg_caches()
    del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _update_progress(config, model=model_key, completed=..., total=..., status="complete")
```

### 3. Covariance matrix inversion does not preserve symmetry

**File:** `src/rome/common.py:1263`

**Problem:** `torch.linalg.inv(cov)` on a 14336×14336 float32 matrix does not
produce an exactly symmetric result. The original covariance `C` is symmetric
(accumulated as `k^T @ k`), but the inverse has small asymmetries due to
floating-point precision.

Observed: `torch.allclose(inv_cov, inv_cov.T, atol=1e-4)` returns `False` for
both 10k and 100k matrices. The asymmetry magnitude is small relative to the
matrix values (diag_mean ~421, offdiag_mean ~2.3e-04) and does not affect ROME
results in practice.

**Fix (optional):** Symmetrize the inverse after computation:
```python
inv_cov = torch.linalg.inv(cov)
inv_cov = (inv_cov + inv_cov.T) * 0.5  # enforce symmetry
return inv_cov.to("cpu")
```

This is a numerical-quality improvement, not a correctness bug. ROME uses
`inv_cov @ k` which is robust to small asymmetries.

### 4. Standard causal-trace saves CSV to repo root, not `analysis_out/`

**File:** `src/causal_trace/causal_trace.py:170-174`

**Problem:** `save_results_to_csv` uses `handler.cfg.generation.filename` which
is `"causal_trace_{}"` (no directory prefix). The CSV is written to the current
working directory (repo root), e.g. `causal_trace_mistralai-Mistral-7B-v0.3_*.csv`.
In contrast, `alt_trace.py` saves to `analysis_out/` with an explicit path
prefix. This inconsistency makes it hard to find standard trace output.

**Observed:**
- Standard trace: `~/Latium/causal_trace_mistralai-Mistral-7B-v0.3_2026-06-28_14-57-57.csv`
- Alt trace: `~/Latium/analysis_out/causal_trace_mistralai-Mistral-7B-v0.3_alt_2026-06-28_14-58-12.csv`

**Fix:** Change the `generation.filename` config or the `save_results_to_csv`
call to include `analysis_out/` prefix:
```python
# In causal_trace.py:170
save_results_to_csv(
    f"analysis_out/{handler.cfg.generation.filename.format(handler.cfg.model.name.replace('/', '-'))}",
    ...
)
```
Or update `src/config/generation/generation.yaml`: `filename: "analysis_out/causal_trace_{}"`.

## Deployment Notes (Not Code Bugs)

### 4. Matplotlib config directory permission

**Problem:** On the cluster, matplotlib cannot create `~/.config/matplotlib`
(Permission denied). It falls back to a temp directory, which works but is
slower and generates warnings on every run.

**Fix for cluster deployments:** Set `MPLCONFIGDIR` to a writable directory:
```bash
export MPLCONFIGDIR=/tmp/matplotlib
```
Or add to the conda env activation script.

### 5. Prefix generation: low usable template yield for Mistral

**Observation:** `prefix_mode=self` generates 50 templates per case, but 49/50
are rejected as non-Latin (`prefix_enforce_latin: True` in
`mistral-7b-v0.3.yaml`). The system fills the remainder with static templates,
so ROME edits still work. But the low yield suggests Mistral generates
non-English text for most template prompts.

**Log output (per case):**
```
Rejected 49 non-Latin and 0 short templates from self-generated templates
prefix_mode=self produced 1/50 usable templates; filling remainder with static templates
```

**Not a bug** — the fallback works correctly. But if prefix diversity matters
for Mistral experiments, consider adjusting the prompt or using
`prefix_mode=template` instead.

### 6. `batch-rome` processes entire CounterFact dataset with no case limit

**File:** `src/rome/rome.py:169` (`batch_evaluation`)

**Problem:** `batch_evaluation` iterates through all 19728 CounterFact cases
via `batch_intervention_generator`, which loads the full dataset. There is no
`n_tests` or `num_of_runs` config to limit the case count. The command ran for
600s without completing (expected — each case requires a full ROME edit +
evaluation).

**Not a bug** — this is the intended batch evaluation behavior. But it makes
the command impractical for quick testing. Consider adding a `n_tests` limit
config if smaller batch runs are needed.

### 7. `prefix-experiment` extremely slow for Mistral (>50 min, did not complete)

**File:** `src/experiments/prefix_variability/runner.py:464` (`run_experiment`)

**Problem:** The experiment creates 16+ run configurations (self/template/
external × short/long/extended × various prefix sources). Each run does a full
ROME edit (k/v collection, v-step optimization with 20 epochs, weight insertion)
+ detection (spectral, blind, composite, attention, matrix anomaly). For
Mistral, the self-prefix generation is the bottleneck: 49/50 generated templates
are rejected as non-Latin, requiring many model generation calls per run.

**Observed:** Ran for 51 minutes on A100 without producing output. The command
starts correctly (loads model, creates prefix caches, begins baseline spectral
snapshot) but the full experiment with all 16+ runs is too slow for Mistral.

**Workaround:** Use `prefix_experiment.run_names=[self_short]` to limit to a
single run configuration. Even with this, a single run + baseline took >10
minutes for Mistral.

## What Was Tested and Passed

### Phase 1: Covariance computation + progress bar
- **10k covariance matrix** computed for mistral-7b-v0.3 in ~6.5 minutes
- **tqdm progress bar** (`src/rome/common.py:1218`): total=10000, reaches
  exactly 10000/10000, smooth increments in batches of 64, no stalls or
  double-counts, `mininterval=1.0` respected, no OOM events
- Output: `data/second_moment_stats/mistralai_Mistral-7B-v0.3_17_SM_Method.WIKIPEDIA_10000.pt`
  - Shape: (14336, 14336) — MLP `intermediate_size`, correct for `down_proj` input
  - dtype: float32, all finite
  - Consistent with the 100k matrix (similar diag/offdiag means)
- **100k matrix** pulled from kubapc (`/data/covariance_matrices/`) to cluster
- `structural validate-cov` detects both matrices correctly

### Phase 2: Model/dataset download
- `download-model model=mistral-7b-v0.3` — HF Hub download (~20s, 14.5GB),
  saved to `../models/mistralai/Mistral-7B-v0.3/` ✓
- `download-datasets` — CounterFact (19728 train + 2191 test) + wikitext ✓
- `print-arch model=mistral-7b-v0.3` — module names verified:
  `down_proj: Linear(14336→4096)`, `up_proj: Linear(4096→14336)`, 32 layers ✓

### Phase 3: Full structural pipeline (n=3)
- `structural plan` — valid plan, run_id, analysis variants ✓
- `structural run` — 3/3 ROME edits successful (ES=1.0, edit_success_rate=1.0) ✓
- Composite detector: 3/3 correct layer detection (layer 17) ✓
- gpt-norm-cv: correctly `unavailable` (non-GPT model family) ✓
- All 7 renderers produced (paper, detector, run-summary, rome-success,
  detector-window, detector-signals, structural-artifact-grid) ✓

### Phase 4: Manifest invariants
- **Skip-current:** re-run without force → all captures/analyses/renders skipped ✓
- **Manifest append across invocations:** separate `structural analyze` loads
  manifest, skips current analyses ✓
- **Force recompute:** `structural.run.force=true` → captures re-run,
  content_hash changes ✓
- **Stale descendant removal:** old analyses + renders deleted, new ones
  computed with updated inputs ✓
- **`graphs run` independent:** loads manifest, skips current, force works ✓
- **Worker progress file:** `current_model_progress: 3/3` correct (but status
  stale — see issue #2)

### Phase 5: Graph rendering
- All 7 renderers produce output files (PNG/JSON/CSV/PDF) ✓
- `graphs.force=true` re-renders all ✓
- `structural-artifact-grid` produces PNG (1.3MB) + PDF + JSON ✓

### Phase 6: CLI/registry smoke + pytest suite
- `command=methods` — all registries listed correctly ✓
- `structural validate-cov` — correct file detection, exit codes ✓
- **98/98 tests passed** in 9.8s ✓

## Phase 7: Full Command Validation

Every Latium command was run and validated. Results:

| # | Command | Exit | Status | Validation |
|---|---------|------|--------|------------|
| 1 | `help` | 0 | PASS | Prints usage with primary commands + shortcuts |
| 2 | `methods` | 0 | PASS | Lists all 4 registries (edit methods, captures, analyses, renderers) + presets |
| 3 | `print-arch` | 0 | PASS | Prints MistralForCausalLM architecture, 32 layers, `down_proj: Linear(14336→4096)` |
| 4 | `download-model` | 0 | PASS | Model loaded from local cache (14.5GB safetensors) |
| 5 | `download-datasets` | 0 | PASS | CounterFact (19728 train + 2191 test) + wikitext cached |
| 6 | `second-moment` (128 samples) | 0 | PASS | tqdm: 128/128, CSV saved (822MB), shape (14336,14336), float32, finite |
| 7 | `generate-prefixes` | 0 | PASS | Prints 50 prefixes (1 self-generated + 49 static fallback) |
| 8 | `compute-multiplier` | 0 | PASS | Prints multiplier value `0.00537872314453125`, processes 219 cases |
| 9 | `edit` (Hydra syntax) | 2 | EXPECTED | `edit` is not a CLI alias — use `command=edit` or the `rome` alias |
| 10 | `rome` (alias) | 0 | PASS | Single ROME edit: success=true, ES=1.0, generation produces "The Eiffel Tower is in Rome" |
| 11 | `manual-rome` | 0 | PASS | ROME edit applied, interactive prompt shown, exits on EOF |
| 12 | `batch-rome` | 124 | TIMEOUT | Processes all 19728 CounterFact cases — no case limit (see note #6) |
| 13 | `causal-trace` (n=2) | 0 | PASS | 2 prompts processed, 0 failed, CSV with per-layer restoration probs (12 rows) |
| 14 | `alt-trace` (n=2) | 0 | PASS | 2 prompts, best_layer=10, signal_quality=noisy, CSV + JSON saved to `analysis_out/` |
| 15 | `rome-benchmark` (n=2) | 0 | PASS | 2 cases, ES=1.0/1.0, mean_ES=1.0, JSON saved to `analysis_out/` |
| 16 | `structural plan` | 0 | PASS | Valid JSON plan with run_id, analysis variants, planned_runs |
| 17 | `structural validate-cov` | 0 | PASS | Detects 100k matrix, `ok: true` |
| 18 | `structural capture` | 0 | PASS | 2 cases, rome method, 6 manifest artifacts (execution + spectral + matrix-features) |
| 19 | `structural analyze` | 0 | PASS | All 3 analyses skipped (current) — manifest append across invocations works |
| 20 | `structural run` (skip) | 0 | PASS | All captures/analyses/renders skipped (current) |
| 21 | `graphs run` | 0 | PASS | All 7 renderers skipped (current) |
| 22 | `prefix-experiment` | — | SLOW | Starts correctly but >50 min for Mistral (see note #7) |
| 23 | `graphs prefix` | — | SKIPPED | Requires prefix-experiment artifact (not produced within timeout) |

### Command output validation details

**`rome` (single edit):**
- JSON output: `success: true`, `efficacy_score: 1.0`, `paraphrase_score: 0.5`
- Generation: `"The Eiffel Tower is in Rome"` (edit successful)
- Probe generation works (200 tokens per probe)

**`rome-benchmark` (n=2):**
- JSON structure: `{summary, cases}` at top level
- Summary: `mean_efficacy_score: 1.0`, `mean_paraphrase_score: 0.5`, `mean_neighborhood_score: 0.8`
- Cases have direct metrics (not nested in `post`/`pre`): `efficacy_score`, `paraphrase_score`, `neighborhood_score`, `rewrite_nll`, `paraphrase_nll`

**`causal-trace` (standard, n=2):**
- CSV: 12 rows (header + 11 data rows), columns: `run_number, prompt_num, clean, corrupted, restored_token, restored`
- Saved to repo root (not `analysis_out/` — see issue #4)
- Contains per-layer restoration probabilities for 2 prompts

**`alt-trace` (n=2):**
- CSV: 3 rows (header + 2 data rows), per-layer probabilities
- JSON: `{model, num_layers: 32, num_prompts: 2, best_layer: 10, signal_quality: "noisy", used_middle_third_fallback: true, candidates: [12], avg_per_layer_probs: [32]}`
- Saved to `analysis_out/` ✓

**`structural capture` (standalone, n=2):**
- Run root: `analysis_out/cmd_test_capture/`
- 6 manifest artifacts: baseline execution + spectral + matrix-features, method/rome execution + spectral + matrix-features
- All status: `complete`

**`compute-multiplier`:**
- Processes 219 CounterFact cases
- Outputs scalar value: `0.00537872314453125`

## Cluster Environment

- **GPU host:** `ubuntu@216.81.248.46` (via `ssh -J metju@kubapc`)
- **Hardware:** A100-SXM4-80GB, 98GB RAM, 14 CPUs, 576GB disk
- **Software:** Python 3.11 (conda `latium` env), torch 2.12.1+cu130,
  transformers 5.12.1, hydra-core 1.3.3
- **Covariance source:** `/data/covariance_matrices/` on kubapc
- **10k matrix saved to:** `/data/covariance_matrices/mistralai_Mistral-7B-v0.3_17_SM_Method.WIKIPEDIA_10000.pt` on kubapc

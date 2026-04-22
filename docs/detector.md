# ROME Layer Detectors

This document describes the detectors as they are implemented in:

- `detector/composite_detector_v2.py`
- `detector/gpt_detector.py`

It is intended to match the current code, not older notebook summaries or historical benchmark claims.

---

## Overview

The repository has two post-hoc layer detectors that operate on
`blind_detection.layer_features` saved by `structural_benchmark.py`.

| Detector | Intended scope | Main idea |
|----------|----------------|-----------|
| `composite_detector_v2.py` | Non-GPT architectures | Multi-signal spectral chain with two targeted edge rescues |
| `gpt_detector.py` | GPT-2 family and GPT-J | 3-way `norm_cv` vote with multi-trim fallback |

Important:

- The composite script does **not** automatically exclude GPT-family JSON files.
- The GPT script **does** filter to GPT-family files by default.
- In practice, GPT-family runs should be evaluated with `gpt_detector.py`, not with the composite detector.

Pipeline routing:

- `pipeline_posthoc.py` dispatches by canonical model key, not by filename.
- GPT-2 and GPT-J models go through `gpt_detector.process_file(..., trim=5)`.
- Non-GPT models go through `composite_detector_v2.process_file(..., trim=2)`.

---

## Source Data

Both detectors read:

- `test["blind_detection"]["layer_features"]`

Those features are produced by `BlindMSDDetector.detect_layer_features_only()`
in `src/structural/blind_detector.py`, and are written into structural JSON
outputs by `structural_benchmark.py`.

For standard post-hoc detector usage, a structural run must have:

- no test-level error
- successful ROME edit (`test["rome"]["success"] == True`) for layer detection
- non-empty `blind_detection.layer_features`

Baseline-only binary edit detection relaxes the ROME-success requirement for
baseline files.

---

## Per-Layer Feature Bundle

The current implementation computes these per-layer features:

| Feature | Key | Current implementation |
|---------|-----|------------------------|
| Spectral gap | `spectral_gap` | `sigma_1 / sigma_2` |
| Top-1 energy | `top1_energy` | `sigma_1^2 / sum_j sigma_j^2` |
| Effective rank | `effective_rank` | `exp(H(p))`, where `p_i = sigma_i / sum_j sigma_j` |
| Norm CV | `norm_cv` | `std(row_norms) / mean(row_norms)` |
| Row alignment | `row_alignment` | `max(abs(U[:,0])) / mean(abs(U[:,0]))` |
| Spectral entropy | `spectral_entropy` | `-sum_i q_i log q_i / log(r)`, where `q_i = sigma_i^2 / sum_j sigma_j^2` and `r = len(S)` |
| PCS | `pcs` | Pairwise row-cosine summary from `src.rome.common.pcs()` |

Notes:

- `effective_rank` uses normalized singular values `S / sum(S)`, not squared energy.
- `spectral_entropy` is normalized by `log(r)`.
- The post-hoc detectors described below primarily use `spectral_gap`,
  `top1_energy`, `norm_cv`, `effective_rank`, and `row_alignment`.

---

## Shared Signal Transforms

The detectors derive peaks from transformed per-layer feature series.

### Raw

Use the feature values directly.

### Local z-score

For each layer, compare that layer against its local neighborhood:

`z_i = (x_i - mean(neighbors)) / (std(neighbors) + eps)`

The detectors use the absolute value of that score.

Default windows used by the composite detector:

- SG small window: `5`
- SG large window: `7`
- TE window: `5`
- NC window: `5`

These are configurable inside `detect_layer()` and via composite window-sweep mode.

### Curvature

Absolute second finite difference:

`abs(x_{i-1} - 2x_i + x_{i+1})`

### Peak selection

For each transformed series, the detector takes:

- `argmax(series)` as the peak layer
- a simple global z-score for reporting/debug info

---

## Composite Detector v2

File:

- `detector/composite_detector_v2.py`

Primary entry points:

- `detect_layer()`
- `process_file()`
- `detect_edit_binary()`
- `sweep_file()`

Default trim:

- `2` layers from each end

### Intended scope

This detector is intended for non-GPT architectures.

It will still run on GPT-family JSONs if you pass them in, but that is not the
recommended workflow. GPT-family runs should use `gpt_detector.py`.

### Signals used by the composite detector

Primary spectral signals:

- `SG`: raw `spectral_gap`
- `TE`: local-z `top1_energy`
- `SG_small`: local-z `spectral_gap` with the small window
- `SG_large`: local-z `spectral_gap` with the large window

Secondary structural signals:

- `NC`: local-z `norm_cv`
- `ER`: curvature of `effective_rank`
- `RA`: raw `row_alignment`

`NC` is still recorded in detector metadata for debugging, but the current
final decision path only uses `ER` and `RA` in the structural rescue.

### Core decision chain

Let:

- `sg_l` be the raw spectral-gap peak layer
- `te_l` be the top-1-energy local-z peak layer
- `s5_l` be the small-window spectral-gap local-z peak layer
- `s7_l` be the large-window spectral-gap local-z peak layer

The detector follows this cascade:

1. If `SG == TE`, return `agree`.
2. If `SG_small` is within `+-1` index of `SG` but not `TE`, return `sg(lz{small_window})`.
3. If `SG_small` is within `+-1` index of `TE` but not `SG`, return `te(lz{te_window})`.
4. If `SG_large` is within `+-1` index of `SG` but not `TE`, return `sg(lz{large_window})`.
5. If `SG_large` is within `+-1` index of `TE` but not `SG`, return `te(lz{large_window})`.
6. Otherwise compute Spearman trend `rho` of raw `spectral_gap` over the evaluated layers.

If `abs(rho) > 0.3`:

- If `SG_small` and `SG_large` are within `+-1` index, use the stronger of the two and return `lz_cons(k)` where `k` is the winning window.
- Else if `SG_large` is within `+-1` index of `TE`, return `te(trend)`.
- Else return `s7(trend)`.

If `abs(rho) <= 0.3`:

- Return `sg(fb)`.

### Additional rescue rules and overrides

The current implementation keeps two targeted rescues beyond the basic v5 chain.

#### 1. Edge local-z consensus rescue

If the chosen `lz_cons(*)` peak lies near the boundary of the evaluated range,
but `TE` forms a strong interior peak far enough away, the detector switches to:

- `te(edge_lz{te_window})`

This is an explicit edge-artifact correction.

#### 2. Early structural consensus rescue

If the current method is `sg(lz{small_window})`, but:

- `ER` and `RA` agree closely,
- their consensus is very early,
- and the spectral-gap peak is much later,

the detector switches to:

- `er_ra(edge)`

This is meant to catch late spectral peaks that look like architecture artifacts.

The earlier low-confidence structural fallbacks (`er_ra(fb)`, `nc_er(fb)`,
`ra(fb)`, `nc(fb)`) are no longer part of the implemented detector.

### Returned metadata

For each test, the detector records debug info including:

- window settings
- per-signal peak layer/index/z-score
- evaluated layer range
- Spearman `rho` when the trend fallback is used
- any override metadata in `info["v5_override"]`

### Binary edit detection

`detect_edit_binary()` answers a different question:

- "Was this model edited at all?"

Per test, it computes an anomaly score:

- `max(max(abs(local_zscore(spectral_gap, 5))), max(abs(local_zscore(top1_energy, 5))))`

with the same edge trim.

Then:

- if a baseline file is provided, it compares the edited mean anomaly score to
  the baseline score distribution
- if the baseline has fewer than 2 usable samples or near-zero variance, the
  result is reported as inconclusive
- otherwise the edited model is labeled as edited when separation exceeds `2.0`
  baseline standard deviations
- if no baseline is provided, it uses an absolute threshold, default `3.0`

### Window sweep mode

The composite detector also supports hyperparameter sweeps over:

- trim
- small local-z window
- large local-z window

In sweep mode, `TE` and `NC` use the same window as the chosen small window.

This is exposed through:

- `--window-sweep`
- `--sweep-trims`
- `--sweep-small-windows`
- `--sweep-large-windows`
- `--sweep-top-k`

---

## GPT Detector v6

File:

- `detector/gpt_detector.py`

Primary entry points:

- `detect()`
- `process_file()`

Default trim:

- `5` layers from each end

### Scope and file selection

By default, the GPT detector only collects files whose names match GPT-family
slugs:

- `gpt-j`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`

Passing `--all-models` disables that filename filter, but the intended use is
still GPT-family runs only.

### Signals used by the GPT detector

The GPT detector uses only `norm_cv`, under three transforms:

- raw `norm_cv`
- local-z `norm_cv` with window `5`
- curvature of `norm_cv`

### Voting rule

At the requested trim, each transform votes for its peak layer.

If any layer gets at least two votes:

- return that layer
- method tag: `nc3_t{trim}`

### Multi-trim fallback

If there is no majority at the requested trim, the detector pools votes across:

- trim `4`
- trim `5`
- trim `6`

The winner of those pooled votes is returned with:

- method tag: `nc3_mt4-6`

If no votes are available at all, it returns:

- `nc3_fail`

### Why GPT has its own detector

The GPT-specific detector exists because the spectral signals used by the
composite detector are not reliable enough for GPT-family runs in this repo's
post-hoc setting. The GPT detector instead leans entirely on `norm_cv`.

---

## Running

### Composite detector

Use this for non-GPT model outputs:

```bash
python detector/composite_detector_v2.py pipeline_out/n50_s30/structural
```

Generate graphs:

```bash
python detector/composite_detector_v2.py \
  pipeline_out/n50_s30/structural \
  --graphs \
  --graph-dir detector/graphs
```

Run binary edit detection against matching baselines:

```bash
python detector/composite_detector_v2.py \
  --binary \
  pipeline_out/n50_s30/structural \
  --baseline-dir ultrasupertest
```

Sweep detector windows:

```bash
python detector/composite_detector_v2.py \
  pipeline_out/n50_s30/structural/rome_structural_qwen3-8b_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_s01_r01_2026-04-21_11-06-40.json \
  --window-sweep \
  --sweep-top-k 5
```

### GPT detector

Use this for GPT-family outputs:

```bash
python detector/gpt_detector.py pipeline_out/n50_s30/structural
```

Generate GPT detector graphs:

```bash
python detector/gpt_detector.py \
  pipeline_out/n50_s30/structural \
  --graphs \
  --graph-dir detector/graphs/gpt
```

---

## Output Artifacts

Typical graph outputs include:

```text
detector/graphs/
  runs/
    signals_*.png
    gpt_signals_*.png
  averages/
    signals_*_average.png
  summary_accuracy_all_runs.png
  summary_accuracy_model_average.png
  method_breakdown_all_runs.png
  method_breakdown_model_average.png
```

For run-scoped visualization across pipeline outputs, `detector/visualize_detector.py`
can aggregate multiple runs and produce per-run plus per-model-average plots.

---

## Accuracy Notes

This document intentionally does not hard-code a single accuracy number.

Detector accuracy depends on:

- model family
- run slice
- benchmark size
- detector window settings
- whether you are using the intended detector for that model family

Use the current script outputs on your current structural JSON files as the
source of truth.

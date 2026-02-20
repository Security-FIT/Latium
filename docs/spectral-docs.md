# Spectral Detector

## Overview

Given model weights across layers, detect which layer looks most anomalous after an edit.

The detector has **two spectral signals**:
- **Signal A (`sv_z_scores`)**: curvature energy of z-scored top singular values of `c_proj`.
- **Signal B (`sv_ratio_scores`)**: curvature energy of top singular-value ratios `c_proj / c_fc`.



## Signal A (z-scored singular values)

For each layer, compute top-$K$ singular values of `c_proj`.

Then:
1. z-score each singular-value index across layers,
2. compute second-derivative energy over depth.

Here, "over depth" means a centered second difference at layer $\ell$ using only its two nearest neighbors ($\ell-1$ and $\ell+1$):
$$
\Delta^2 x_{\ell} = x_{\ell-1} - 2x_{\ell} + x_{\ell+1}
$$

Large energy means a local spectral break at that layer.

---

## Signal B (ratio singular values)

If `c_fc` weights are available and layer indices match `c_proj`:
1. compute top-$K$ singular values for both,
2. build ratio `sv_proj / (sv_fc + eps)`,
3. compute second-derivative energy over depth.

Same depth rule here: each layer is compared through its immediate neighbors (not all layers directly) via that second-difference operator.

This helps cancel shared scale trends and highlight local `c_proj` distortions.

---

## 4) Layer trimming and candidate selection

The detector supports:
- `trim_first_layers`
- `trim_last_layers`

These layers are removed **before** all spectral calculations.

Then `boundary` should be applied to candidate selection inside the evaluated range. (for now its hardcoded to 0)

---

## FInal layer

(just a prototype, we might need it for tests)

- Rank candidates by Signal A (higher score = more anomalous).
- If Signal B exists, rank by Signal B too and average the two ranks.
- Pick layer with best (lowest) combined rank.

Returned `detection_score` is a rank-based confidence:
$$
\text{detection\_score} = N_{candidates} - \text{winning\_combined\_rank}
$$

---

## 6) Current detector output schema

`SpectralDetector.detect(...)` returns:
- `anomalous_layer`
- `detection_score`
- `sv_z_scores` (dict for all original layers)
- `sv_ratio_scores` (dict for all original layers; zeros if Signal B unavailable)
- `has_fc_weights`
- `config` (`top_k`, `boundary`, `trim_first_layers`, `trim_last_layers`)
- `excluded_layers` (trimmed out)
- `evaluated_layers` (used in spectral math)

For empty/over-trimmed cases, it returns a safe null-style result (`anomalous_layer=None`, score `0.0`).

---

# Spectral Detector

## Overview

Given model weights across layers, detect which layer looks most anomalous after an edit.

The detector has two singular-value signals and four PCS terms:
- **Signal A (`sv_z_scores`)**: curvature energy of z-scored top singular values of `c_proj`.
- **Signal B (`sv_ratio_scores`)**: curvature energy of top singular-value ratios `c_proj / c_fc`.
- `pcs_neighbor_var_scores`, `pcs_next_jump_scores`, and
  `pcs_next_curvature_scores` from projection directions.
- `pcs_cross_shift_scores` from projection/FC directions when FC exists.



## Signal A (z-scored singular values)

For each layer, compute top-$K$ singular values of `c_proj`.

Using SVD notation:
$$
W^{(p)}_\ell = U^{(p)}_\ell\Sigma^{(p)}_\ell(V^{(p)}_\ell)^\top
$$

Then:
1. z-score each singular-value index across layers,
2. compute second-derivative energy over depth.

For singular index $i$:
$$
z_{\ell,i} = \frac{\sigma^{(p)}_{\ell,i}-\mu_i}{s_i+\varepsilon}
$$
and per-layer Signal A is:
$$
A_\ell = \sum_i (z_{\ell-1,i}-2z_{\ell,i}+z_{\ell+1,i})^2
$$

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

SVD for FC:
$$
W^{(f)}_\ell = U^{(f)}_\ell\Sigma^{(f)}_\ell(V^{(f)}_\ell)^\top
$$

Signal B is:
$$
r_{\ell,i}=\frac{\sigma^{(p)}_{\ell,i}}{\sigma^{(f)}_{\ell,i}+\varepsilon},
\quad
B_\ell = \sum_i (r_{\ell-1,i}-2r_{\ell,i}+r_{\ell+1,i})^2
$$

Same depth rule here: each layer is compared through its immediate neighbors (not all layers directly) via that second-difference operator.

This helps cancel shared scale trends and highlight local `c_proj` distortions.

---

## 4) Layer trimming and candidate selection

The detector supports:
- `trim_first_layers`
- `trim_last_layers`

These layers are removed before detector score calculations. Capture-time SVD
primitives remain reusable across trim variants.

Then `boundary` is applied to candidate selection inside the evaluated range.

---

## 5) Output 

`SpectralDetector.detect(...)` returns:
- `anomalous_layer`
- `detection_score`
- `sv_z_scores` (dict for all original layers)
- `sv_ratio_scores` (dict for all original layers; zeros if Signal B unavailable)
- the four score-relevant PCS maps listed below
- `sv_z_rolling_z_scores`, `sv_ratio_rolling_z_scores`,
  `pcs_composite_rank_scores`, `sv_pcs_contradiction_scores`, and
  `rome_hybrid_scores`
- `has_fc_weights`
- `config` (`top_k`, `boundary`, `trim_first_layers`, `trim_last_layers`)
- `excluded_layers` (trimmed out)
- `evaluated_layers` (used in spectral math)

For empty/over-trimmed cases, it returns a safe null-style result (`anomalous_layer=None`, score `0.0`).

---

## 6) PCS detection

### PCS signals

The current score calculates and returns only the PCS maps it consumes:

- `pcs_neighbor_var_scores`
- `pcs_next_jump_scores`
- `pcs_next_curvature_scores`
- `pcs_cross_shift_scores`

When FC information cannot be used, the cross-shift map is zero. Historical
mean/min/flip and cross-curvature diagnostics are absent from the default path
because they do not affect `rome_hybrid_scores`.

### How PCS is calculated

For each evaluated layer, take top-$K$ principal directions and singular values.

1. Canonicalize direction signs (to remove SVD sign ambiguity):
$$
\widetilde v_{\ell,i} = s_{\ell,i}\,v_{\ell,i},\quad
s_{\ell,i}\in\{-1,+1\}
$$
where the sign is chosen so the largest-absolute component of $\widetilde v_{\ell,i}$ is positive.

2. Compare two layers with weighted signed cosine:
$$
\operatorname{wPCS}(\ell,m)=\sum_{i=1}^{K}\widetilde w_{\ell,m,i}\,\langle \widetilde v_{\ell,i},\widetilde v_{m,i}\rangle,
\quad
\widetilde w_{\ell,m,i}=\frac{\tfrac12(\sigma_{\ell,i}+\sigma_{m,i})}{\sum_j\tfrac12(\sigma_{\ell,j}+\sigma_{m,j})+\varepsilon}
$$

3. Neighbor variance within radius $r$ around each layer:
$$
\mathcal N_r(\ell)=\{m:\,0<|m-\ell|\le r\}
$$
$$
\operatorname{neighbor\_var}_\ell=\operatorname{var}_{m\in\mathcal N_r(\ell)}\operatorname{wPCS}(\ell,m)
$$

4. Next-layer PCS is retained internally only long enough to calculate jump
and curvature:
$$
\operatorname{pcs\_next}_\ell=\operatorname{wPCS}(\ell,\ell+1)
$$
$$
\operatorname{pcs\_next\_jump}_\ell = \left|\operatorname{pcs\_next}_\ell-\operatorname{pcs\_next}_{\ell-1}\right|
$$

Implementation edge handling:
- for `pcs_next_jump_scores`, the first entry copies the first finite jump,
- if there are fewer than 2 evaluated layers, these arrays stay zero.

`pcs_next_curvature_scores` is second-derivative energy of `pcs_next_scores` over depth (same rule as Signal A/B).

5. Cross-projection PCS (when FC exists):

`detect(...)` computes this only when `fc_weights` is provided **and** FC layer keys exactly match `c_proj` layer keys (`has_fc_weights=True`).

For this term, the code compares:
- left singular vectors from `c_proj` (`U^T` rows), and
- right singular vectors from `c_fc` (`V^T` rows).

If vector dimensions do not match (or effective $K \le 0$), cross signals are returned as zeros.

$$
\operatorname{pcs\_cross}_\ell = \sum_{i=1}^{K}\widetilde w_{\ell,i}\,\langle \widetilde v^{(proj)}_{\ell,i},\widetilde v^{(fc)}_{\ell,i}\rangle
$$

Only the consumed shift is emitted:
$$
\operatorname{pcs\_cross\_shift}_\ell = 1-\operatorname{pcs\_cross}_\ell
$$

The capture stores dot/weight cumulative sums only for the configured neighbor
radius. It does not calculate flip fractions or all-pairs PCS. Edited cases
recompute SVD/PCS primitives only for changed layers and their required
neighbors; materialization overlays those rows on the clean baseline capture.

### Final scoring

The detector computes a hybrid score per layer (`rome_hybrid_scores`) and selects:
$$
\hat\ell = \arg\max_{\ell\in\mathcal C}\;\text{rome\_hybrid\_scores}_\ell
$$
where $\mathcal C$ is the candidate set after trimming and boundary exclusion.

If `boundary` removes all internal candidates (`n - 2*boundary <= 0`), the code falls back to using all evaluated layers as candidates.

Hybrid helper outputs:
- `sv_z_rolling_z_scores`
- `sv_ratio_rolling_z_scores`
- `pcs_composite_rank_scores`
- `sv_pcs_contradiction_scores`
- `rome_hybrid_scores`

The active singular-value branch uses an absolute rolling z-score with the
configured centered window (forced odd). With FC, this is
`sv_ratio_rolling_z_scores` and `sv_z_rolling_z_scores` is zero; without FC,
the inverse applies. The unused rolling transform is not calculated.

Rolling z-score used by code:
$$
\operatorname{RZ}_\ell(x)=\left|\frac{x_\ell-\mu_{\mathcal N(\ell)}}{s_{\mathcal N(\ell)}+\varepsilon}\right|
$$
where $\mathcal N(\ell)$ is the centered local window of size 5 clipped to valid layer indices.

`pcs_composite_rank_scores` is a rank-mean of key PCS anomaly terms:
$$
\operatorname{pcs\_rank}_\ell = \operatorname{mean}\big(\operatorname{rank01}(\operatorname{pcs\_next\_jump}_\ell),\operatorname{rank01}(\operatorname{pcs\_neighbor\_var}_\ell),\operatorname{rank01}(\operatorname{pcs\_next\_curvature}_\ell)[,\operatorname{rank01}(\operatorname{pcs\_cross\_shift}_\ell)]\big)
$$

With FC, `sv_pcs_contradiction_scores` is:
$$
\operatorname{contradiction}_\ell = \operatorname{sv\_rank}_\ell\cdot(1-\operatorname{pcs\_rank}_\ell)
$$

The corresponding singular-value rank is:
$$
\operatorname{sv\_rank}_\ell =
\frac{1}{2}\left(\operatorname{rank01}(A_\ell)+\operatorname{rank01}(B_\ell)\right)
$$

where $A_\ell$ is Signal A (`sv_z_scores`) and $B_\ell$ is Signal B (`sv_ratio_scores`).
Without FC, contradiction is not part of the hybrid and is returned as zero
without calculating the unused rank term.

With FC weights:
$$
H_\ell = 0.55\,\operatorname{RZ}(\operatorname{sv\_ratio})_\ell + 0.25\,\operatorname{contradiction}_\ell + 0.20\,\operatorname{pcs\_rank}_\ell
$$

Without FC weights:
$$
H_\ell = 0.75\,\operatorname{RZ}(\operatorname{sv\_z})_\ell + 0.25\,\operatorname{pcs\_rank}_\ell
$$

### Detection score

`detection_score` is the winning hybrid value:
$$
\operatorname{detection\_score} = \max_{\ell\in\mathcal C} H_\ell
$$

This replaces the older rank-based confidence.

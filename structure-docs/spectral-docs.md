# Spectral Detector

## Overview

Given model weights across layers, detect which layer looks most anomalous after an edit.

The detector has **two spectral signals**:
- **Signal A (`sv_z_scores`)**: curvature energy of z-scored top singular values of `c_proj`.
- **Signal B (`sv_ratio_scores`)**: curvature energy of top singular-value ratios `c_proj / c_fc`.



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

These layers are removed **before** all spectral calculations.

Then `boundary` is applied to candidate selection inside the evaluated range.

---

## 5) Output 

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

## 6) PCS detection

### PCS signals

Within-projection PCS maps:
- `pcs_neighbor_mean_scores`
- `pcs_neighbor_shift_scores`
- `pcs_neighbor_var_scores`
- `pcs_neighbor_min_shift_scores`
- `pcs_neighbor_flip_fraction_scores`
- `pcs_next_scores`
- `pcs_next_shift_scores`
- `pcs_next_jump_scores`
- `pcs_next_curvature_scores`

Cross-projection PCS maps:
- `pcs_cross_scores`
- `pcs_cross_shift_scores`
- `pcs_cross_curvature_scores`

All of these keys are always present in `detect(...)` output; when FC information is unavailable/incompatible, cross maps are all zeros.

### How PCS is calculated

For each evaluated layer, take top-$K$ principal directions and singular values.  
(I kinda vibed up these formulas for markdown but im like 98% sure they match the code)

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

3. Neighbor-based PCS signals (within radius $r$ -number of layers- around each layer):
$$
\mathcal N_r(\ell)=\{m:\,0<|m-\ell|\le r\}
$$
$$
\operatorname{neighbor\_mean}_\ell=\operatorname{mean}_{m\in\mathcal N_r(\ell)}\operatorname{wPCS}(\ell,m)
$$
$$
\operatorname{neighbor\_shift}_\ell=\operatorname{mean}_{m\in\mathcal N_r(\ell)}\left(1-\operatorname{wPCS}(\ell,m)\right)
$$
$$
\operatorname{neighbor\_var}_\ell=\operatorname{var}_{m\in\mathcal N_r(\ell)}\operatorname{wPCS}(\ell,m)
$$

`pcs_neighbor_min_shift_scores` uses the strongest local drop:
$$
\operatorname{neighbor\_min\_shift}_\ell = 1 - \min_{m\in\mathcal N_r(\ell)}\operatorname{wPCS}(\ell,m)
$$

`pcs_neighbor_flip_fraction_scores` is the weighted fraction of principal components with negative pairwise dot product.

4. Next-layer PCS signals:
$$
\operatorname{pcs\_next}_\ell=\operatorname{wPCS}(\ell,\ell+1)
$$
$$
\operatorname{pcs\_next\_shift}_\ell = 1-\operatorname{pcs\_next}_\ell
$$
$$
\operatorname{pcs\_next\_jump}_\ell = \left|\operatorname{pcs\_next}_\ell-\operatorname{pcs\_next}_{\ell-1}\right|
$$

Implementation edge handling:
- for `pcs_next_scores`, the last entry copies the previous one,
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

Then:
$$
\operatorname{pcs\_cross\_shift}_\ell = 1-\operatorname{pcs\_cross}_\ell
$$
and `pcs_cross_curvature_scores` is its second-derivative energy over depth.

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

`sv_z_rolling_z_scores` and `sv_ratio_rolling_z_scores` use absolute rolling z-score with centered window size 5 (forced odd).
When `has_fc_weights=False`, `sv_ratio_rolling_z_scores` is all zeros.

Rolling z-score used by code:
$$
\operatorname{RZ}_\ell(x)=\left|\frac{x_\ell-\mu_{\mathcal N(\ell)}}{s_{\mathcal N(\ell)}+\varepsilon}\right|
$$
where $\mathcal N(\ell)$ is the centered local window of size 5 clipped to valid layer indices.

`pcs_composite_rank_scores` is a rank-mean of key PCS anomaly terms:
$$
\operatorname{pcs\_rank}_\ell = \operatorname{mean}\big(\operatorname{rank01}(\operatorname{pcs\_next\_jump}_\ell),\operatorname{rank01}(\operatorname{pcs\_neighbor\_var}_\ell),\operatorname{rank01}(\operatorname{pcs\_next\_curvature}_\ell)[,\operatorname{rank01}(\operatorname{pcs\_cross\_shift}_\ell)]\big)
$$

`sv_pcs_contradiction_scores` is:
$$
\operatorname{contradiction}_\ell = \operatorname{sv\_rank}_\ell\cdot(1-\operatorname{pcs\_rank}_\ell)
$$

`sv_rank` used by code is:
$$
\operatorname{sv\_rank}_\ell=
\begin{cases}
\frac{1}{2}\left(\operatorname{rank01}(A_\ell)+\operatorname{rank01}(B_\ell)\right), & \text{with FC}\\
\operatorname{rank01}(A_\ell), & \text{without FC}
\end{cases}
$$

where $A_\ell$ is Signal A (`sv_z_scores`) and $B_\ell$ is Signal B (`sv_ratio_scores`).

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


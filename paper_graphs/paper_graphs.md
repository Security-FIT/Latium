# Paper Graphs

This note explains every graph rendered by `paper_graphs.ipynb`.

## Common Conventions

Let the projection matrix at layer $\ell$ be $W^{(p)}_\ell$ and, when available, the paired FC matrix be $W^{(f)}_\ell$.
Their singular value decompositions are

$$
W^{(p)}_\ell = U^{(p)}_\ell \Sigma^{(p)}_\ell (V^{(p)}_\ell)^\top,
\qquad
W^{(f)}_\ell = U^{(f)}_\ell \Sigma^{(f)}_\ell (V^{(f)}_\ell)^\top.
$$

For each model, the notebook plots the mean across valid ultrasupertest cases. If a matching baseline file exists, it is overlaid as a dashed line.

The dotted vertical line marks the target edit layer. `TRIM_FIRST_LAYERS` and `TRIM_LAST_LAYERS` remove layers from the start or end of every layer-indexed graph. They do not affect the rank-indexed SVD graph.

## 1. Top-k Proj Singular Values At Target Layer

This graph shows the target layer singular spectrum directly from `raw_spectral.sv_proj_topk`.

Formula:

$$
\bigl(\sigma^{(p)}_{\ell^*,1}, \sigma^{(p)}_{\ell^*,2}, \dots, \sigma^{(p)}_{\ell^*,K}\bigr)
$$

Interpretation:
- It is the cleanest view of how concentrated the target-layer spectrum is.
- A sharper drop after the first few singular values suggests stronger low-rank structure.

## 2. Signal A (`sv_z_scores`)

Signal A measures how unusual the top-$K$ projection singular values are over depth after per-rank z-scoring.

Formula:

$$
z_{\ell,i} = \frac{\sigma^{(p)}_{\ell,i} - \mu_i}{s_i + \varepsilon},
\qquad
A_\ell = \sum_{i=1}^{K} \left(z_{\ell-1,i} - 2 z_{\ell,i} + z_{\ell+1,i}\right)^2.
$$

Interpretation:
- High values indicate a local spectral break in the projection stack.
- It is useful when the edited layer disturbs singular values relative to neighboring layers.

## 3. Signal B (`sv_ratio_scores`)

Signal B compares projection singular values against FC singular values at the same layer, then looks for curvature over depth.

Formula:

$$
r_{\ell,i} = \frac{\sigma^{(p)}_{\ell,i}}{\sigma^{(f)}_{\ell,i} + \varepsilon},
\qquad
B_\ell = \sum_{i=1}^{K} \left(r_{\ell-1,i} - 2 r_{\ell,i} + r_{\ell+1,i}\right)^2.
$$

Interpretation:
- This suppresses global scale trends and emphasizes proj-vs-fc distortion.
- It is only meaningful when FC weights are available.

## 4. Blind Spectral Gap

This is the ratio of the largest and second-largest singular values.

Formula:

$$
\mathrm{gap}_\ell = \frac{\sigma_{\ell,1}}{\sigma_{\ell,2} + \varepsilon}.
$$

Interpretation:
- A large gap means one dominant direction is separating from the rest.
- ROME-like edits often increase this concentration at the edited layer.

## 5. Blind Top-1 Energy

This is the fraction of spectral energy captured by the top singular value.

Formula:

$$
\mathrm{top1}_\ell = \frac{\sigma_{\ell,1}^2}{\sum_j \sigma_{\ell,j}^2 + \varepsilon}.
$$

Interpretation:
- Higher values indicate stronger rank-1 dominance.
- It is a direct low-rank concentration measure.

## 6. Stable/Effective Rank Ratio

This graph comes from `novel_metrics_detection.per_metric_series.stable_effective_ratio`.

Stable rank:

$$
\mathrm{srank}_\ell = \frac{\lVert W_\ell \rVert_F^2}{\sigma_{\ell,1}^2 + \varepsilon}.
$$

Effective rank uses normalized singular values $p_{\ell,i} = \sigma_{\ell,i} / \sum_j \sigma_{\ell,j}$:

$$
\mathrm{erank}_\ell = \exp\left(-\sum_i p_{\ell,i} \log(p_{\ell,i} + \varepsilon)\right).
$$

The plotted ratio is

$$
\mathrm{ratio}_\ell = \frac{\mathrm{srank}_\ell}{\mathrm{erank}_\ell + \varepsilon}.
$$

Interpretation:
- Stable rank and effective rank respond differently to low-rank perturbations.
- Their ratio can spike when the edited layer deviates from the normal depth trend.

## 7. PCS Next Jump

First define the weighted next-layer PCS score between layers $\ell$ and $\ell+1$ using the canonicalized top-$K$ singular directions:

$$
\mathrm{PCS}_{\ell \to \ell+1} = \sum_{i=1}^{K} \widetilde{w}_{\ell,i}
\left\langle \widetilde{v}_{\ell,i}, \widetilde{v}_{\ell+1,i} \right\rangle.
$$

The plotted jump score is

$$
\mathrm{jump}_\ell = \left|\mathrm{PCS}_{\ell \to \ell+1} - \mathrm{PCS}_{\ell-1 \to \ell}\right|.
$$

Interpretation:
- It highlights abrupt changes in directional continuity.
- Large values mean one layer boundary behaves differently from its neighbors.

## 8. PCS Next Curvature

This is the second-derivative energy of the next-layer PCS series.

Formula:

$$
\mathrm{curv}^{\mathrm{next}}_\ell = \left(\mathrm{PCS}_{\ell-1} - 2\mathrm{PCS}_{\ell} + \mathrm{PCS}_{\ell+1}\right)^2.
$$

Interpretation:
- It is a curvature-style anomaly score rather than a raw similarity score.
- High values mean the next-layer alignment bends sharply at that depth.

## 9. PCS Cross Scores

This compares projection directions and FC directions within the same layer.

Formula:

$$
\mathrm{PCS}^{\mathrm{cross}}_\ell = \sum_{i=1}^{K} \widetilde{w}_{\ell,i}
\left\langle \widetilde{v}^{(p)}_{\ell,i}, \widetilde{v}^{(f)}_{\ell,i} \right\rangle.
$$

Interpretation:
- It measures within-layer proj-vs-fc directional agreement.
- A disturbed edited layer can break the normal cross-family alignment pattern.

## 10. PCS Cross Curvature

This is the curvature energy of the cross-PCS series.

Formula:

$$
\mathrm{curv}^{\mathrm{cross}}_\ell = \left(\mathrm{PCS}^{\mathrm{cross}}_{\ell-1} - 2\mathrm{PCS}^{\mathrm{cross}}_{\ell} + \mathrm{PCS}^{\mathrm{cross}}_{\ell+1}\right)^2.
$$

Interpretation:
- It emphasizes local bends in cross-family alignment over depth.
- It is often more informative than the raw cross score when the signal is subtle but localized.

## 11. Blind Norm CV

This is the coefficient of variation of row norms inside a layer.

Formula:

$$
\mathrm{CV}_\ell = \frac{\mathrm{std}_i\, \lVert w_{\ell,i} \rVert_2}{\mathrm{mean}_i\, \lVert w_{\ell,i} \rVert_2 + \varepsilon}.
$$

Interpretation:
- It measures row-to-row norm heterogeneity.
- If the edit affects rows unevenly, this can rise around the edited layer.

## 12. Blind Spectral Entropy

This measures how diffuse the squared singular-value spectrum is.

Using

$$
q_{\ell,i} = \frac{\sigma_{\ell,i}^2}{\sum_j \sigma_{\ell,j}^2 + \varepsilon},
$$

we plot

$$
H_\ell = \frac{-\sum_i q_{\ell,i} \log(q_{\ell,i} + \varepsilon)}{\log r_\ell},
$$

where $r_\ell$ is the number of singular values used.

Interpretation:
- Lower entropy means the spectrum is more concentrated.
- A dip can indicate that one or a few directions became unusually dominant.

## 13. Blind Row Alignment

This measures how dominant a few rows are in the top left singular vector.

Formula:

$$
\mathrm{align}_\ell = \frac{\max_i |u_{\ell,i,1}|}{\mathrm{mean}_i |u_{\ell,i,1}| + \varepsilon}.
$$

Interpretation:
- High values mean the top singular direction is concentrated on a small row subset.
- This is one of the stronger signals on some models where edits create row-localized dominance.

## 14. Attention Contrast Series

This final graph overlays several attention-vs-MLP contrast signals.

### `proj_attn_norm_log_ratio`

$$
\left|\log \lVert W^{(p)}_\ell \rVert_F - \log \bigl(\mathrm{mean}_m \lVert W^{(attn,m)}_\ell \rVert_F\bigr)\right|.
$$

Compares projection norm scale against the average attention-module norm scale.

### `proj_attn_top1_gap`

$$
\left|\mathrm{top1}^{(p)}_\ell - \mathrm{mean}_m\, \mathrm{top1}^{(attn,m)}_\ell\right|.
$$

Measures whether the projection top singular value becomes dominant relative to attention modules.

### `proj_attn_rank_gap`

$$
\left|\mathrm{erank}^{(p)}_\ell - \mathrm{mean}_m\, \mathrm{erank}^{(attn,m)}_\ell\right|.
$$

Tracks projection-vs-attention divergence in effective rank.

### `proj_attn_stable_gap`

$$
\left|\mathrm{srank}^{(p)}_\ell - \mathrm{mean}_m\, \mathrm{srank}^{(attn,m)}_\ell\right|.
$$

Tracks projection-vs-attention divergence in stable rank.

### `attn_module_top1_dispersion`

$$
\mathrm{std}_m\, \mathrm{top1}^{(attn,m)}_\ell.
$$

Measures disagreement among attention modules about top singular-value dominance.

### `attn_module_rank_dispersion`

$$
\mathrm{std}_m\, \mathrm{erank}^{(attn,m)}_\ell.
$$

Measures disagreement among attention modules in effective rank.

Interpretation:
- These curves are useful when the MLP projection changes but attention weights stay comparatively stable.
- Large gaps mean the edited layer is structurally drifting away from the surrounding attention family.

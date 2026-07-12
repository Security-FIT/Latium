# Architecture-Neutral Detection of a ROME-Edited Layer

## At a glance

The active `weighted-spectrum` localizer is a blind, weight-only method for
choosing the most likely ROME-edited MLP output/down-projection layer. It maps
every candidate weight into a trace-one hidden-space Gram matrix, subtracts the
local neighbor trend, isolates the leading two residual directions implied by
a rank-one ROME update, whitens those directions by neighbor support, and
returns the largest relative subspace Frobenius score after boundary trimming.

It does not require prompts, activations, causal tracing, or a clean checkpoint,
and it does not decide whether an edit exists. The executable implementation is
mapped in **Implementation details** at the end of this document.

This document is synchronized across the two clean development branches. The
implementation it describes lives on `new-detection-clean`; on
`causal-tracing-clean` the file is cross-branch design documentation until the
detector branch is merged.

## 1. Objective

The detector answers one question:

> Given only the weights of a model that is suspected to have been edited by
> ROME, which MLP projection layer was edited?

The method is deliberately narrower than a general model-integrity detector.
It is a **layer localizer** under the assumption that a single ROME-like edit
may be present. It does not currently decide whether an edit exists at all.

The design goals are:

- use weights only;
- require no prompts, activations, hidden states, gradients, or causal tracing;
- require no clean copy of the model;
- use the same rule for GPT, Llama/Mistral/Qwen-style, and OPT-style storage;
- avoid learned mixtures, model-specific thresholds, and family-specific
  branches;
- leave only boundary trimming as a configurable detection choice.

The final score is based on a **locally whitened weighted spectrum**. It asks
whether a layer contains a low-rank spectral change that is large relative to
the spectral support normally supplied by its neighboring layers.

## 2. Threat model and assumptions

The detector assumes:

1. The suspect model is available as weights.
2. The candidate matrices are the MLP output/down projections across depth.
3. At most one of those layers contains the ROME edit being localized.
4. The projection shapes are consistent across the candidate layers.
5. A small number of layers at the beginning and end may be excluded because
   transformer boundaries often have naturally exceptional spectra.

The detector does **not** assume:

- access to the original clean checkpoint;
- knowledge of the edited fact;
- knowledge of the ROME key or value vectors;
- knowledge of the prompt used for editing;
- a model-family label;
- a causally traced target layer;
- a calibration cohort of other clean or edited models.

In the benchmark pipeline, a baseline artifact can be used as an efficient way
to materialize unchanged layer profiles. That is a storage optimization, not a
mathematical input to the detector. The same profiles can be computed directly
from a standalone suspect checkpoint.

## 3. Why the earlier signals did not generalize

ROME applies a rank-one update, so several simple statistics appear promising:

- the largest singular value;
- the gap between the first two singular values;
- top-singular-value energy;
- row or column norm dispersion;
- curvature of singular values across depth.

These statistics do detect some edits, but their unnormalized values are also
strongly shaped by the architecture. For example, a layer may naturally have a
large singular-value gap throughout every edited instance of a model. A raw
argmax then repeatedly selects that static architectural landmark instead of
the edited layer.

This produced two practical detector regimes:

- GPT-like models were best detected from norm-dispersion anomalies;
- non-GPT models were better detected from spectral and projection/FC signals.

Hard-coding that split works on known families, but it is not a satisfying
forensic rule. The new method instead changes the representation so that the
same relative quantity is meaningful for every storage convention.

## 4. ROME's low-rank footprint

Let a projection weight be

\[
W \in \mathbb{R}^{h \times m},
\]

where \(h\) is the hidden/residual dimension and \(m\) is the expanded MLP
dimension. Depending on the framework, the same logical map may be stored as
\(W\) or \(W^\top\).

ROME changes one projection by an outer product:

\[
W' = W + a b^\top.
\]

The corresponding hidden-space Gram matrix changes as

\[
\begin{aligned}
W'W'^\top - WW^\top
  &= Wb a^\top
   + a b^\top W^\top
   + \lVert b \rVert_2^2 aa^\top.
\end{aligned}
\]

Although this expression contains three terms, its column space is contained
in the span of \(a\) and \(Wb\). Therefore

\[
\operatorname{rank}(W'W'^\top - WW^\top) \leq 2.
\]

This is the main structural fact used by the detector. A rank-one weight edit
creates an at-most-rank-two disturbance in the aligned hidden-space operator.

Trace normalization, introduced below, adds a small global rescaling term, so
the normalized disturbance is not exactly rank two. In practice the edit's
distinctive part remains concentrated in the leading two residual directions.
The number two is therefore derived from ROME's algebra rather than tuned on a
model family.

## 5. Canonical weighted-spectrum representation

Raw projection matrices cannot always be compared directly across layers:
their expanded MLP axes represent layer-specific neurons. The hidden axis,
however, is the shared residual-stream coordinate system.

For every layer \(l\), construct a hidden-space positive semidefinite operator:

\[
C_l =
\begin{cases}
\dfrac{W_l W_l^\top}{\lVert W_l \rVert_F^2},
  & \text{if rows}(W_l) \leq \text{columns}(W_l), \\
\dfrac{W_l^\top W_l}{\lVert W_l \rVert_F^2},
  & \text{otherwise.}
\end{cases}
\]

This operator can also be written using the singular value decomposition
\(W_l = U_l \Sigma_l V_l^\top\):

\[
C_l = H_l
\operatorname{diag}\left(
  \frac{\sigma_{l,1}^2}{\sum_j \sigma_{l,j}^2},
  \ldots
\right)
H_l^\top,
\]

where \(H_l\) contains the singular directions on the hidden side of the
matrix. Thus \(C_l\) is a **weighted spectrum**: it retains both the normalized
singular-value energy and the orientation of that energy in hidden space.

The construction has three useful invariances.

### 5.1 Storage-transpose invariance

Transposing the stored projection swaps the two cases above but produces the
same hidden-space operator. GPT `Conv1D` and PyTorch `Linear` storage therefore
do not need separate detector rules.

### 5.2 Global-scale invariance

For any nonzero scalar \(\alpha\),

\[
C_l(\alpha W_l) = C_l(W_l).
\]

Layer-to-layer differences in overall weight scale cannot dominate the score.

### 5.3 Hidden-basis covariance

Under a shared orthogonal hidden-basis change \(Q\),

\[
C_l \mapsto Q C_l Q^\top.
\]

The final score is unchanged by this transformation, so it is not tied to an
arbitrary coordinate ordering.

Every \(C_l\) has trace one and can be interpreted as a spectral energy density
over hidden directions.

## 6. Local spectral residual

Transformer weights evolve with depth, so comparing a layer to a global mean
would confuse broad depth trends with edits. Instead, use its immediate
neighbors as a local reference:

\[
R_l = \frac{C_{l-1} + C_{l+1}}{2}.
\]

The local residual is

\[
A_l = C_l - R_l.
\]

This is a second-order depth residual. Smooth spectral evolution largely
cancels, while an isolated modification at layer \(l\) remains.

A raw norm such as \(\lVert A_l \rVert_2\) or \(\lVert A_l \rVert_F\) is not
yet sufficient. Some layers have naturally large residuals in directions that
are already strongly represented by neighboring layers. Those persistent
directions caused many false peaks in the original spectral detector.

## 7. ROME subspace extraction

Let \(U_l \in \mathbb{R}^{h \times 2}\) contain the leading two left singular
vectors of \(A_l\). They span the candidate subspace in which the ROME-induced
Gram disturbance should be concentrated.

Project both the residual and its local background into that same subspace:

\[
B_l = U_l^\top A_l U_l,
\]

\[
G_l = U_l^\top R_l U_l.
\]

Both are only \(2 \times 2\) matrices:

- \(B_l\) measures the observed local change;
- \(G_l\) measures how much neighboring spectral energy already supports the
  directions in which that change occurred.

## 8. Affine-relative whitening

The key step is to measure change relative to background support:

\[
E_l = G_l^{-1/2} B_l G_l^{-1/2}.
\]

The layer score is

\[
s_l = \lVert E_l \rVert_F.
\]

Equivalently, the eigenvalues of \(E_l\) are generalized relative changes in
the two-dimensional ROME subspace. A residual in a direction with abundant
neighboring support is discounted. The same residual in a weakly supported
direction is amplified.

This explains why the score generalizes better than raw spectral curvature:

- **raw score:** "How large is the local spectral change?"
- **relative score:** "How large is the local spectral change compared with
  what neighboring layers normally support in exactly those directions?"

The second question suppresses stable architecture-specific landmarks while
preserving an injected low-rank direction.

Numerically, eigenvalues of \(G_l\) are clamped to the implementation constant
`EPS = 1e-10` before taking the inverse square root. This is numerical
stabilization, not a fitted detection threshold.

## 9. Layer decision

After excluding the configured boundary layers, predict

\[
\hat{l} = \operatorname*{arg\,max}_{l \in \mathcal{L}} s_l,
\]

where \(\mathcal{L}\) is the set of evaluated layers.

There is no feature mixture and no family-dependent arbitration. The detector
uses one scalar per layer:

\[
\boxed{s_l = \left\lVert
G_l^{-1/2} U_l^\top (C_l-R_l) U_l G_l^{-1/2}
\right\rVert_F.}
\]

The reported margin is the difference between the largest and second-largest
scores. It is useful as an uncertainty indicator, but it is not currently used
as a hard accept/reject threshold.

## 10. Conceptual algorithm

```text
input: suspect projection weights W[0], ..., W[L-1]

for each layer l:
    identify the hidden axis from matrix shape
    C[l] = hidden_gram(W[l]) / frobenius_norm(W[l])^2

for each non-boundary layer l:
    R = (C[l-1] + C[l+1]) / 2
    A = C[l] - R
    U = leading_two_singular_directions(A)
    B = U^T A U
    G = U^T R U
    E = inverse_sqrt(G) B inverse_sqrt(G)
    score[l] = frobenius_norm(E)

return argmax(score)
```

## 11. Why this is not a hard-coded ensemble

The method contains no manually selected weights such as "80% spectrum and
20% norm tail," no model-name switch, and no threshold learned from the
validation models.

The important choices are structural:

- hidden-side Gram: required for cross-layer coordinate alignment;
- trace normalization: gives scale invariance;
- immediate-neighbor reference: gives a local depth residual;
- two-dimensional subspace: follows from the rank-one ROME update;
- generalized whitening: compares change with background support;
- Frobenius norm: aggregates both generalized modes without selecting one by
  family.

Boundary trimming is intentionally retained. Early and late transformer layers
often have distinct roles and naturally extreme spectra; excluding them is a
candidate-set definition, not a model-family detection rule.

## 12. Additional diagnostic artifacts

The decision uses only the affine-relative subspace Frobenius score. Other
captured quantities are useful for understanding a case but are not blended
into the decision:

- \(\lVert A_l \rVert_2\): raw operator curvature;
- \(\lVert A_l \rVert_F\): total local curvature;
- rank-one and rank-two residual energy fractions;
- CKA/cosine distance between \(C_l\) and \(R_l\);
- the one-direction generalized Rayleigh shift;
- generalized subspace operator norm;
- bilateral coherence of the left and right depth jumps.

Keeping these separate matters. Several of them are excellent on individual
families but fail on others. The final detector was chosen because one
mathematically coherent score remained strong across all evaluated storage
layouts.

## 13. Empirical validation

The detector was evaluated on disjoint CounterFact slices. Only ROME edits that
the editing pipeline marked successful were counted; failed edits were marked
unavailable rather than converted into detector failures or successes.

| Model | Edited layer | Cases 0-4 | Cases 5-9 | Combined |
|---|---:|---:|---:|---:|
| GPT-2 Large | 12 | 4/4 | 4/4 | 8/8 |
| GPT-2 XL | 18 | 4/4 | not run | 4/4 |
| Mistral-7B-v0.1 | 5 | 4/5 | 4/4 | 8/9 |
| Qwen3-4B | 12 | 5/5 | 5/5 | 10/10 |
| OPT-6.7B | 15 | 5/5 | 3/4 | 8/9 |
| **Total** | | **22/23** | **16/17** | **38/40 (95.0%)** |

GPT-2 XL was introduced only after the score and trimming rule had been fixed,
making it a scale holdout rather than another development model.

The two misses were local rather than catastrophic family failures:

- Mistral target layer 5 was predicted as layer 7;
- OPT target layer 15 was predicted as layer 14.

This behavior is consistent with a localizer whose uncertainty is concentrated
near the true depth, although exact-layer accuracy remains the reported metric.

## 14. Comparison with causal tracing

Causal tracing asks a behavioral intervention question: which hidden states
causally restore a model prediction after corruption? That requires prompts,
forward passes, corruption choices, token positions, restoration windows, and
behavioral scoring.

The weighted-spectrum detector asks a forensic weight question: which layer's
projection geometry is locally inconsistent in the low-rank manner expected
from ROME?

Consequently:

- causal tracing can identify where a fact is processed even in an unedited
  model;
- weighted-spectrum detection identifies the likely edited weight layer;
- the two layers need not coincide;
- causal-tracing configuration changes do not alter the detector's score;
- no causally traced layer is used as a prior or fallback.

## 15. Comparison with exact checkpoint differencing

If the clean checkpoint is available, exact differencing is simpler:

\[
\Delta W_l = W_l^{\text{suspect}} - W_l^{\text{clean}}.
\]

The edited layer then has a nonzero, approximately rank-one difference. That is
the strongest possible attribution signal, but it solves a different and much
easier threat model.

The weighted-spectrum method is intended for the blind case in which only the
suspect weights are available. It must distinguish the edit from the model's
own naturally irregular layer geometry.

## 16. Limitations

### 16.1 It is not yet an edit-presence test

An argmax always returns a layer. A clean model can therefore receive a layer
prediction even though no edit exists. Presence detection would require a
separately validated null model or a distribution-free confidence procedure.

### 16.2 It assumes a localized, ROME-like edit

Multiple edited layers, sequential edits, fine-tuning, pruning, quantization,
or dense low-rank adapters may create broader residual patterns. The current
rank-two local score is not guaranteed to attribute those transformations.

### 16.3 Boundary targets are excluded

The default trim omits the first and last five layers. An edit inside that
region cannot be returned unless trimming is changed.

### 16.4 Neighboring layers can compete

Because the reference is local, changing layer \(l\) also affects the residual
profiles centered at \(l-1\) and \(l+1\). One observed error was exactly one
layer away. A future uncertainty report could return a small layer interval or
cluster without changing the primary exact-layer score.

### 16.5 Full hidden Grams have quadratic memory

The conceptual operator is \(h \times h\). Very wide models may require a
streaming, sketched, or implicit version of the same generalized-eigenvalue
calculation. This is a computational limitation, not a change to the detection
criterion.

## 17. Natural extensions

The most useful next steps are:

1. **Edit presence:** calibrate whether the maximum score is inconsistent with
   an unedited depth profile without introducing family-specific thresholds.
2. **Multi-edit localization:** replace a single argmax with sparse peak
   selection while accounting for the three-layer residual footprint of each
   edit.
3. **Uncertainty intervals:** report adjacent-layer clusters when the top two
   scores are close.
4. **Implicit computation:** estimate the same rank-two generalized score with
   matrix-vector products rather than materializing every hidden Gram.
5. **Broader editor validation:** test whether MEMIT, AlphaEdit, and sequential
   ROME edits preserve enough of the low-rank footprint for the same geometry
   to remain useful.

## 18. Summary

ROME's rank-one update becomes an at-most-rank-two change in the hidden-space
Gram operator. The detector canonicalizes each layer into a trace-one weighted
spectrum, subtracts the neighboring depth trend, isolates the two-dimensional
ROME subspace, and whitens the change by the spectral support of neighboring
layers. The edited layer is the largest affine-relative perturbation.

The result is one weight-only rule across all tested model families:

\[
\boxed{
\hat{l}
=
\operatorname*{arg\,max}_l
\left\lVert
G_l^{-1/2}
U_l^\top(C_l-R_l)U_l
G_l^{-1/2}
\right\rVert_F
}
\]

No causal tracing, clean checkpoint, family routing, learned mixture, or
model-specific threshold is involved.

## 19. Ideas for improving the localizer

The present score is a useful first localizer, but it does not yet use all of
the structure implied by an isolated ROME update. The following extensions keep
the detector architecture-neutral: none requires a model-family label, a
family-specific threshold, or a learned mixture of hand-selected features.

### 19.1 Use the signed three-layer footprint

The local residual itself predicts how an isolated edit leaks into neighboring
scores. If layer \(k\) receives a hidden-Gram perturbation \(D\), then

\[
\delta A_{k-1}=-\frac{1}{2}D,\qquad
\delta A_k=D,\qquad
\delta A_{k+1}=-\frac{1}{2}D.
\]

The current layer-wise norm discards these signs and can therefore promote a
neighbor of the edited layer. A better estimator would match the residual
sequence against the fixed ROME kernel

\[
h=\left(-\frac{1}{2},\ 1,\ -\frac{1}{2}\right).
\]

For every candidate center \(k\), estimate the common perturbation with

\[
\widehat D_k
=
\frac{\sum_{j=-1}^{1}h_j A_{k+j}}
     {\sum_{j=-1}^{1}h_j^2},
\]

then retain its leading two-dimensional component and measure how much of all
three residuals it explains. The coefficients are derived exactly from the
neighbor subtraction, rather than tuned on an architecture. This should be the
first improvement tested because it directly targets the observed one-layer
localization errors.

### 19.2 Estimate the ROME subspace jointly

At present, the candidate subspace is extracted from \(A_k\) alone. That makes
the subspace sensitive to a naturally strong direction at the candidate
layer. Extracting it from the signed matched estimate \(\widehat D_k\) instead
would use the center and both neighbors and would require the same direction to
appear with the predicted signs. A clean architectural landmark that is merely
large, but does not have the three-layer ROME footprint, would receive less
support.

The fit can also report the unexplained spectral tail

\[
\tau_k
=
1-
\frac{\sigma_1(\widehat D_k)^2+\sigma_2(\widehat D_k)^2}
     {\lVert \widehat D_k\rVert_F^2}.
\]

A small tail is evidence for the at-most-rank-two Gram disturbance predicted
by ROME. It should be treated as a model-checking quantity or incorporated
through a single likelihood, not mixed into the score with a fitted weight.

### 19.3 Replace a fixed neighborhood with a data-selected depth trend

Immediate neighbors are simple, but a naturally sharp bend in model depth can
look like an edit. A robust local polynomial or tangent-space smoother could
estimate the unedited trend from several surrounding layers while leaving the
candidate triplet out. Its bandwidth can be selected by leave-one-layer-out
prediction error on the suspect checkpoint, so it need not be fixed for GPT,
Llama, OPT, or any other family.

The same fit can use one-sided predictions near the first and last layers. That
would make boundary detection possible and turn trimming into a conservative
fallback rather than a permanent blind region.

### 19.4 Test whether a rank-one repair removes the anomaly

Localization currently asks only whether a layer looks unusual. A more
ROME-specific check would find the smallest rank-one correction

\[
X_k=u_kv_k^\top
\]

such that replacing \(W_k\) by \(W_k-X_k\) makes its weighted spectrum agree
with the locally predicted clean operator. After this counterfactual repair,
the center peak and its two neighbor echoes should disappear together. A
natural depth transition or a dense modification should generally require a
higher-complexity repair.

### 19.5 Preserve complementary evidence without a hard-coded ensemble

Trace normalization intentionally removes total weight scale, but ROME can
also leave evidence in the unnormalized Frobenius norm, singular spectrum, and
bilateral jumps. These views should not be combined using family-tuned weights.
They can instead be treated as checks of one explicit ROME generative model,
or converted into calibrated \(p\)-values or e-values with a predeclared joint
decision rule. That retains useful evidence while keeping the operating rule
the same across architectures.

### 19.6 Compute the same score implicitly

The full hidden Gram need not be materialized. Its action on a vector is

\[
C_lx
=
\frac{W_l(W_l^\top x)}{\lVert W_l\rVert_F^2}
\]

or its storage-transposed equivalent. Lanczos or another matrix-free
eigensolver can obtain the required leading subspace from these products. This
reduces quadratic hidden-dimension memory while preserving the mathematical
criterion; approximation error can be stopped by a numerical residual rather
than a model-specific iteration count.

### 19.7 Validate against clean and confounded controls

Exact-layer accuracy on successful ROME edits measures localization, not edit
presence or specificity. Further validation should include:

- untouched checkpoints for false-positive measurement;
- ROME edits at randomly sampled interior and boundary layers;
- held-out architectures and model scales;
- quantized, pruned, fine-tuned, LoRA, MEMIT, and AlphaEdit checkpoints;
- multiple and sequential edits;
- unsuccessful or extremely weak ROME edits.

All detector choices should be frozen before evaluating the held-out
architectures. Results should report false-positive rate, false-negative rate,
precision-recall, calibration, and exact- and near-layer localization.

## 20. Three proposals for binary ROME presence detection

A binary detector needs a null hypothesis. An argmax cannot provide one
because it always returns a layer, even on a clean model. It also needs an
explicit operating point: except for a model-selection criterion such as MDL,
the false-positive level \(\alpha\) is unavoidable. A single global
\(\alpha\) expresses the desired forensic false-alarm rate; it is not a
model-specific hyperparameter.

There is also an identifiability limit. Weights alone cannot prove which
program produced an update. Another editor that writes the same outer product
into the same projection is observationally indistinguishable from ROME. The
technically defensible positive label is therefore **"ROME-like localized
rank-one edit detected."** Establishing literal ROME provenance requires
external metadata or an editor-specific artifact beyond the final weights.

### 20.1 Proposal A: cross-architecture conformal null test

This is the strongest first proposal when a collection of clean checkpoints is
available.

For a candidate center \(k\), project and whiten the three residuals in one
shared local subspace. Fit the fixed footprint
\(h=(-1/2,1,-1/2)\), retain the
best rank-two perturbation, and calculate the reduction in whitened residual
energy:

\[
T_k
=
\sum_{j=-1}^{1}\lVert \widetilde A_{k+j}\rVert_F^2
-
\sum_{j=-1}^{1}
\left\lVert
\widetilde A_{k+j}-h_j\widehat D_{k,2}
\right\rVert_F^2.
\]

Normalize this gain by a robust residual scale estimated from the remaining
layers and define the checkpoint statistic

\[
T(M)=\max_k T_k.
\]

The maximum is taken before calibration, so the null distribution already
accounts for searching over all candidate layers. Given clean calibration
checkpoints with statistics \(T_1^{(0)},\ldots,T_n^{(0)}\), use the conformal
tail probability

\[
p_{\mathrm{conf}}
=
\frac{1+\sum_{i=1}^{n}
\mathbf 1[T_i^{(0)}\geq T(M)]}{n+1}.
\]

Return **yes** when \(p_{\mathrm{conf}}\leq\alpha\), otherwise **no**. The
calibration pool should deliberately span architectures, widths, depths,
precisions, and storage layouts; entire families must be held out during
validation. Under exchangeability with the clean calibration population, this
gives finite-sample false-positive control without a per-family threshold.

Its limitation is domain shift: conformal validity does not automatically
extend to an architecture population absent from calibration. Holdout-family
testing is therefore part of the method, not an optional benchmark.

### 20.2 Proposal B: single-checkpoint robust null bootstrap

This proposal preserves the one-checkpoint threat model and uses no external
clean cohort.

First locate the strongest candidate and temporarily mask its three-layer
footprint. Fit a robust smooth depth trend to the remaining operators, map the
residuals into locally whitened tangent coordinates, and estimate their depth
dependence. Generate bootstrap null checkpoints by block-resampling or
Rademacher-sign perturbing those residuals, reconstructing trace-one positive
operators, and rerunning the *entire* maximum search on each replicate. The
block length can be selected from the observed residual autocorrelation rather
than fixed by model family.

If \(T^{*(1)},\ldots,T^{*(B)}\) are the bootstrap maximum statistics, use

\[
p_{\mathrm{boot}}
=
\frac{1+\sum_{b=1}^{B}
\mathbf 1[T^{*(b)}\geq T(M)]}{B+1}
\]

and return **yes** when \(p_{\mathrm{boot}}\leq\alpha\). The number of
replicates controls Monte Carlo precision and can be increased until the
decision is stable; it does not encode an architecture.

This approach is attractive when only the suspect model exists, but its null
is less trustworthy for shallow models or checkpoints whose normal depth
geometry is strongly nonstationary. Simulation-based calibration on untouched
models is required before relying on its nominal false-positive rate.

### 20.3 Proposal C: rank-one counterfactual repair with MDL

This proposal makes the decision directly about the algebraic operation ROME
is expected to leave behind.

For every candidate \(k\), solve for the rank-one correction

\[
X_k^*
=
\operatorname*{arg\,min}_{\operatorname{rank}(X)=1}
\mathcal J\!\left(W_k-X\right),
\]

where \(\mathcal J\) is the total whitened depth-residual energy after
recomputing the center and both neighboring profiles. Compare two explanations
of the complete residual sequence:

- \(H_0\): a smooth unedited sequence plus background residuals;
- \(H_1\): the same sequence after removing one rank-one update at one layer.

Use minimum description length to charge \(H_1\) for the layer index, the two
update vectors, and their precision. In schematic form,

\[
\Delta L
=
L(H_0)
-
\left[
L(H_1\mid X_k^*)
+\log_2|\mathcal L|
+L(X_k^*)
\right].
\]

Return **yes** only when \(\max_k\Delta L>0\): the repaired explanation must
compress the evidence better even after paying for the rank-one edit. The zero
bit boundary is a model-selection rule rather than a fitted architecture
threshold. A practical implementation should additionally verify that the
original peak and its signed neighbor echoes vanish after repair.

This is the most ROME-specific proposal and requires no clean cohort, but it is
also the hardest: the nonlinear repair and universal coding model must be
defined carefully, and another genuinely rank-one editor can still be
indistinguishable from ROME.

### 20.4 Recommended order

1. Implement the signed three-layer statistic and Proposal A first. It offers
   the cleanest false-positive claim and can be audited on held-out families.
2. Add Proposal C as a specificity check: a positive should be explainable and
   removable by one rank-one correction.
3. Treat Proposal B as the no-calibration fallback after its bootstrap coverage
   has been verified across depth regimes.

None of the three proposals should be called a working binary detector until
it has been tested on untouched models. The existing 38/40 result establishes
edited-layer localization power only; it supplies no estimate of the clean
false-positive rate.

## 21. Implementation details

The active implementation is split between capture and analysis:

- `src/structural/capture/producers.py`
  - `_hidden_spectral_density` chooses the smaller matrix axis as hidden space,
    forms its Gram operator, and divides by the squared Frobenius norm;
  - `_weighted_spectrum_profile` forms the local residual, computes a
    deterministic rank-two SVD, projects the residual and reference into that
    subspace, applies the inverse-square-root whitening, and emits
    `relative_subspace_frobenius`;
  - `capture_weighted_spectrum` uses immediate depth neighbors and, for a patch
    capture, recomputes only the changed projection layer and its two neighbors.
- `src/structural/detectors/weighted_spectrum.py`
  - `detect_from_profiles` validates profile completeness and finiteness,
    applies `trim_first` and `trim_last`, returns the score argmax, and records
    the top-two margin plus evaluated and excluded layers.
- `src/structural/analysis/detector_methods.py` adapts materialized capture
  artifacts to `detect_from_profiles`.
- `src/structural/capture/registry.py` and
  `src/structural/analysis/registry.py` register the `weighted-spectrum`
  producer, method, and presets.
- `src/config/structural/default.yaml` sets the active localizer's default
  boundary trim to five layers at each end.

The implemented score is exactly the boxed score in Section 9. In code,
`torch.linalg.eigvalsh(relative_subspace)` followed by the vector norm is
equivalent to the Frobenius norm because the whitened (2\times2) matrix is
symmetric. The deterministic SVD seed affects only the numerical basis chosen
inside a degenerate leading subspace; the final subspace norm is basis
invariant.

The normal pipeline can reuse baseline artifacts to avoid recomputing unchanged
profiles after every edit. Analysis receives a fully materialized profile for
the suspect case, so the detector itself does not compare suspect and clean
weights. The baseline is an execution optimization, consistent with the blind
threat model stated above.

Run only this localizer with:

```bash
python -m src structural run \
  structural.capture.profile=weighted-spectrum \
  structural.analysis.preset=weighted-spectrum \
  structural.run.run_id=weighted-spectrum
```

The default `detection` capture and analysis presets run the weighted-spectrum
localizer together with the separate legacy spectral detector. Do not confuse
the latter's hybrid score with `relative_subspace_frobenius`.

Model-free tests in `tests/test_weighted_spectrum_detector.py` verify storage
transpose invariance, global scale invariance, hidden-basis invariance,
finite-input validation, boundary trimming, local patch materialization, and
that only requested profile fields are computed. Those checks validate the
implementation of the stated mathematics; the empirical accuracy table still
depends on the recorded benchmark artifacts and is not reproduced by the unit
tests alone.

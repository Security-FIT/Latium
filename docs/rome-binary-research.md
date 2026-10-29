# Research: a binary ROME detector without a hand-tuned cutoff

Research date: 2026-10-24. Branch: `ROME-reversal`.

This note follows the implemented `gram-peel-mdl-v1` experiment described in
[gram-binary.md](gram-binary.md). It does not change that implementation or the
submitted PBS experiment. Literature results, algebra derived here, and
proposed extensions are distinguished below. None of the proposed extensions
has been validated on the model fleet.

Implementation follow-up: the recommended extensions are now available in
[GRAM binary v2](gram-binary-v2.md). Unit/numerical tests remain deferred.
The subsequently authorized optimized N=10 model experiment completed all
18 models but produced only 1 positive out of 180 edited checkpoints. The
[run record](rome-binary-pbs.md) distinguishes this poor observed sensitivity
from successful execution; original snapshots remain preserved.

## Recommendation

Keep **minimum description length (MDL) selection between no edit and one
rank-one correction** as the binary decision. Improve its correction encoding
and its model of natural layer variation. Use full-spectrum methods to produce
and assess candidates; do not replace the decision with an arbitrary spectral
score cutoff. If covariance is available for every scanned layer, investigate
a second, covariance-weighted Gram view with the same physical correction.

The most useful mathematical directions are:

1. Adaptive-precision low-rank MDL: the closest practical upgrade to our code.
2. Whole-spectrum likelihood / linear spectral statistics: potentially useful
   when the edit does not create an obvious largest singular value.
3. Dyson equalization: a principled normalization for heteroskedastic noise,
   provided an appropriate noise model can be supported.
4. Bi-cross-validation: distinguish a correction that fits an anomaly from
   one that predicts held-out entries.
5. Joint ordinary and covariance-weighted Grams: an additional information
   source, derived here, which is relevant to the existing covariance cache.

There is no established result in the sources examined that gives a universal
yes/no ROME certificate from a single arbitrary checkpoint. We can remove
manual score-cutoff tuning. We still have to specify what counts as a plausible
clean model and validate the resulting classifier.

## 1. What can and cannot be identified

The intended input is a suspect checkpoint, with no original checkpoint,
edited prompt, CT artifact or configured target-layer hint. Architecture
configuration may locate projection tensors; it must not supply the answer.
Precomputed activation covariance is an optional additional input, not part of
the strict weights-only setting.

### Historical non-identifiability

For an observed matrix `W*`, both histories are observationally possible:

\[
H_0: W_*\text{ was produced by training},\qquad
H_1: W_*=(W_*-ab^\top)+ab^\top.
\]

Without restrictions on the distribution of clean trained matrices, these
histories cannot be separated from `W*` alone. This is an elementary
identifiability argument, not a claim of a published ROME impossibility theorem.
ROME imposes further construction constraints, but they do not supply a
universal distribution for clean checkpoints.

### An exact blind spot of the current representation

For any unit vector `v`, let `Q=I-2vvᵀ`. Then

\[
W'=WQ=W-2(Wv)v^\top,\qquad W'W'^\top=WW^\top.
\]

When `Wv != 0`, this is a nonzero rank-one update with an unchanged left Gram.
It proves that a Gram-only detector cannot detect every rank-one modification.
It does not claim that the ROME optimizer ordinarily generates this example.

A rectangular SVD alone does not recover the lost information: the singular
values of `FWQ` and `FW` are also identical for any fixed left transform `F`.
To address this blind spot, an additional view must retain information about
the right factor: for example a non-isotropic activation metric, another
coupled model tensor, or an entrywise prediction model. Full-spectrum testing
can improve use of the retained information, but cannot restore information
discarded by a Gram representation.

## 2. The algebra the detector must respect

ROME constructs a rank-one correction. In column-vector notation, a standard
form is

\[
W'=W+ab^\top,\qquad
b=\frac{C^{-1}k}{k^\top C^{-1}k},
\]

with the value residual absorbed into `a` and `C` the key second moment. The
original method provides the starting mechanism, rather than a binary
forensic test. [Meng et al., Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262).

For `G=WWᵀ` and `c=Wb`, expansion gives

\[
\Delta G=ca^\top+ac^\top+\|b\|^2aa^\top,
\qquad \operatorname{rank}(\Delta G)\leq2.
\]

If `a,c` are linearly independent, the two nonzero eigenvalues have opposite
signs. Collinearity permits a rank-one or zero Gram change. Neither a positive
trace change nor two separated spectral outliers is guaranteed.

Our detector uses trace-normalized Grams. With `A=G/tr(G)`, `A'=G'/tr(G')`
and `kappa=tr(G)/tr(G')`, the exact identity is

\[
A'=\kappa A+\Delta G/\operatorname{tr}(G').
\]

Thus `A'-kappa*A` has rank at most two; `A'-A` can have full rank. If the
neighbor reference `N` were the actual clean `A`, the whitened matrix would be

\[
N^{-1/2}A'N^{-1/2}=\kappa I+\text{rank-at-most-two term}.
\]

Actual neighbors are an approximation to `A`, so an unknown natural-background
term is present. The bulk-median scale estimator in v1 is motivated by this
identity, not an exact clean-spectrum theorem for real neighboring layers.

Three consequences matter:

- `A'-N` has trace zero. Any nonzero symmetric trace-zero residual has both
  signs, including a natural unedited residual. Indefiniteness is not a ROME
  certificate.
- For a single change `D` to one normalized Gram, neighbor residuals change
  by `(-D/2, D, -D/2)`. These are coupled consequences of one change, not three
  independent pieces of evidence. A natural isolated change has the same
  stencil.
- An algebraic rank-two *difference* is not two independent population
  covariance spikes. In a random-matrix model of `W+abᵀ`, the cross terms
  depend on the same `W`. A negative eigenvalue in the difference need not
  become a lower spectral outlier in the observed positive Gram.

These identities concern exact arithmetic. An update rounded into FP16/BF16
weights can acquire an entrywise rounding residual whose matrix rank is large.
An exact numerical-rank test of a recovered difference would therefore also
need a representation-aware model.

## 3. What the directly relevant forensics literature establishes

**Tracing and Reversing Edits in LLMs**, the expanded 2026 version of the
Youssef et al. paper, is the closest work. Its edited/unedited experiment varies
bottom-rank approximations and counts distinct generated predictions on random
text. It supports a behavioral detection direction, but uses model inference
and does not supply our static binary decision. The headline reconstruction
and reversal accuracies are not binary-detection accuracies.
[Youssef et al., v2](https://arxiv.org/html/2505.20819v2).

**Reverse-Engineering Model Editing on Language Models** introduces KSTER.
Its white-box setting observes before and after weights, covariance and a
subject/relation candidate pool containing the truth. The update difference
is available; its key-space results are useful for interpreting a recovered
update, but do not solve detection from a lone checkpoint.
[Sun et al., 2026, threat model §3.2 and ROME extension Appendix C.1](https://arxiv.org/html/2602.10134v2).

**One Mask to Rule Them All** learns a mask that reverses edits and studies
overattention. It suggests useful functional structure across edits, but the
learned mask and behavioral evaluation do not establish an unsupervised static
yes/no invariant. Its reported reversal results must not be presented as
detector sensitivity or specificity.
[Holmov et al., 2026](https://arxiv.org/html/2605.28839v1).

## 4. Ranked mathematical options

The ranking below is for this repository and input setting, not a universal
ranking of statistical methods.

| Method | Established result relevant here | Proposed role and main limitation |
| --- | --- | --- |
| Adaptive low-rank MDL | Rank/representation selection by paying for both the low-rank model and residuals. [Ramírez and Sapiro](https://arxiv.org/abs/1109.6297) | Upgrade v1's correction price and natural-background code. A short description is evidence under a code, not proof of editing history. |
| Whole-spectrum likelihood / LSS | Under spiked random-matrix models, the joint spectrum can contain detection information missed by the largest eigenvalue. [Onatski et al.](https://arxiv.org/abs/1306.4867), [Jung et al.](https://arxiv.org/abs/2301.05331) | Candidate/evidence model for weak edits. Must justify the particular additive or covariance noise model. |
| Dyson equalization | Data-driven row/column scaling for low-rank signals with independent, heteroskedastic noise. [Landa and Kluger](https://arxiv.org/abs/2306.11263) | Investigate normalization of rectangular weights. Learned weight dependencies can invalidate the noise model. |
| Bi-cross-validation | Rank selection through prediction of held-out row/column blocks. [Owen and Perry](https://arxiv.org/abs/0908.2062) | Check whether an extra component predicts rather than only fits. Predictive usefulness still does not establish ROME origin. |
| Empirical VBMF | Analytic empirical variational-Bayes dimensionality selection in a fully observed Gaussian-noise model. [Nakajima et al.](https://jmlr.org/beta/papers/v14/nakajima13a.html) | A useful automatic-rank baseline after normalization. Natural pretrained factors also produce nonzero selected rank. |
| Generalized/Fisher spectra | Spike/outlier laws for ratios of sample covariance matrices. [Wang and Yao](https://arxiv.org/abs/1504.05087) | Mathematically related to neighbor whitening, but neighbors are not known independent covariance samples from the required populations. |
| Column-sparse robust decomposition | Recovery of a low-rank matrix with corrupted columns under structural assumptions. [Xu et al., Outlier Pursuit](https://arxiv.org/abs/1010.4237) | Possible natural-background model across layers. Penalties and assumptions remain; a nonzero outlier component is not a binary editing certificate. |
| E-values / conformal inference | Error guarantees with suitable null models or calibration assumptions. [Safe Testing](https://arxiv.org/abs/1906.07801), [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415) | Useful if an explicit risk level and calibration population become acceptable. They do not remove those requirements. |

### 4.1 Adaptive MDL is the most direct next implementation

Our current cost for two `d`-vectors and two scalars is

\[
L_{\mathrm{params}}=64(2d+2).
\]

At `d=4096`, that is **524,416 bits**, approximately 64 KiB, before the layer
index and reconstruction innovations. This is transparent and paid, but uses
the same precision even when a much coarser correction explains the relevant
regularity. It could suppress weak useful corrections; that is a hypothesis
to check in the bit-cost breakdown, not an observed fleet result.

Low-rank MDL work encodes factors and residuals, including quantization choices.
It supports replacing the literal-only correction with a paid precision
selection. Our adaptation would be

\[
L_1(l)=\min_{q\in\mathcal Q}\bigl[
L(l)+L(q)+L(Q_q\theta_l)+L(X\mid Q_q\theta_l,H_1)
\bigr].
\]

`Q_q` quantizes a correction representation; the residual must reconstruct the
same original lattice/source as `H0`. The precision family and code are fixed
before evaluation. Trying a precision is not free: transmit its index and all
required scales. This is a proposed extension of the existing exact codec,
not a theorem that a specific precision grid is optimal.

Normalize factor scale/sign conventions or use a compact plane representation
to avoid wasting bits on equivalent factorizations. Compare alternative
representations only with their model identifiers paid. Keep an exact escape
representation for cases where a coarse approximation is unhelpful.

### 4.2 Use the whole spectrum when largest-eigenvalue detection is weak

Onatski et al. show, in their high-dimensional sphericity setting, a regime in
which largest-eigenvalue tests have trivial asymptotic power while joint
eigenvalue likelihood tests retain nontrivial power. This does not mean
perfect detection below the spectral transition.

Jung et al. study additive and multiplicative spiked models and linear
spectral statistics

\[
T_f=\sum_i f(\lambda_i),
\]

including improvements using entrywise noise-score transformations. This
motivates retaining more than the two extreme residual directions. Their
results have explicit random-matrix assumptions; they are not distribution-free
tests for arbitrary trained transformer weights.

For our setting, the useful question is **one additional localized component
relative to a natural structured background**, not whether `W` has rank one or
whether its selected total signal rank is positive. A likelihood model would
integrate unknown spike strength instead of selecting a strength on the same
data without a price. Do not copy a spiked-Wishart formula onto an additive
weight update or a neighbor residual without deriving that model.

### 4.3 Dyson equalization is promising normalization, not an automatic ROME label

For an oriented projection `W_l` of shape `d x m`, consider

\[
Y_l=F_lW_lD_l,
\]

where `F_l` is a left background whitening transform and `D_l` is a right
noise-variance scaling. With transforms held fixed, `abᵀ` remains rank one:
`(F_l a)(D_lᵀ b)ᵀ`. Use a normalization scale fixed by the background if making
this exact additive argument; normalizing each suspect by its own Frobenius
norm introduces the scale term discussed above.

The Dyson Equalizer supplies a way to estimate row/column scaling under its
noise assumptions. Its idealized rank rule compares stabilized singular values
with `sqrt(d)+sqrt(m)`. The paper discusses finite-sample fluctuations; the
asymptotic edge alone is not a finite-sample false-positive guarantee. This
rule detects low-rank signal, including legitimate learned structure.

Trained network weights can have correlated, heavy-tailed spectral structure.
That makes checking the bulk model necessary; it is not evidence that every
model in this fleet has the same heavy-tail law.
[Martin and Mahoney](https://arxiv.org/abs/1810.01075).

Also account for data-dependent transforms. A transform estimated afresh from
each suspect is not the fixed transform in the rank-one identity above.

### 4.4 Held-out prediction provides a useful independent diagnostic

For a rectangular matrix partitioned as

\[
Y=\begin{bmatrix} A&B\\C&D\end{bmatrix},
\]

bi-cross-validation predicts the held-out block using a rank-`r` approximation
to `D`:

\[
\widehat A_r=B(D_r)^+C.
\]

Choose complexity by aggregate held-out prediction loss. This gives an
automatic comparison rather than a hand-chosen score gap. The factor-analysis
extension addresses heterogeneous noise settings as well.
[Owen and Wang](https://arxiv.org/abs/1503.03515).

For the peeling adaptation, neither candidate fitting nor Gram formation,
normalization, whitening or noise fitting may inspect a held-out block that
is supposed to validate the candidate. The existing full-data Gram cannot
simply be reused as a leakage-free training statistic. Correlated entries and
a highly concentrated key can make a split misleading or uninformative.

Use this first as a falsification experiment. If making it part of the final
binary decision, define one predictive code/likelihood that counts observations
once. Do not combine MDL, BCV and spectral scores with an arbitrary vote or
weighted sum. A fully paid MDL code remains a valid code even when fitted on
the full data; BCV addresses predictive behavior, not the validity of that code.

## 5. A covariance extension specific to our ROME setting

This section is an algebraic proposal derived here, not a published detection
result in the papers reviewed.

Use two views of the same layer:

\[
G_I=WW^\top,\qquad G_C=WCW^\top.
\]

`G_C` is the second moment of the MLP output under the key second moment `C`
when that moment applies to the actual inputs. For any symmetric metric `M`,
the same weight update gives

\[
\Delta G_M=(WMb)a^\top+a(WMb)^\top+(b^\top Mb)aa^\top.
\]

For ROME, `Cb` is proportional to `k`, so the weighted cross direction is
proportional to `Wk`. Ordinary and weighted views therefore share one
physical update rather than two independently fitted rank-two anomalies.

A practical implementation can apply a candidate correction already produced
by the peel to both views. With `W_p=(I-uzᵀ)W*`,

\[
G_{M,p}=(I-uz^\top)G_{M,*}(I-zu^\top).
\]

This can assess a shared correction without storing a separate dense update
matrix. However, the candidate was optimized in the ordinary view; a joint
search could find candidates it misses. The exact code and joint background
model still need to be specified. Do not add independent log-likelihoods for
the two deterministic views without a valid joint/conditional model.

For the Householder example, `G_I` is unchanged but

\[
G_C'=WQCQ^\top W^\top
\]

generally differs from `G_C`. The extra metric can expose some changes hidden
from ordinary GRAM. It still has blind spots and does not establish ROME origin.
If `C` is isotropic, it adds no such information.

Operational constraints:

- The detector needs the covariance view for every eligible layer, or an
  explicitly declared coverage domain. Using the existence of a covariance
  file at the configured editing layer would leak the target.
- The present fleet covariance cache was prepared for ROME editing; the
  earlier audit verified one needed covariance per model. That does not
  establish an all-layer covariance cache. Do not quietly start expensive
  covariance recomputation.
- Covariance must match the architecture, neuron ordering and relevant
  activation distribution. Whether it comes from a trusted clean checkpoint
  or the suspect changes the input assumptions.
- `Cb` exists for every vector `b`. Calling it a recovered key is not evidence
  of a real edited subject without a plausible key-manifold or candidate test.
  KSTER obtains stronger information from an observed before/after difference.

Direct rectangular prediction across adjacent MLPs also requires care: their
intermediate neurons can be permuted independently. The residual-space Grams
avoid that particular nuisance. Two layers' column indices are not a natural
entrywise correspondence merely because their matrices have equal shapes.

## 6. A concrete binary decision with no fitted score cutoff

Let `X` be the fixed representation of all scanned layers. Let `B` index a
small, declared family of natural-background codes. Include smooth depth drift,
paid resets/stage changes, and a broad-transition model. An overly restrictive
null can mislabel normal architecture as an edit.

For no correction:

\[
\mathcal L_0=\min_B\{L(B)+L(X\mid B,H_0)\}.
\]

For one correction:

\[
\mathcal L_1=\min_{B,l,q,\theta}\{
L(B)+L(l)+L(q)+L(Q_q\theta)
+L(X\mid B,H_{1,l},Q_q\theta)\}.
\]

Both describe exactly the same source data. Layer selection, model selection,
finite precision and reconstruction residuals must all be paid. The null and
alternative can share a common one-bit hypothesis header. Then

\[
\boxed{\texttt{edited}=\mathbf 1[\mathcal L_1<\mathcal L_0]}
\]

with ties assigned to `false`. No per-model cutoff, CT match, special target
layer or collapse gate is involved. The output means the edit explanation won
under these codes. It does not have a calibrated probability interpretation.

This extends v1 rather than repudiating its decision. V1 already includes
paid resets, normalization-aware candidate planes, exact reconstruction costs
and all eligible layers. Its first priorities for investigation are the
literal FP64 correction cost, restrictive background family and candidate
search, not lowering a required bit-gain threshold.

### Bayesian equivalent and explicit assumptions

If we can justify probability models, an alternative is a global Bayes factor
for zero versus exactly one edit. With a uniform prior over eligible layers,

\[
p(X\mid H_1)=\frac1{|E|}\sum_{l\in E}
\int p(X\mid H_{1,l},\theta)\,\pi_l(d\theta).
\]

This averages over the unknown layer and parameters rather than using an
unpenalized maximum. Equal prior odds and equal error losses give `YES` when
`BF=p(X|H1)/p(X|H0)>1`. More generally the boundary is

\[
\log BF>
\log\frac{1-\pi_{\rm edit}}{\pi_{\rm edit}}
+\log\frac{C_{FP}}{C_{FN}}.
\]

Thus a Bayes decision avoids an empirically tuned score cutoff but still makes
prior and loss assumptions. A best two-part MDL code with a layer index and a
Bayesian mixture are related constructions, not identical calculations.

Do not substitute ordinary `BIC = n log(RSS/n) + k log(n)` with the number of
Gram entries as `n`. The entries are dependent and zero rank makes factor
parameters unidentified. Singular-model theory explains why ordinary BIC can
misprice such alternatives. [Drton and Plummer](https://arxiv.org/abs/1309.0911).

## 7. Methods I would not use as the final gate

**A bare Marchenko-Pastur/BBP edge:** useful under a supported noise model,
but a learned component beyond the edge need not be an edit, and finite-sample
fluctuations remain. Generalized/Fisher thresholds additionally require the
right sample-covariance setting.

**Gavish-Donoho hard thresholding:** principled for matrix-denoising error under
its noise assumptions. Its objective is reconstruction MSE, not binary edit
detection or a desired false-positive rate. It could generate candidates.
[Gavish and Donoho](https://arxiv.org/abs/1305.5870).

**Entrywise robust PCA:** a ROME outer product is usually dense over matrix
entries. If stacking vectorized Grams as columns over depth, the anomaly is
sparse over layers, making column-sparse models more relevant. A depth
Laplacian spreads one modification into the coupled three-layer stencil.
Robust decomposition still needs assumptions and model-selection accounting.

**Layer-permutation or conformal p-values without a valid null:** layer depth
is not exchangeable, and independently permuting trained weights destroys
natural structure as well as any edit. Neighbor exclusion alone does not
create exchangeability. Trusted calibration checkpoints would change what is
possible, but they change the input setting.

**Treating `2^mdl_delta_bits` as an e-value:** this is not automatically valid
for the composite null of natural unedited LLMs. An e-value needs null
expectation at most one; controlled rejection at `1/alpha` still requires a
risk level `alpha`. If valid per-layer e-values were available, their mean
could aggregate layers without requiring independence. Products of dependent
stencil scores do not have that justification.

**Opposite eigenvalue signs, a fixed kurtosis number, logit-lens concentration,
or a fixed peel-collapse percentage:** useful diagnostics or research ideas,
but no necessary-and-sufficient static ROME test follows from them.

## 8. Validation that would resolve the remaining uncertainty

1. Preserve the frozen v1 PBS experiment. Read per-layer background savings,
   correction costs and reconstruction costs before diagnosing misses. A miss
   can be a poor candidate, expensive representation or inadequate information.
2. Add paid precision selection first; evaluate it as v2 with the same binary
   rule. Compare clean-background codes separately so their effects are visible.
   Freeze code choices before the final test set is inspected.
3. Use a small architecture-diverse pilot for the spectrum/normalization and
   held-out-prediction hypotheses. Check whether the assumed noise bulk and
   stable prediction actually occur. Avoid launching a full new fleet before
   measuring their runtime.
4. Evaluate unedited checkpoints, native isolated low-rank components, natural
   depth transitions, fine-tunes, quantization, norm-matched random rank-one
   modifications and LoRA-style modifications. Distinguishing these matters
   if the claim is ROME-specific, rather than rank-one-anomaly detection.
5. For each model, 100 edits probe sensitivity on that model. If each starts
   from the same base, repeating its clean scan 100 times does not create 100
   independent negative checkpoints. Report distinct clean checkpoints and
   per-model false detections; do not manufacture a narrow false-positive
   interval from repeated identical controls.
6. Keep change-of-weights labels separate from edit efficacy. Report detection
   of all nonzero edits and, separately, behaviorally successful edits. A ROME
   attempt that changes no stored weight is a different case.
7. Test off-default editing layers with ground truth accessible only to the
   evaluator. Current v1 inherits GRAM's 10% interior trim and excludes
   endpoints. Its negative verdict covers that eligible domain. A claim about
   edits anywhere requires an explicit endpoint/background extension, not a
   silent assumption that the configured ROME layer is the only possibility.
8. Report binary TP/FN and clean FP/TN separately from localization accuracy,
   plus numerical failures, coverage, runtime and memory. Hold out model
   families or checkpoint origins when evaluating generalization. Related
   fine-tunes are not independent architecture tests.

The existing dense peel search remains a runtime concern. Reuse Gram
factorizations and spectra and only recode blocks affected by a candidate.
For full-spectrum functions, use available eigenvalues first; approximate
trace/log-determinant estimators would need their own numerical-error handling.
Cross-validation and all-layer covariance add real work. This note does not
increase PBS walltime requests or submit new jobs.

## 9. Numerical sanity checks performed for this research

Using NumPy float64, seed `20261024`, a random `8 x 13` matrix and a rank-one
update, the checks gave:

| Quantity | Result |
| --- | --- |
| Rank of raw Gram change | 2 |
| Rank of naive normalized Gram difference | 8 |
| Rank after the exact normalization-scale correction | 2 |
| Error of the repeated bulk generalized eigenvalue | about `1.2e-14` |
| Rank of a nonzero Householder weight update | 1 |
| Relative ordinary Gram change for that update | `3.84e-16` |
| Relative covariance-weighted Gram change for that update | `0.211` |

The covariance example used a random positive-definite, non-isotropic metric.
It demonstrates extra information in that example, not detection accuracy on
ROME or real checkpoints. These checks verify algebra and representation
limits; they do not validate a statistical noise model or the binary detector.

An additional two-dimensional check used `H=diag(2, 1/2)`. It admits
`H=TTᵀ` with `rank(T-I)=1` to about `3.1e-16` reconstruction error. More
generally, a positive-definite `H` whose `H-I` is rank-two indefinite has an
isotropic direction `vᵀ(H-I)v=0`. In a basis `(u,v)` of that plane,

\[
H_{\rm plane}=\begin{bmatrix}h_{11}&h_{12}\\h_{12}&1\end{bmatrix},
\quad
T_{\rm plane}=\begin{bmatrix}
\sqrt{h_{11}-h_{12}^2}&h_{12}\\0&1
\end{bmatrix}.
\]

Then `TTᵀ=H` and `T-I` has rank at most one; extend by the identity off the
plane. This shows how broad rank-one-compatible Gram geometry can be. Such
compatibility alone does not identify a particular editing algorithm.

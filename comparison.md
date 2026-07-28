# Current minimal ROME detector versus the previous weighted-spectrum method

## Short answer

The current detector is a **consolidation**, not a completely different
mathematical localizer.

The central layer-localization chain described in `explain.md` was retained:

```text
editable weight
  -> normalized hidden Gram
  -> two-neighbor depth residual
  -> leading two-dimensional SVD subspace
  -> 2x2 neighbor-support whitening
  -> Frobenius score
  -> argmax over eligible layers
```

What was removed was the larger experimental shell around that score:
additional footprint statistics, morphology products, blind outlier rules,
alternative M0-M2 scoring candidates, uncalibrated binary variants, graph
rendering, and the cluster-ablation runtime.

The result is substantially easier to understand, test, and operate. It also
does less work and emits much smaller artifacts than the former full
ROME-presence or ablation pipelines. It does **not**, however, eliminate the
dominant hidden-Gram and top-two SVD calculations. The asymptotic cost of the
checkpoint-only localizer is therefore essentially unchanged.

## Side-by-side summary

| Property | Previous method described by `explain.md` and its full ROME-presence pipeline | Current minimal detector |
|---|---|---|
| Public schema | Several localizer and presence artifacts | One `rome-detector-minimal-v1` artifact |
| Checkpoint-only output | Always returns the strongest candidate layer | Same behavior: always returns the strongest eligible layer |
| Checkpoint-only binary decision | Blind peak and blind footprint rules were proposed | Removed because they were not calibrated against independent negatives |
| Clean-reference binary decision | Exact single-matrix, rank-one weight-delta rule | B0: numerically significant, rank-at-most-two hidden-Gram delta |
| Localization score | Normalized Gram, neighbor residual, top-two SVD, 2x2 whitening, Frobenius norm | The same winning M3 score |
| Per-layer footprint fields | Localizer score plus rank-two energy, bilateral coherence, and bilateral balance when presence analysis was enabled | Only `relative_subspace_frobenius` |
| Alternative scores | M0, M1, M2, and M3 were evaluated during ablation | Only M3 remains |
| Trimming | Fixed first/last layer counts, historically `5/5` | One architecture-neutral fraction, `10%`, while always excluding endpoints needed by the neighbor calculation |
| Model-specific routes | None in weighted spectrum | None |
| Learned thresholds | None | None |
| Blind statistical cutoff | Median/MAD normalization and `sqrt(2 log n)` universal bound | Removed |
| Visual output | Complete detector explainer graphs and exports were available | No detector graph pipeline |
| Attribution wording | “ROME-like” localized rank-one edit | Deliberately narrower: “ROME-compatible low-rank edit” |
| Known specificity | Not established | Still not established |

## What stayed

### 1. Architecture-neutral hidden Gram

For an editable projection \(W_l\), both methods orient the Gram matrix onto
the smaller shared hidden axis:

\[
G_l =
\begin{cases}
W_lW_l^\top, & \text{if rows} \leq \text{columns},\\
W_l^\top W_l, & \text{otherwise}.
\end{cases}
\]

The checkpoint-only localizer normalizes it by the squared Frobenius norm:

\[
C_l = \frac{G_l}{\lVert W_l\rVert_F^2}.
\]

This keeps the useful properties of the previous method:

- no model-name or model-family routing;
- tolerance of transposed weight-storage conventions;
- invariance to positive global rescaling of a projection;
- a common hidden-space object that can be compared across neighboring
  transformer layers.

### 2. Local depth residual

Both methods estimate the expected profile of layer \(l\) from its immediate
neighbors:

\[
N_l = \frac{C_{l-1}+C_{l+1}}{2},
\qquad
R_l = C_l-N_l.
\]

This is the local-curvature assumption explained at length in `explain.md`.
It is also still the main heuristic: a normal model is assumed to change
reasonably smoothly through depth, while a localized edit creates an unusual
residual.

### 3. Top-two SVD subspace

Both methods take the two leading singular directions \(U_l\) of \(R_l\).
The use of two directions comes from the fact that a rank-one weight update
creates a rank-at-most-two change in the **unnormalized** hidden Gram:

\[
(W+uv^\top)(W+uv^\top)^\top-WW^\top
\]

is a sum of at most two independent outer-product directions.

The exact rank-two statement does not apply unchanged after independently
normalizing the two Gram matrices. That limitation from `explain.md` still
applies.

### 4. Two-dimensional support whitening

Both methods project the residual and neighbor support into the same
two-dimensional subspace:

\[
A_l=U_l^\top R_lU_l,
\qquad
B_l=U_l^\top N_lU_l.
\]

They then whiten the residual by the neighbor support:

\[
E_l=B_l^{-1/2}A_lB_l^{-1/2}.
\]

Only a \(2\times2\) eigendecomposition is needed for this whitening step.

### 5. Frobenius localization score

The retained M3 score is:

\[
s_l=\lVert E_l\rVert_F,
\qquad
\hat l=\operatorname*{arg\,max}_{l\in\mathcal L}s_l.
\]

Thus the Gram, SVD, whitening, and Frobenius pieces the project wanted to keep
are still the entire mathematical core of checkpoint-only localization.

## What was thrown out

### Rank-two energy as a localizer multiplier

The old full footprint captured the fraction of residual energy explained by
the first two singular values. Experimental variants multiplied or otherwise
combined this with simpler scores.

It was removed from production localization because M3, which did not need
that multiplier, was the only candidate that passed the predeclared
development selection rule.

This does **not** mean rank-two structure disappeared everywhere. The
clean-reference B0 decision still checks the energy beyond rank two in the
unnormalized Gram delta, where the ROME algebra applies directly.

### Bilateral coherence and bilateral balance

The former full presence capture measured:

- whether the center layer differed from both neighbors in a coherent way;
- whether the left and right jumps had similar energy.

These quantities attempted to encode the three-layer
\((-1/2,1,-1/2)\) footprint. They were removed because they did not contribute
to the selected M3 localization score and had no validated binary-specificity
benefit.

### Morphology product

The old blind-footprint rule multiplied:

```text
whitened spectral score
  x rank-two energy
  x bilateral coherence
  x bilateral balance
```

This conjunction looked mathematically purposeful, but it introduced more
moving pieces without independent clean and hard-negative evidence showing
that the product distinguished ROME from other changes. It is no longer
calculated.

### `log1p`, median/MAD, and the universal extreme cutoff

The old blind rules transformed the layer series with `log1p`, normalized the
peak using a robust median and MAD, and compared it with:

\[
\sqrt{2\log n}.
\]

This was a generic Gaussian-noise extreme bound, not an empirically calibrated
ROME threshold. Both blind-peak and blind-footprint decisions were therefore
removed.

This is an honesty improvement as well as a simplification: the current
detector no longer turns an unvalidated suspect-only outlier score into a
binary ROME verdict.

### M0, M1, and M2

The N=50 development ablation evaluated four localizers:

| Candidate | Exact localization over 450 requested cases |
|---|---:|
| M0 | 174/450 = 38.7% |
| M1 | 222/450 = 49.3% |
| M2 | 314/450 = 69.8% |
| M3 | 386/450 = 85.8% |

M0-M2 were rejected rather than retained as optional modes. Production no
longer computes, stores, compares, or configures them.

### B1 and B2

B1 was never calibrated because the evaluation lacked independent negative
checkpoints. B2 was only an experimental control. Neither is exposed by the
minimal detector.

### Ablation and cluster machinery

The following completed experimental infrastructure was removed from the
production branch:

- M0-M3 evaluator and comparison code;
- B0-B2 experimental evaluator paths;
- recapture manifests and cluster jobs used only by the ablation;
- large raw ablation artifacts from the active tree;
- bootstrap/non-inferiority selection runtime;
- renderer and graph-explainer paths tied to the removed decisions.

The compact N=50 golden fixture and reports were retained so that the
simplification remains auditable without keeping the experiment as production
code.

### The former clean-delta rule

The previous clean-reference decision required exactly one canonical MLP
output matrix to change and checked whether its direct weight delta was rank
one within roundoff.

The current B0 rule instead forms the unnormalized hidden-Gram change:

\[
\Delta G_l =
G_l^{\text{suspect}}-G_l^{\text{clean}},
\]

then checks:

1. that its magnitude is above a dtype-, dimension-, and scale-derived
   numerical bound; and
2. that energy beyond rank two remains within its numerical bound.

This is closer to the Gram-space argument used by the localizer, but it is
also intentionally attributed only as a
`generic_rank_at_most_two_gram_change`. Another rank-one editing method can
produce the same footprint.

## What changed rather than disappeared

### Fixed trim became fractional trim

The old configuration excluded fixed numbers of early and late layers,
historically five on each side. That does not scale naturally between shallow
and deep models.

The current rule uses:

\[
\left\lfloor0.10L\right\rfloor
\]

on each side, with the endpoints always excluded because their two-neighbor
reference does not exist.

This is still a design constant, but it is one architecture-independent
fraction rather than a set of model-specific layer counts. It also preserves
useful interior layers on smaller architectures better than a fixed `5/5`
rule.

### Binary detection now has an explicit threat model

The current API separates two claims:

1. **Suspect checkpoint only:** localize the strongest candidate layer. This
   is not a yes/no detector.
2. **Clean and suspect checkpoints:** return a B0 boolean for a numerically
   significant ROME-compatible low-rank Gram change.

If no clean reference is provided, the boolean is reported as unavailable
rather than guessed from an uncalibrated blind cutoff.

## Is the current method faster?

### Compared with the old checkpoint-only localizer

Only slightly, and possibly not measurably without a controlled benchmark.

The old `explain.md` localizer and current M3 execute the same expensive
operations:

- construct hidden-space Gram matrices;
- form neighbor residuals;
- run a randomized top-two SVD for each evaluated residual.

The current implementation removes a redundant eigenvalue-norm calculation
and directly takes the Frobenius norm of the \(2\times2\) whitened matrix, but
that is tiny compared with constructing a large Gram matrix or running the
SVD.

Therefore no large speedup should be claimed for **M3 localization alone**.

### Compared with the old full ROME-presence pipeline

The current method does less work:

- no full-residual norm for `rank2_energy`;
- no two bilateral jump energies;
- no coherence/balance calculation;
- no morphology product;
- no blind-peak and blind-footprint analyses;
- no multiple decision artifacts;
- no detector explainer graph rendering.

These removed calculations are mostly \(O(h^2)\) elementwise work per layer,
plus analysis and I/O. They are cheaper than the Gram multiplication and SVD,
so the likely detector-compute reduction is real but not an order-of-magnitude
algorithmic improvement.

Rendering and artifact I/O can show a much larger wall-clock reduction when
the old explainer graphs were enabled.

### Compared with the M0-M3/B0 ablation run

The historical experimental capture recorded:

- 389.3 aggregate detector seconds across 450 cases;
- approximately 551 MiB maximum estimated peak detector working memory.

That run computed and evaluated M0-M3 plus B0. The minimal production path
does not compute M0-M2 or run the associated evaluator, so it must perform
less work than that experimental path.

There is no matched old-versus-current timing run in the preserved evidence.
Consequently, an exact speedup such as “2x faster” is **not supported**.

### When B0 is enabled

B0 requires both clean and suspect weights, creates their unnormalized hidden
Grams, and estimates the rank-two tail. This is additional work that the
checkpoint-only M3 localizer does not perform.

The fairest runtime comparison is therefore:

- M3 versus the former weighted-spectrum localizer; and
- M3+B0 versus the former full ROME-presence pipeline.

Comparing M3+B0 with the old localizer alone would mix different outputs and
could make the current method appear slower simply because it now provides a
clean-reference binary result.

## Memory and artifact size

The dominant working object is still one hidden-space Gram matrix, so peak
detector memory remains \(O(h^2)\). The rolling three-layer cache prevents
memory from growing as \(O(Lh^2)\), but the simplification does not change the
quadratic dependence on hidden width.

The persistent profile is clearly smaller:

- old full footprint: four scalar profile fields per layer;
- current profile: one scalar field per layer.

That is a 75% reduction in the per-layer scalar profile portion. It is not
necessarily a 75% reduction in the entire run directory because metadata,
edit artifacts, and other captures may dominate.

Removing separate blind-analysis JSON, ablation outputs, CSV exports, and
graphs reduces total artifact count and disk I/O further. No controlled
whole-run byte comparison has been recorded, so an exact storage multiplier
should not be claimed.

## Accuracy and evidence comparison

The old report associated with the long explanation reported 38/40 exact
localizations. The original frozen artifacts needed to reproduce that result
were later unavailable locally.

The current retained M3 evidence is larger but exposed development evidence:

- 386/450 exact localizations over all requested N=50 cases;
- 375/435 over successful ROME edits;
- 434/435 B0 positives over successful edits;
- Falcon localization only 9/50.

The later 13-model execution smoke produced:

- 20/25 exact localizations over successful edits;
- 25/25 B0 positives;
- Falcon 0/2 and OLMo 0/2 exact localization.

These datasets and protocols are different, so 38/40 and 386/450 should not
be treated as a direct before/after accuracy comparison. The simplification
preserved the selected M3 formula; it did not demonstrate that the detector
became more accurate.

## What the current detector can and cannot claim

It can claim:

- architecture-neutral code with no model-family thresholds;
- a single general checkpoint-only layer-localization formula;
- high positive sensitivity for the clean-reference B0 rule on the exposed
  ROME development cases;
- a substantially smaller and clearer production surface.

It cannot yet claim:

- reliable suspect-only binary ROME detection;
- specificity against clean models, fine-tunes, quantization, merges, or
  other editors;
- unique attribution to the ROME program;
- robust localization for every architecture;
- a measured end-to-end runtime or memory speedup factor.

## Bottom line

The mathematical simplification succeeded mainly by deleting unsupported
branches around the winning score. It kept the parts with the clearest ROME
motivation—Gram geometry, top-two SVD, 2x2 whitening, and Frobenius
localization—and removed morphology engineering and uncalibrated statistical
decisions.

This makes the detector less overbuilt and easier to defend. It is faster and
smaller than the former full presence/ablation workflow, but the expensive
core is intentionally unchanged. A major further speedup would require a new
implementation strategy for the Gram/SVD stage, such as implicit
matrix-vector products, randomized sketches, or streamed low-rank
approximations. That would be a new optimization project and would require a
parity study against the frozen M3 evidence.

## Evidence used for this comparison

- Previous mathematical explanation: `explain.md`
- Previous production implementation and documentation: Git commit
  `3c220c8`
- N=50 ablation and selection evidence: `rome-math-n50-cluster-report.md`,
  commit `693a949`
- Minimal consolidation: commits `420bcd8` and `c0f4222`
- Current detector documentation: `docs/detector.md`
- Current remote smoke: `rome-minimal-remote-smoke-report.md`
- Current branch head: `1044795`

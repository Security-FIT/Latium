# ROME detector mathematical simplification

## Question

Can the one-checkpoint ROME layer localizer remove the 2×2 whitening step
without losing the cross-architecture localization accuracy of M3?

The experiment also asks whether a scale-free spike in any surviving layer
score can support a binary ROME yes/no verdict. The detector input remains one
suspect checkpoint only. Clean checkpoints and synthetic edits are evaluation
labels, never detector inputs.

## Non-negotiable constraints

- no clean or paired reference checkpoint;
- no prompt, fact, target, case ID, causal trace, covariance, or edit metadata;
- no model name, architecture branch, family-specific threshold, or
  per-model correction;
- no fitted detector hyperparameter; the existing generic 10% boundary trim
  is the only retained constant;
- rank two is derived from ROME's rank-one weight update, whose hidden-Gram
  change has rank at most two;
- exact ties select the lower layer.

## Candidate ladder

For projection weight \(W_l\), orient the Gram matrix onto the smaller axis and
normalize it:

$$
C_l =
\frac{\operatorname{hidden\_gram}(W_l)}
     {\lVert W_l\rVert_F^2}.
$$

Use the immediate layer neighbors:

$$
N_l = \frac{C_{l-1}+C_{l+1}}{2},
\qquad
R_l = C_l-N_l.
$$

The candidates are ordered from least to most mathematical machinery.

| Name | Score | Added machinery |
|---|---|---|
| Gram Frobenius | \(\lVert R_l\rVert_F\) | Gram + Frobenius only |
| Relative Gram Frobenius | \(\lVert R_l\rVert_F/\lVert N_l\rVert_F\) | one scalar division; no SVD |
| Top-2 Frobenius | \(\sqrt{\sigma_1^2+\sigma_2^2}\) | top-2 SVD |
| Scalar-relative top-2 | \(\sqrt{\sigma_1^2+\sigma_2^2}/\lVert U_l^\top N_lU_l\rVert_F\) | one projected support norm |
| Diagonal-relative top-2 | \(\sqrt{\sum_{i=1}^2(\sigma_i/(u_i^\top N_lu_i))^2}\) | separate scalar support per direction |
| M3 control | \(\lVert B_l^{-1/2}A_lB_l^{-1/2}\rVert_F\) | full 2×2 whitening |

Here \(U_l=[u_1,u_2]\), \(A_l=U_l^\top R_lU_l\), and
\(B_l=U_l^\top N_lU_l\).

The diagonal-relative candidate is not full whitening. It never
eigendecomposes \(B_l\), rotates by its eigenvectors, or constructs a matrix
inverse square root. Its possible weakness is that it ignores off-diagonal
support coupling. M3 is calculated in the same pass as a paired control,
reusing the same Gram matrices and rank-two basis.

## Frozen N=50 evidence before the new recapture

The previous nine-model development run contains 435 successful ROME edits.
It already rejects the first, third, and scalar-relative candidates:

| Model | Gram Frobenius | Top-2 Frobenius | Scalar-relative | M3 |
|---|---:|---:|---:|---:|
| DeepSeek 7B Base | 49/49 | 49/49 | 49/49 | 49/49 |
| Falcon 7B | 0/49 | 0/49 | 0/49 | 9/49 |
| GPT-J 6B | 0/49 | 0/49 | 49/49 | 48/49 |
| GPT-2 XL | 28/48 | 2/48 | 1/48 | 47/48 |
| Granite 4 Micro | 45/45 | 45/45 | 45/45 | 45/45 |
| Llama-2 7B | 37/49 | 46/49 | 49/49 | 46/49 |
| Mistral v0.1 | 0/50 | 34/50 | 48/50 | 46/50 |
| Mistral v0.3 | 7/46 | 36/46 | 14/46 | 35/46 |
| OPT 6.7B | 0/50 | 0/50 | 49/50 | 50/50 |
| **Total** | **166/435** | **212/435** | **304/435** | **375/435** |

On successful edits, equal-model macro exact localization was 38.8%, 49.4%,
69.6%, and 86.2%, respectively. Scalar normalization is useful, but one
scalar for the whole top-2 subspace does not preserve M3 accuracy.

An additional post-hoc check tested a simple bilateral Gram-curvature ratio
from the frozen profiles. It localized only about 24% of successful edits.
Multiplying that ratio into Frobenius, top-2, or scalar-relative scores did not
improve the corresponding candidate. It is rejected and is not part of the
new capture.

The frozen artifacts do not retain the full neighboring Gram or the diagonal
entries of \(U_l^\top N_lU_l\). Therefore relative Gram Frobenius and
diagonal-relative top-2 require a focused recapture; they cannot be honestly
reconstructed from the stored scalars.

## Binary evaluation

Each candidate produces three parameter-free checkpoint summaries:

- robust peak above the profile median, scaled by median absolute deviation;
- peak divided by the second-highest eligible score;
- peak divided by its immediate eligible layer neighbors.

The local statistic has no fitted radius. It uses only the predecessor and
successor in the eligible layer sequence.

Thresholds are development evaluation tools, not detector constants. They are
fitted on all but one exposed model family and tested on the held-out family.
Any yes/no candidate must also reject:

- magnitude-matched random rank-one projection edits;
- magnitude-matched random rank-two projection edits;
- magnitude-matched rank-one edits in two layers;
- a matched edit to a matrix the detector does not inspect;
- standalone clean checkpoints.

The matched-edit magnitude source sizes evaluation specimens only. It is not
provided to the detector.

An arbitrary rank-one writer can produce the same algebraic Gram footprint as
ROME. Therefore a spike rule may detect a localized low-rank anomaly but still
fail as a ROME provenance detector. A boolean is allowed only if the empirical
hard-negative result contradicts that known failure mode with high
specificity; otherwise production must remain localization-only.

## Execution safety

The focused remote runner executes only ROME generation and the opt-in
simple-Gram capture. Normal detector analyses and graph rendering are
disabled. Exact existing current-main 100,000-sample covariances are required;
none are recomputed.

Both runners refuse to start if either the GPU has a compute process or any
relevant Python/Latium driver is alive. This prevents starting during the
brief GPU-idle gap between another worker's pipeline stages.

## Focused recapture result

The cluster run captured every candidate and the exact M3 control together.
ROME generation and tensor capture happened once per case; all localization,
spike, threshold, and report calculations below read the stored JSON profiles.

Across 260 requested cases on 13 model families, 240 ROME edits met the edit
success criterion:

| Candidate | Exact | Micro rate | Equal-model macro | Within one |
|---|---:|---:|---:|---:|
| Gram Frobenius | 92/240 | 38.3% | 41.9% | 112/240 |
| Relative Gram Frobenius | 78/240 | 32.5% | 36.2% | 98/240 |
| Top-2 Frobenius | 131/240 | 54.6% | 56.5% | 163/240 |
| Scalar-relative top-2 | 178/240 | 74.2% | 75.1% | 186/240 |
| **Diagonal-relative top-2** | **196/240** | **81.7%** | **82.7%** | **201/240** |
| M3 control | 198/240 | 82.5% | 83.5% | 203/240 |

Diagonal-relative and M3 tied in exact successes on 12/13 models. The only
difference was Falcon, where diagonal-relative localized 3/19 and M3 localized
5/19. OLMo remained a shared failure at 0/20. Across all successful edits,
the two methods selected the same layer in 204/240 cases. Of the 36 selection
disagreements, M3 alone was correct twice and both methods were wrong 34 times.

The equal-model macro loss is 0.81 percentage points, inside the predeclared
2.5-point simplification margin. This supports deleting the 2×2 support
eigendecomposition, eigenvector rotation, inverse square root, and two-sided
whitening. It does not solve the known Falcon or OLMo localization weaknesses.

The frozen localization aggregate is:

```text
analysis_out/rome-simple-gram-n20-v1/evaluation-all13-clean-only.json
SHA-256 f119e82ccba128be21970cf4f730243adc5ddcc9078362a2abd2d5f6f4aefe88
```

## Matched-hard-negative result

The five-family development subset contains 94 successful ROME edits, five
standalone clean checkpoints, and 200 magnitude-matched hard negatives. Each
family contributes ten specimens from each of four categories. Leave-one-
family-out calibration uses one global high-is-positive cutoff per training
fold; there is no model-specific rule.

For the retained diagonal-relative profile:

| Spike statistic | Sensitivity | Specificity | Macro balanced accuracy | Rank-one specificity | Rank-two specificity |
|---|---:|---:|---:|---:|---:|
| Robust peak | 61.7% | 37.6% | 50.4% | 36.0% | 38.0% |
| Global prominence | 78.7% | 34.6% | 56.7% | 18.0% | 24.0% |
| Local prominence | 45.7% | 75.6% | 61.8% | 48.0% | 54.0% |

The best macro result still misses both sides of a useful binary detector.
Local prominence rejected every standalone clean checkpoint and non-target
matrix edit, but it detected fewer than half of the ROME positives and
accepted more than half of the matched random rank-one edits. The M3 control
was not a remedy: its best macro result was 63.3%, with 56.4% sensitivity,
68.3% specificity, and 42.0% matched-rank-one specificity.

The frozen binary aggregate is:

```text
analysis_out/rome-simple-gram-hard-negatives-v1/evaluation-with-hard-negatives.json
SHA-256 eeb7c5608f0f8971eb47b6229a0e8d140af86dd039c2220fd01eb9125cbf326e
```

Bundle hashes:

| Model | SHA-256 |
|---|---|
| GPT-2 XL | `a0b340f2a0dbdf433b78484d93e209ebb3eda3b69bcdbaa399dbd20124734790` |
| Mistral v0.1 | `e94c0b6851ad84600104819bc5af0570a37db4b7e6af2eab12b5ee0288e59729` |
| Falcon 7B | `5d34b4a9755ccfdd0b23672de53c18cc22fd575c385ca3f65db2953cc0e05cce` |
| OLMo 3 7B | `aabddc12a755c9ab1bf3b8af85a73f06cae9c7eb44a6a49bca2a519539c76d45` |
| Granite 4 Micro | `1b6858b4f613219872fc48f6521e8df95e0681180335662e01ee27780cc04d20` |

## Final decision

Production advances to `rome-detector-minimal-v3` and stores one
`diagonal_relative` score per eligible layer. It needs one suspect checkpoint
only. It does not consume a clean/reference model and does not expose a binary
ROME verdict.

The following M3 operations are deleted from production:

- 2×2 support eigendecomposition;
- rotation by support eigenvectors;
- inverse matrix square root;
- two-sided support whitening.

The failed Gram/Frobenius candidates, spike thresholds, hard-negative
generator, offline evaluator, opt-in capture, and cluster runners are also
removed after recording this evidence. They remain recoverable in Git history
but are not shipped as production or dormant ablation paths.

# Minimal evidence-supported ROME localizer

The active ROME component is a one-checkpoint layer localizer. It is not a
binary provenance detector.

It consumes only the ordered editable projection weights from one checkpoint.
It does not consume clean weights, prompts, case metadata, causal traces,
second moments, model names, or family-specific thresholds. The public schema
is `rome-detector-minimal-v3`.

## Layer score

For each editable projection, orient the Gram onto the smaller shared hidden
axis and remove scale:

```text
C_l = hidden_gram(W_l) / ||W_l||_F^2
```

Subtract the immediate-neighbor reference:

```text
N_l = (C_{l-1} + C_{l+1}) / 2
R_l = C_l - N_l
```

Let `u_1, u_2` be the two leading left singular directions of `R_l`, with
singular values `sigma_1, sigma_2`. Measure the neighboring support separately
in each direction:

```text
b_i = u_i^T N_l u_i
```

Divide each singular value by its own scalar support and take the vector norm:

```text
s_l = sqrt((sigma_1 / b_1)^2 + (sigma_2 / b_2)^2)
```

The localizer returns:

```text
selected_layer = argmax_l s_l
```

Eligibility uses one generic 10% fractional trim and excludes endpoints
without two neighbors. A scale-aware numerical tolerance protects vanishing
support. Exact ties select the lower layer. Storage transpose, positive weight
scaling, and hidden-space orthogonal basis changes preserve the score.

The artifact stores only `diagonal_relative`. The former 2×2 support
eigendecomposition, inverse square root, and whitening are gone. Rank-two
energy, bilateral features, morphology products, blind thresholds, and
clean-reference fields are also not part of the active contract.

## Simplification evidence

The candidates were calculated together in one capture pass and compared
offline from the saved profiles. Across 13 model families and 240 successful
ROME edits, the retained diagonal-relative score localized 196/240 exactly
(81.7%; 82.7% equal-model macro). The former M3 score localized 198/240
(82.5%; 83.5% macro). The 0.81-point macro loss is inside the predeclared
2.5-point simplification margin. The two additional M3 successes were both
Falcon cases; the methods tied on every other model.

The change removes concepts and code more than runtime. Hidden-Gram
construction and top-two SVD remain the dominant operations; eliminating
2×2 whitening is only a small constant-factor saving.

## Why there is no boolean ROME verdict

A final checkpoint does not uniquely encode which program created it. ROME
uses a rank-one weight update, but another writer can create the same
rank-one geometry—or the identical final tensor.

The single-checkpoint development experiment tested 94 successful ROME edits,
five standalone clean checkpoints, and 200 magnitude-matched hard negatives
across five exposed families. The best transparent two-statistic rule achieved
70.2% sensitivity, 64.4% specificity, and only 18% specificity on matched
random rank-one edits. A two-feature logistic baseline was worse. OLMo was
non-separable at the selected M3 peak.

The diagonal-relative simplification was checked again on the same five
families with newly captured profiles. Its best spike rule reached 45.7%
sensitivity, 75.6% specificity, and 48.0% specificity on matched random
rank-one edits. A higher-sensitivity variant reached 78.7% sensitivity but
only 34.6% specificity and 18.0% matched-rank-one specificity. The simpler
score therefore does not change the provenance conclusion.

Therefore the repository does not expose `is_rome_like`, B0, B1, B2, or a
default blind threshold. The evidence supports only a broader localized
low-rank anomaly interpretation, not ROME provenance.

See `rome-single-checkpoint-impossibility-report.md` for the corpus, confidence
intervals, and original failure cases. See
`rome-simple-gram-simplification-report.md` for the direct v3 ablation and its
artifact hashes.

## API

```python
from src.structural.detectors.rome_presence_resident import RomeDetector

result = RomeDetector().detect_one_checkpoint(suspect_proj)
selected_layer = result["localization"]["selected_layer"]
scores = result["localization"]["layer_scores"]
```

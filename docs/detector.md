# Minimal evidence-supported ROME localizer

The active ROME component is a one-checkpoint layer localizer. It is not a
binary provenance detector.

It consumes only the ordered editable projection weights from one checkpoint.
It does not consume clean weights, prompts, case metadata, causal traces,
second moments, model names, or family-specific thresholds. The public schema
is `rome-detector-minimal-v2`.

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

Let `U_l` contain the two leading singular directions of `R_l`, then project
the residual and local support:

```text
A_l = U_l^T R_l U_l
B_l = U_l^T N_l U_l
```

Whiten this 2×2 subspace and take its Frobenius norm:

```text
E_l = B_l^(-1/2) A_l B_l^(-1/2)
s_l = ||E_l||_F
```

The localizer returns:

```text
selected_layer = argmax_l s_l
```

Eligibility uses one generic 10% fractional trim and excludes endpoints
without two neighbors. Exact ties select the lower layer. Storage transpose,
positive weight scaling, and hidden-space orthogonal basis changes preserve
the score.

The artifact stores only `relative_subspace_frobenius`. Rank-two energy,
bilateral features, morphology products, blind thresholds, and clean-reference
fields are not part of the active contract.

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

Therefore the repository does not expose `is_rome_like`, B0, B1, B2, or a
default blind threshold. The evidence supports only a broader localized
low-rank anomaly interpretation, not ROME provenance.

See `rome-single-checkpoint-impossibility-report.md` for the corpus, confidence
intervals, failure cases, and evidence hashes.

## API

```python
from src.structural.detectors.rome_presence_resident import RomeDetector

result = RomeDetector().detect_one_checkpoint(suspect_proj)
selected_layer = result["localization"]["selected_layer"]
scores = result["localization"]["layer_scores"]
```

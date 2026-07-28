# Minimal evidence-supported ROME detector

The active ROME detector is a checkpoint-only layer localizer. A calibrated
single-checkpoint boolean is under evaluation and is not yet a production
claim.

It does not route on model names or families and has no per-model thresholds.
It consumes only the ordered editable projection weights from one checkpoint.
The public artifact schema is `rome-detector-minimal-v2`.

Run it through the ROME-presence profile:

```bash
python -m src structural run \
  structural.capture.profile=rome-presence \
  structural.analysis.preset=rome-presence
```

The broader `detection` preset still runs this localizer together with the
independent spectral detector. Legacy paper analyses remain available through
the explicit `paper` profile and preset.

## Layer localization

For editable projection `W_l`, orient its Gram into the smaller shared hidden
space and remove global scale:

```text
G_l = hidden_gram(W_l) / ||W_l||_F^2
```

For each eligible interior layer, subtract the two-neighbor reference:

```text
N_l = (G_{l-1} + G_{l+1}) / 2
R_l = G_l - N_l
```

Let `U_l` contain the two leading singular directions of `R_l`, and project
the residual and its neighbor support:

```text
A_l = U_l^T R_l U_l
B_l = U_l^T N_l U_l
```

Whiten only this 2×2 subspace and take its Frobenius norm:

```text
E_l = B_l^(-1/2) A_l B_l^(-1/2)
s_l = ||E_l||_F
```

The selected layer is:

```text
selected_layer = argmax_l s_l
```

Eligibility uses a fixed 10% fractional trim and excludes endpoints requiring
missing neighbors. Exact score ties select the lower layer. Storage transpose,
positive weight scaling, and hidden-space orthogonal basis changes preserve
the score.

The capture calculates and stores only `relative_subspace_frobenius`. Rank-two
energy, bilateral coherence/balance, morphology products, logarithmic
transforms, and blind threshold decisions are not part of the production
localizer.

## Frozen evidence and limitations

The development evaluation preserved at commit `693a949` contains 450
requested edits across nine exposed models:

- localization: 386/450 overall and 375/435 successful edits;
- Falcon localization: 9/50, a known unresolved generalization failure;
- all non-Falcon models combined: 377/400 localization.

The compact regression fixture is
`tests/fixtures/rome_detector_n50_golden.json`; the detailed report is
`rome-math-n50-cluster-report.md`.

Historical clean-reference B0 results remain in the archived N=50 report, but
B0 is not part of the active schema or API. The corpus does not estimate
single-checkpoint specificity. It lacks independent clean
checkpoints and hard negatives such as non-ROME low-rank edits, other editing
methods, ordinary fine-tunes, quantized checkpoints, and merged checkpoints.
No single-checkpoint binary claim is currently supported.

## Direct API

```python
from src.structural.detectors.rome_presence_resident import RomeDetector

result = RomeDetector().detect_one_checkpoint(suspect_proj)
selected_layer = result["localization"]["selected_layer"]
```

# Minimal ROME localizer versus the previous detector

The current implementation retains the smallest candidate that matched M3
within the declared localization margin:

```text
one checkpoint
  -> normalized hidden Gram
  -> two-neighbor residual
  -> top-two SVD subspace
  -> two scalar directional supports
  -> relative vector/Frobenius score
  -> layer argmax
```

## What remains

| Component | Status |
|---|---|
| Generic editable-weight discovery and orientation | kept |
| Hidden-Gram scale normalization | kept |
| Two-neighbor depth residual | kept |
| Deterministic top-two SVD | kept |
| Per-direction scalar normalization | kept |
| Frobenius layer score and deterministic argmax | kept |
| Generic fractional trimming and numerical safeguards | kept |

The focused compute-once recapture produced all candidate profiles in the same
pass, then evaluated them offline. Across 13 models and 240 successful ROME
edits, the retained diagonal-relative score localized 196/240 exactly (81.7%;
82.7% equal-model macro). M3 localized 198/240 (82.5%; 83.5% macro). The
observed macro loss was 0.81 percentage points, inside the predeclared
2.5-point simplification margin. Only two Falcon cases differed in correctness;
the methods tied on all other models.

This is primarily a mathematical and maintenance simplification, not a large
wall-clock optimization. Gram construction and the top-two SVD still dominate
runtime. Removing a 2×2 eigendecomposition and a few 2×2 matrix products saves
some per-layer work, but the expected end-to-end speedup is small.

## What was removed

| Component | Reason |
|---|---|
| M0, M1, M2 | materially worse localization than M3 |
| 2×2 support eigendecomposition | diagonal support matched M3 within the simplification margin |
| support eigenvector rotation | no longer needed without full whitening |
| inverse matrix square root | no longer needed without full whitening |
| two-sided 2×2 whitening | replaced by two scalar divisions |
| rank-two energy multiplier | not used by the winning localizer |
| bilateral coherence and balance | no validated binary benefit |
| morphology product and `log1p` | no verdict gain or specificity evidence |
| B0 clean-reference decision | violates the one-checkpoint threat model |
| B1/B2 blind decisions | uncalibrated and failed negative evaluation |
| model/family thresholds | scientifically forbidden |
| graph and ablation runtime | experiment completed |

## Binary conclusion

The new negative corpus contains 94 successful ROME edits, five standalone
clean checkpoints, and 200 hard negatives. The best transparent
single-checkpoint rule achieved:

```text
sensitivity                 70.2%
overall specificity         64.4%
matched-rank-one specificity 18.0%
equal-family macro BA       67.6%
worst-family BA             50.0%
```

A deterministic two-feature logistic baseline was worse.

The new diagonal-relative profile was also tested directly on 94 successful
ROME edits, five clean checkpoints, and 200 newly captured matched negatives.
Its best macro spike rule achieved:

```text
sensitivity                  45.7%
overall specificity          75.6%
matched-rank-one specificity 48.0%
equal-family macro BA        61.8%
```

A higher-sensitivity diagonal rule reached 78.7% sensitivity but collapsed to
34.6% specificity and 18.0% matched-rank-one specificity. It also fails.

This is expected: ROME's rank-one footprint is shared by generic rank-one
writers, and two different programs can produce the same final checkpoint.
The repository therefore keeps diagonal-relative layer localization but makes
no binary ROME provenance claim. Full provenance evidence is in
`rome-single-checkpoint-impossibility-report.md`; the direct mathematical
simplification evidence is in `rome-simple-gram-simplification-report.md`.

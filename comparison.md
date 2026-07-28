# Minimal ROME localizer versus the previous detector

The current implementation retains only the M3 localization chain:

```text
one checkpoint
  -> normalized hidden Gram
  -> two-neighbor residual
  -> top-two SVD subspace
  -> 2x2 support whitening
  -> Frobenius score
  -> layer argmax
```

## What remains

| Component | Status |
|---|---|
| Generic editable-weight discovery and orientation | kept |
| Hidden-Gram scale normalization | kept |
| Two-neighbor depth residual | kept |
| Deterministic top-two SVD | kept |
| 2×2 support whitening | kept |
| Frobenius layer score and deterministic argmax | kept |
| Generic fractional trimming and numerical safeguards | kept |

M3 localized 386/450 requested edits in the frozen nine-model N=50
development experiment. Its known failures include Falcon. A later five-family
signed recapture retained exactly the same M3 score and localized 58/94
successful edits because it intentionally included Falcon and OLMo.

## What was removed

| Component | Reason |
|---|---|
| M0, M1, M2 | materially worse localization than M3 |
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

This is expected: ROME's rank-one footprint is shared by generic rank-one
writers, and two different programs can produce the same final checkpoint.
The repository therefore keeps M3 localization but makes no binary ROME
provenance claim. Full evidence is in
`rome-single-checkpoint-impossibility-report.md`.

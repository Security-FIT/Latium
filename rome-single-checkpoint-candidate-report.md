# Single-checkpoint ROME detector candidate ledger

This append-only development ledger evaluates transparent checkpoint-level
statistics derived from the existing M3 layer-score vector. All 13 families
are exposed development data. The corpus currently contains 240 successful
ROME edits and 13 standalone clean checkpoints; hard negatives are still
pending. These rows cannot support a production binary claim.

| Candidate | Hypothesis | LOFO sensitivity | Clean specificity | Equal-family macro BA | Worst-family BA | AUROC | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| robust peak v1 | A ROME edit creates an unusually large peak relative to the profile median and MAD. | 82.1% | 61.5% | 73.9% | 44.7% | 85.3% | Reject: fails before hard negatives. |
| global prominence v1 | The M3 peak is unusually large relative to the second-highest eligible layer. | 65.4% | 84.6% | 75.4% | 27.8% | 80.9% | Reject: sensitivity and worst family collapse. |
| local prominence v1 | The M3 peak is unusually large relative to its immediate eligible depth neighbors. | 65.8% | 76.9% | 72.8% | 50.0% | 79.4% | Reject: sensitivity and clean specificity fail. |
| signed residual v1 | The normalized trace of the signed 2×2 M3 residual separates a ROME peak from structural curvature. | 80.9% | 60.0% | 70.9% | 50.0% | 78.5% | Reject: OLMo clean and all 20 edits are identical at the M3 peak. |
| signed residual v1 + hard negatives | The same signed threshold rejects magnitude-matched non-ROME low-rank edits. | 80.9% | 60.0% | 59.0% | 47.7% | 62.0% | Reject: 10% rank-one and 14% multi-layer specificity. |
| signed + global prominence v1 | A two-threshold conjunction separates clean curvature and generic rank-one edits. | 70.2% | 80.0% | 67.6% | 50.0% | 72.9% | Reject: 18% rank-one specificity and 64.4% overall specificity. |
| signed + robust-z logistic v1 | A deterministic two-feature linear model separates overlap that thresholds cannot. | 67.0% | 60.0% | 54.4% | 49.2% | 53.2% | Reject: worse than the transparent rule; 22% rank-one specificity. |

The cutoff in each leave-one-family-out fold is calibrated on the other
exposed families by equal-family balanced accuracy. No model-specific cutoff
or family branch is used. The complete per-case outputs are local research
artifacts; their aggregate hashes are:

- robust peak: `518c1a12c188b365c1a796231870615a183c7d0c7ce9801dd3b2c9f903583e44`
- global prominence: `cd4dd2e076a2e4159342e7258efc6632454e5bbf239385018acbb6100809005a`
- local prominence: `d798279154133dbf4211e19504845ceb7e15886e35abebfa05f419da679626d0`

The signed candidate could not be reconstructed from the frozen captures
because those artifacts retain only the nonnegative Frobenius score. Its
focused recapture was therefore necessary; unchanged scalar candidates were
not rerun.

That recapture covered five exposed families, 100 requested edits, and 94
successful edits. The signed candidate failed before hard negatives. In OLMo,
the standalone clean checkpoint and all 20 successful ROME checkpoints have
exactly the same signed-at-M3 statistic (`0.9894351452338724`), so no cutoff
can exceed 50% balanced accuracy on that family. The complete clean-only
result has SHA-256
`27e1529822ef4e4b14f0dd61b06cfa1929301fd370ac9b79372cab870ba01371`;
the raw recapture inventory has SHA-256
`cd68c02fb3da8991dfd5b5c1f12cfae995d6f10e726253228a147b984b46f9da`.

## Matched hard-negative result

The development hard-negative corpus contains 200 specimens: ten per model
and category across the same five exposed families. Rank-one, rank-two, and
multi-layer updates were matched to the historical ROME hidden-Gram change
magnitude; the maximum relative matching error was `0.0000661`. Non-target
FC edits changed a matrix the detector does not inspect and serve as an
untouched-projection control. The bundle inventory SHA-256 is
`d9f268d36887d1145d64a3f728c6a3c7d78cbab819bc0bb5ed5464ab3128f303`.

The signed rule produced 76 true positives, 18 false negatives, 74 true
negatives, and 131 false positives. Its matched-rank-one specificity was 10%.
The best family-blocked transparent two-stat rule combined signed consistency
with global prominence. It produced 66 true positives, 28 false negatives,
132 true negatives, and 73 false positives. Its matched-rank-one specificity
was 18%, and its family-blocked 95% interval for macro balanced accuracy was
56.0%–80.0%. OLMo remained exactly non-separable at the selected M3 peak.

Complete result hashes:

- signed with hard negatives:
  `09144356f6e8b793572840fe308ccdf5adcdeae0eb994340dc39aa5228fa79e6`
- exhaustive transparent two-stat rules:
  `a8b0bafdc8f19aa7da7f1e199561c94a77f1a1cad6dc1a18cd867e3cd8d79b8e`
- two-feature logistic baseline:
  `b54e4b208fbcd91baf72e4716f430a9664355a47a896688f150a44f63fa99281`

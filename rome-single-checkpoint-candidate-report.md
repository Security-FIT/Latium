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

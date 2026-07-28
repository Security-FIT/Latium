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

The cutoff in each leave-one-family-out fold is calibrated on the other
exposed families by equal-family balanced accuracy. No model-specific cutoff
or family branch is used. The complete per-case outputs are local research
artifacts; their aggregate hashes are:

- robust peak: `518c1a12c188b365c1a796231870615a183c7d0c7ce9801dd3b2c9f903583e44`
- global prominence: `cd4dd2e076a2e4159342e7258efc6632454e5bbf239385018acbb6100809005a`
- local prominence: `d798279154133dbf4211e19504845ceb7e15886e35abebfa05f419da679626d0`

The next candidate is a compact signed statistic of the M3 three-layer
residual. It cannot be reconstructed from the frozen captures because those
artifacts retain only the nonnegative Frobenius score. A focused recapture is
therefore required; this is not evidence for rerunning unchanged scalar
candidates.

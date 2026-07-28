# Single-checkpoint binary ROME detector: development conclusion

## Conclusion

The requested provenance detector is not identifiable from one final
checkpoint under the stated threat model.

A one-checkpoint M3 profile can localize a broad class of low-rank anomalies,
but the tested architecture-neutral checkpoint statistics cannot distinguish
ROME edits from magnitude-matched non-ROME rank-one updates. The strongest
defensible public output remains M3 layer localization. It must not be exposed
as a calibrated boolean ROME verdict.

This is both an algebraic and empirical conclusion:

1. If two programs produce the same final weight tensor, every deterministic
   one-checkpoint detector receives exactly the same input and must return the
   same output. Program provenance is therefore not a function of final
   weights alone.
2. ROME's rank-one update yields a rank-at-most-two hidden-Gram change, but an
   arbitrary rank-one writer yields the same algebraic class.
3. In the matched development corpus, the candidate rules detected that
   generic class: rank-one specificity ranged from 10% to 30%, far below the
   required 90%.

The narrower statement supported by the evidence is:

> A suspicious checkpoint may contain a localized low-rank-edit signature.
> That signature does not identify ROME as the producing program.

## Provenance

All runs used the remote A100 host `150-136-40-217` through the persistent
tmux session `single-checkpoint-rome`.

- detector capture commit: `f03af7bf3e47a1e559b419720cdb8fd12b36f6bf`
- matched-negative generator commit: `27edd75`
- latest-main model/runtime overlay: `29ea56819d0f1f9f2a51fc41bb0f76292aa92ada`
- result evidence commit before this report: `e879fc3`

The main overlay was limited to model loading, ROME generation, and current
model configuration. It was not a detector input. Exact 100,000-sample
covariances were used only to generate labeled ROME checkpoints.

The detector itself consumed only the suspect checkpoint's ordered editable
projection weights. No baseline weights, case metadata, configured edit
layer, covariance, second moment, causal trace, model name, or family-specific
threshold entered any candidate decision.

## Development corpus

Every model below was already exposed in previous detector work. This is a
development failure analysis, not a held-out generalization result.

| Model | Requested ROME | Successful | M3 exact |
|---|---:|---:|---:|
| GPT-2 XL | 20 | 18 | 17 |
| Mistral 7B v0.1 | 20 | 20 | 19 |
| Falcon 7B | 20 | 19 | 5 |
| OLMo 3 7B | 20 | 20 | 0 |
| Granite 4 Micro | 20 | 17 | 17 |
| **Total** | **100** | **94** | **58** |

The signed recapture inventory SHA-256 is
`cd68c02fb3da8991dfd5b5c1f12cfae995d6f10e726253228a147b984b46f9da`.
Six failed ROME requests were excluded from the positive class and were not
relabeled as negatives.

The negative corpus contains:

- five standalone clean checkpoints, counted once each;
- 50 matched random rank-one projection edits;
- 50 matched random rank-two projection edits;
- 50 non-target FC edits;
- 50 matched multi-layer rank-one edits.

The low-rank perturbations were matched to historical successful-ROME
hidden-Gram change magnitudes. Maximum relative matching error was
`0.0000661`. The hard-negative bundle inventory SHA-256 is
`d9f268d36887d1145d64a3f728c6a3c7d78cbab819bc0bb5ed5464ab3128f303`.

The corpus does not include ordinary fine-tunes, merged LoRA checkpoints,
quantization, or a second knowledge editor. Those categories were not needed
to reject the detector: the required matched rank-one category already
falsified the binary claim.

## Candidate results

All cutoffs were global. Development evaluation used leave-one-exposed-family
out folds with equal-family calibration.

| Candidate | Sensitivity | Overall specificity | Macro BA | Worst family | AUROC | Rank-one specificity |
|---|---:|---:|---:|---:|---:|---:|
| robust M3 peak | 82.1% | 61.5% clean-only | 73.9% | 44.7% | 85.3% | not run |
| global prominence | 65.4% | 84.6% clean-only | 75.4% | 27.8% | 80.9% | not run |
| local prominence | 65.8% | 76.9% clean-only | 72.8% | 50.0% | 79.4% | not run |
| signed M3 residual | 80.9% | 36.1% | 59.0% | 47.7% | 62.0% | 10.0% |
| signed + global prominence | 70.2% | 64.4% | 67.6% | 50.0% | 72.9% | 18.0% |
| signed + robust-z logistic | 67.0% | 42.9% | 54.4% | 49.2% | 53.2% | 22.0% |

The best transparent two-statistic rule produced:

```text
TP = 66   FN = 28
TN = 132  FP = 73
```

Its family-blocked 95% intervals were:

- sensitivity: 42.6%–96.7%;
- specificity: 30.7%–87.3%;
- macro balanced accuracy: 56.0%–80.0%.

No bound approaches the completion gates. The tiny two-feature logistic
baseline was tested only after all transparent candidates failed; it did not
recover the lost specificity.

## Named failure mechanisms

### OLMo is invisible at the M3 maximum

The OLMo standalone clean checkpoint, all 20 successful ROME edits, and all
40 hard negatives have the exact same signed statistic at the selected M3
peak: `0.9894351452338724`. The edit changes scores near layer 6, but the much
larger native peak remains at layer 3. Any rule restricted to statistics at
the M3 maximum is therefore non-separable on this family.

### Falcon's native peak masks the edit

Falcon's clean M3 maximum is at layer 9 while ROME edits layer 5. Exact
localization is only 5/19 in this recapture. A presence threshold can classify
the shared native peak as positive or negative, but cannot simultaneously
retain clean specificity and ROME sensitivity.

### Random rank-one edits reproduce the signal

On GPT-2 XL, the signed ranges were:

```text
ROME:                 0.9504–1.3835
matched random rank1: 0.9508–1.1189
```

The overlap is the expected algebraic confounder, not a calibration accident.
An exhaustive two-threshold rule fitted and tested on GPT-2 itself could
retain at most 55.6% ROME sensitivity while satisfying the negative
specificity gates.

## Acceptance-gate disposition

| Gate | Result |
|---|---|
| One suspect checkpoint only | satisfied by all candidates |
| No family branches or per-model thresholds | satisfied |
| ROME sensitivity ≥95% | failed |
| Clean specificity ≥95% | failed |
| Each hard-negative specificity ≥90% | failed, especially rank-one |
| Macro balanced accuracy ≥95% | failed |
| Worst-family balanced accuracy ≥80% | failed |
| M3 localization non-inferiority | unchanged by the decision experiments |
| Final untouched-family run | correctly not opened; no candidate earned it |

Because no candidate passed development, no calibration or untouched-family
corpus was exposed. Running new families would not repair a rule that fails
its own exposed development and required hard-negative gates.

## Production decision

Production remains `rome-detector-minimal-v2`, a one-checkpoint M3 layer
localizer. It emits a selected layer and evidence profile but no boolean ROME
claim. B0 and every clean-reference field remain removed.

The signed capture, hard-negative generator, threshold evaluators, and learned
baseline are experiment-only evidence and are removed from the active runtime
after this report. Git history preserves their implementations and detailed
results.

Detailed result SHA-256 values:

- signed clean-only:
  `27e1529822ef4e4b14f0dd61b06cfa1929301fd370ac9b79372cab870ba01371`
- signed with hard negatives:
  `09144356f6e8b793572840fe308ccdf5adcdeae0eb994340dc39aa5228fa79e6`
- transparent two-stat:
  `a8b0bafdc8f19aa7da7f1e199561c94a77f1a1cad6dc1a18cd867e3cd8d79b8e`
- two-feature logistic:
  `b54e4b208fbcd91baf72e4716f430a9664355a47a896688f150a44f63fa99281`

## Final production smoke

The surviving public API was run from a clean detached remote worktree at
commit `0680578254c2181a859c3884e82460c4e9695cdd` on GPT-2 XL. It loaded one
checkpoint, extracted only editable projection weights, and returned:

```text
schema_version = rome-detector-minimal-v2
selected_layer = 42
eligible_layers = 40
boolean_present = false
```

The direct API exited successfully. No covariance, causal trace, clean
reference, or ROME request was supplied.

# GOAL: Build a Minimal Single-Checkpoint Binary ROME Detector

Use this entire document as the goal and steering prompt for the next agent.
Do not stop after producing another proposal or smoke test. Iterate,
recompute, and validate until the completion gate is genuinely satisfied or a
scientific impossibility is demonstrated with evidence.

## Workspace and starting point

Work in:

```text
/home/metju/Latium
```

Branch:

```text
detector-simplification
```

Expected starting head:

```text
e20788e docs: clarify detector simplification in English and Slovak
```

Remote execution host:

```text
ssh ubuntu@150.136.40.217
```

Use persistent `tmux` sessions for remote runs. Do not modify `main`,
`new-causal-tracing`, or unrelated worktrees. Current causal tracing and model
configuration come from the latest `origin/main`.

## 1. Non-negotiable threat model

The production detector must take exactly **one suspect checkpoint**.

At inference it must not require or consume:

- the original clean checkpoint;
- baseline weights;
- a paired clean/suspect subtraction;
- `baseline_proj_weights`;
- a clean-reference artifact;
- the ROME request, prompt, subject, target, or case ID;
- the configured ROME edit layer;
- causal-tracing outputs;
- second moments or covariance matrices;
- model-family-specific thresholds or branches.

The public operation must conceptually be:

```text
detect_one_checkpoint(suspect_checkpoint)
    -> {
         is_rome_like: true | false,
         selected_layer: int | null,
         evidence: ...,
       }
```

The detector may inspect only the suspect model's editable projection weights
and generic structural metadata needed to find and order those weights.

ROME covariance and causal tracing may be used **only to generate labeled ROME
positive checkpoints for evaluation**. They are not detector inputs.

## 2. What “clean reference” meant and why it is removed

The current B0 path compares:

```text
suspect model weights - original unedited model weights
```

That requires possession of the exact clean model at inference. This is
outside the required threat model.

Therefore:

1. stop running B0;
2. remove B0 from the active capture, analysis, resident API, schema, tests,
   and default documentation;
3. remove the need to pass clean weights into the minimal detector;
4. bump the minimal schema to a clear single-checkpoint version such as
   `rome-detector-minimal-v2`;
5. preserve the historical B0 reports and Git history as evidence, but do not
   keep dead B0 production machinery.

Do not substitute a hidden baseline, downloaded clean copy, or reconstructed
clean weights. The finished detector must work when no clean reference exists.

## 3. Critical distinction: negative evaluation is still required

“No clean reference” does **not** mean that validation may contain only ROME
positives.

A standalone clean checkpoint may and must be passed to the same one-input
detector as a **negative test specimen**:

```text
detect_one_checkpoint(clean_checkpoint) -> false
```

The detector must never receive that clean checkpoint together with its
edited counterpart. There is no subtraction and no pairing at inference.

Without standalone clean and hard-negative specimens, false-positive rate and
specificity cannot be measured. A binary “ROME: yes/no” claim is scientifically
impossible without negative examples. If such negative specimens cannot be
obtained, do not claim completion and do not invent a theoretical cutoff.

## 4. Preserve the mathematical core

Retain the current architecture-neutral M3 layer score unless an ablation on
proper development data proves a replacement is better.

For each editable projection `W_l`:

```text
C_l = hidden_gram(W_l) / ||W_l||_F^2
N_l = (C_{l-1} + C_{l+1}) / 2
R_l = C_l - N_l
U_l = top two singular directions of R_l
A_l = U_l^T R_l U_l
B_l = U_l^T N_l U_l
E_l = B_l^(-1/2) A_l B_l^(-1/2)
s_l = ||E_l||_F
```

The existing localization output is:

```text
selected_layer = argmax_l s_l
```

Keep:

- hidden Gram geometry;
- Frobenius/trace scale normalization;
- two-neighbor depth residual;
- deterministic top-two SVD;
- 2x2 support whitening;
- direct Frobenius score;
- generic fractional trimming;
- deterministic tie behavior.

Do not restore by default:

- `rank2_energy` as a localizer multiplier;
- `bilateral_coherence`;
- `bilateral_balance`;
- morphology products;
- hand-built feature blends;
- family-specific corrections;
- M0, M1, or M2;
- B0, B1, or B2;
- detector graph rendering;
- the old cluster-ablation framework.

The objective is a minimal binary layer-profile decision, not another large
mathematical feature collection.

## 5. Frozen development evidence

Treat all currently reported models and cases as **exposed development data**.
They are not a final held-out set.

### Existing N=50 evidence

Nine models, 450 requested edits:

```text
M3 exact localization: 386/450 = 85.8%
M3 exact on successful edits: 375/435 = 86.2%
Falcon: 9/50
```

### Existing 13-model N=20 evidence

```text
Requested: 260
Successful ROME edits: 240/260
M3 exact localization: 198/240 = 82.5%
Equal-model macro localization: 83.5%
Within one layer: 84.6%
Falcon: 5/19
OLMo: 0/20
Excluding Falcon and OLMo: 193/201 = 96.0%
```

Existing remote artifacts:

```text
analysis_out/rome-minimal-all-models-n20-v1/report.md
analysis_out/rome-minimal-all-models-n20-v1/summary.json
```

B0's 240/240 positive result is historical only. It must not be used as
evidence that a single-checkpoint binary detector works.

Do not rerun these ROME edits merely to regenerate identical inputs. Reuse and
checksum existing artifacts when they contain the weights or profiles needed
for an iteration. Recompute only the candidate detector outputs when possible.

## 6. The research question

Determine whether the layer-score vector from one checkpoint contains enough
architecture-neutral information for a binary decision:

```text
ROME-like localized edit present: yes or no
```

The strongest claim possible from weights is “ROME-like” or
“ROME-compatible,” not proof of which editing program produced the model. A
different editor can deliberately produce the same final rank-one geometry.

The hard-negative set must test this boundary rather than hiding it.

## 7. Start with the smallest candidate

The first candidate should use only the existing M3 score vector and one
globally calibrated checkpoint-level statistic.

Recommended first candidate:

```text
peak = max_l s_l
center = median_l s_l
scale = MAD_l(s_l), with a dtype-derived zero-scale safeguard
z_peak = (peak - center) / scale
```

Calibrate one global decision rule from development families. Do not reuse the
old unvalidated `sqrt(2 log n)` rule as if it had been proven.

Prefer a held-family-out empirical or conformal calibration. A stored global
calibration artifact is allowed; a paired clean checkpoint at inference is
not.

If the one-statistic candidate fails, iterate in this order:

1. peak prominence relative to the second-highest eligible layer;
2. peak prominence relative to a local depth neighborhood;
3. one compact signed three-layer residual consistency statistic;
4. a small two-statistic rule selected through family-blocked validation.

At most two additional scalar statistics beyond the M3 score may enter a
production candidate unless an explicit ablation proves that each extra
quantity materially improves held-family performance. Delete rejected
candidates after recording their results.

Do not create an opaque learned classifier unless all transparent candidates
fail. If a learned classifier is tested, restrict it to a very small number of
architecture-normalized scalar inputs and compare it against the one-statistic
baseline.

## 8. Required negative corpus

Build a versioned manifest whose unit is a complete checkpoint, not a
duplicated case row.

It must include:

### Clean negatives

- standalone unedited base checkpoints;
- multiple independent model families and scales;
- each identical clean checkpoint counted once, not once per CounterFact
  request.

### Hard negatives

Include as many of the following as are feasible:

- matched random rank-one updates to an editable projection;
- matched random rank-two or low-rank perturbations;
- a non-ROME knowledge-editing method;
- ordinary short fine-tunes;
- LoRA merged into the base weights;
- quantized/dequantized checkpoints;
- model merges;
- edits to non-target matrix families;
- multiple-layer edits.

Match perturbation magnitudes to observed successful ROME updates where
applicable. Otherwise the detector may learn only that “a large change
exists.”

Do not label a failed ROME request as a negative without verifying whether it
changed the weights. Edit failure is not equivalent to an untouched model.

## 9. Split discipline

Generalization must be measured by model family, not random case rows.

Use three disjoint levels:

1. **Development families:** iterate on formulas and implementation.
2. **Calibration families:** choose the single global threshold or conformal
   cutoff.
3. **Final untouched families:** run once after the candidate is frozen.

All 13 models in the reported N=20 table are already exposed and may only be
development data.

For the final gate, add previously unused model families or checkpoints.
Neither their positive cases nor their clean/hard-negative results may be
examined while choosing the formula.

Never place edits derived from the same base checkpoint on both sides of a
family split. Use family-blocked or base-model-blocked resampling and
confidence intervals.

If the final untouched set fails, record the failure. Do not tune on it and
call the same set held out again. Return to development and reserve new
untouched families for the next final gate.

## 10. Iteration loop

For every candidate iteration:

1. State one concrete hypothesis.
2. Implement the smallest code needed behind an experimental opt-in.
3. Add mathematical and serialization invariants.
4. Recompute the candidate on cached development artifacts.
5. Run a small execution smoke only if the capture contract changed.
6. Evaluate binary sensitivity, clean specificity, each hard-negative
   specificity, balanced accuracy, AUROC, and calibration.
7. Evaluate exact and within-one localization on successful ROME edits.
8. Report equal-family macro results and the worst family, not only pooled
   micro totals.
9. Measure detector-only runtime, peak memory, and artifact size.
10. Compare with the previous iteration in an append-only ledger.
11. Keep the candidate only if the improvement survives family-blocked
    uncertainty analysis.
12. Delete dead candidate code before starting the next iteration.

Do not stop after the first failed formula. Try the next minimal hypothesis.
Do not repeatedly rerun ROME generation when only scalar decision math
changed.

Use only the required structural profile and candidate detector. Disable:

- unrelated detectors;
- normal analysis presets;
- rendering;
- graph creation;
- unused captures;
- M0-M2 and B0-B2 evaluation.

## 11. Completion gates

The task is complete only if one frozen, single-checkpoint detector satisfies
all of the following on previously untouched model families:

### Binary presence

```text
ROME-positive sensitivity >= 95%
standalone-clean specificity >= 95%
hard-negative specificity >= 90% for every required category
equal-family macro balanced accuracy >= 95%
worst-family balanced accuracy >= 80%
```

Also report family-blocked 95% uncertainty intervals. If sample size is too
small for meaningful intervals, collect more independent checkpoints instead
of declaring success.

### Localization

The presence rule must not degrade the retained M3 localizer:

```text
equal-family macro exact localization is non-inferior to M3
non-inferiority margin = 2.5 percentage points
```

Report Falcon, OLMo, and every new family individually. Do not exclude a weak
family from the headline result.

### Simplicity

- one suspect checkpoint at inference;
- no family-specific branch;
- no per-model threshold;
- no clean-reference input;
- no more than two justified scalar decision statistics beyond M3;
- one compact schema and one public API;
- no inactive experimental code in the production path.

### Cost

The final binary calculation should add no more than 10% detector-only runtime
and no more than 10% peak working memory over M3 localization on the same
artifacts. If it exceeds this, explain the measured cause and simplify again.

These are acceptance criteria for the experiment, not hardcoded detector
hyperparameters.

## 12. OPT, Qwen, and covariance recovery

OPT and Qwen must not be silently excluded merely because an existing
covariance has the wrong layer, revision, shape, or sample count.

When new ROME positives are required:

1. use latest `origin/main` causal tracing to determine the current target
   layer;
2. validate model revision, layer, shape, dtype, corpus method, and sample
   count;
3. if any field is absent or mismatched, calculate a new exact 100,000-sample
   covariance for the traced layer;
4. checksum and record it;
5. run only ROME generation and the candidate single-checkpoint detector.

Again, this covariance is used to create the ROME-positive test checkpoint.
It must never be read by the detector.

## 13. Production consolidation after evidence

Do not replace production with an unvalidated candidate.

Once a candidate passes the completion gates:

1. freeze a compact versioned golden fixture;
2. verify that B0 and all clean-reference API fields have already been
   removed from the active detector;
3. consolidate the winning suspect-only boolean with M3;
4. bump the artifact schema;
5. remove rejected experimental candidates;
6. update `docs/detector.md`, `comparison.md`,
   `comparison-sk.md`, and `explain-simplified.md`;
7. add direct API, schema, invariance, negative, and regression tests;
8. verify focused and full test suites;
9. run Ruff and any repository formatting checks;
10. run one clean remote smoke from the exact committed integration tree;
11. commit and push the result to `origin/detector-simplification`;
12. leave local and remote worktrees clean.

The final report must include:

- exact commit and integration-tree provenance;
- dataset and checkpoint hashes;
- all included and excluded models with reasons;
- positive, clean-negative, and hard-negative confusion matrices;
- micro, equal-family macro, and worst-family metrics;
- confidence intervals;
- localization metrics;
- runtime, memory, and artifact-size comparison;
- failed candidate ledger;
- honest claim boundary.

## 14. Stop conditions

Do not mark the goal complete because:

- a smoke test ran;
- ROME positives alone look good;
- the detector localized a layer;
- pooled accuracy is high after excluding Falcon or OLMo;
- B0 was 240/240;
- one exposed split passes;
- there is no time left.

Mark it complete only when the frozen single-checkpoint detector passes the
full untouched-family gates and production is consolidated.

If repeated architecture-neutral candidates cannot separate ROME positives
from matched non-ROME rank-one hard negatives, report the scientific
conclusion explicitly:

> ROME provenance is not identifiable from a single final checkpoint under
> this threat model; only a broader localized low-rank-edit signature is
> detectable.

That evidence-backed conclusion is preferable to a misleading binary
detector. Continue investigating simpler defensible claim scopes before
stopping.

## 15. Immediate first actions

1. Push the two existing local documentation commits if they are still ahead
   of `origin/detector-simplification`.
2. Audit and checksum the N=20 artifacts on the remote host.
3. Freeze their M3 results as development evidence.
4. Create the negative-corpus manifest without duplicating identical clean
   checkpoints.
5. Disable B0 and all unrelated capture/analysis/render paths in the
   experimental runner.
6. Implement the one-statistic robust M3 peak candidate.
7. Evaluate it on family-blocked development and calibration data.
8. Record the result, then continue through the minimal candidate order until
   a candidate earns the final untouched-family run.

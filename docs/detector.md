# Artifact-Only Structural Analyses

Structural analyses consume saved captures through `AnalysisContext`; they are
not part of model execution.

Capture reusable measurements:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.run.edit_methods='[rome]' \
  structural.capture.profile=paper \
  structural.capture.matrix_features.feature_set=paper \
  structural.run.run_id=detector-input
```

Run the default detector preset:

```bash
python -m src command=structural/analyze \
  structural.analyze.run_root=analysis_out/detector-input \
  structural.analysis.preset=paper
```

Run only the CCS composite detector. A structural run automatically captures
its spectral primitives and required matrix columns:

```bash
python -m src command=structural/run \
  structural.run.models='[qwen3-8b]' \
  structural.analysis.preset=ccs-composite
```

Analyses are stored under:

```text
plans/<model>/<plan-id>/baseline/analysis/<category>/<analysis>/<config-hash>.json
plans/<model>/<plan-id>/methods/<method>/analysis/<category>/<analysis>/<config-hash>.json
```

| Analysis | Required captures |
|---|---|
| `spectral` | `spectral` |
| `blind` | `matrix-features` with its seven required columns |
| `ccs-composite` | `matrix-features` with its five required columns, `spectral` |
| `gpt-norm-cv` | `matrix-features` with `norm_cv` |
| `rank1-blind` | `matrix-features` with its six required columns |
| `edit-presence` | `matrix-features` with its six required columns |
| `bottom-rank-svd` | `bottom-rank-tokens` |
| `gram-localization` | `gram-localization` (single checkpoint) |
| `rome-profile-experiments` | `gram-localization` |
| `rome-matrix-experiments` | `gram-experiments-v1` |
| `rome-control-experiment` | `gram-experiments-v1`, `gram-control-v1` |

Artifact studies (`ipr`, `symmetry`, `interlayer`, `attention`, and `matrix-anomaly`)
use the same contract and are stored under `analysis/artifact-study/`.

Every analysis ID is also a single-method preset. During `structural run`, the
selected analyses automatically add their declared captures and matrix columns;
users do not need to coordinate a separate capture profile. Explicitly disabling
a required capture is rejected before model execution. Analysis-only replay still
requires those artifacts to exist in the saved run because it cannot load the model.
The default capture profile is `none`, so end-to-end runs collect only what the
selected analyses request. Capture-only runs must choose a profile or enable a
capture explicitly.
Selected end-to-end renderers participate in the same dependency resolution;
for example, the structural artifact grid adds its five matrix columns even when
the selected detector itself needs a narrower feature set.

`gpt-norm-cv` is selected for GPT model families. `ccs-composite` is selected for
other model families. Unsupported selections produce an `unavailable`
artifact.

The CCS composite and GPT norm-CV calculations live in:

- `src/structural/detectors/composite.py`
- `src/structural/detectors/gpt_norm_cv.py`

`matrix-features` is a scalar feature capture with Hydra-selected feature sets.
The `paper` set stores only `spectral_gap`, `top1_energy`, `row_alignment`,
`norm_cv`, and `effective_rank`. Bottom-rank SVD/token sweeps are not part of
`matrix-features`; they live in `bottom-rank-tokens`.

An analysis-only replay never recomputes a missing measurement from a model.
For example, when preparing captures separately for a later
`bottom-rank-svd` replay, the capture-only command must request its input:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.capture.profile=paper \
  structural.capture.enable='[bottom-rank-tokens]'
```

## ROME layer localization

The one-checkpoint localizer is an explicit opt-in and does not change
the `paper` defaults. Use
`structural.capture.profile=gram-localization` together with
`structural.analysis.preset=gram-localization`.

For an editable projection matrix $W_l$, it builds a normalized Gram matrix in
the smaller hidden space:

```text
G_l = W_l W_l^T / ||W_l||_F^2   when rows <= columns
G_l = W_l^T W_l / ||W_l||_F^2   otherwise
```

Each eligible layer is compared with the mean of its immediate neighbors. The
score is the norm of the two leading residual singular values divided by the
neighbor Gram's support in those directions. The highest score wins and exact
ties select the lower layer. Eligibility trims 10% at each end while always
excluding the first and last layers.

The capture contains only the `diagonal_relative` profile field. It validates
matrix shape, finite non-zero weights, complete scores, unique layers, and the
minimum three-layer requirement. This method localizes a suspected ROME-style
edit; it does not decide whether a checkpoint has been edited. It uses no clean
reference checkpoint, causal trace, covariance, prompt, model identity, or edit
metadata.

The checked-in replay fixture represents 13 model families and records 196
exact localizations among 240 successful edits (81.67%). It is development
evidence, not an independent scientific test set.

## Opt-in ROME experiments

The false-positive experiments reuse the structural capture and analysis
workflow. They do not change `gram-localization` or the default detector preset.
The scalar experiments need no recapture:

```bash
python -m src command=structural/analyze \
  structural.analyze.run_root=analysis_out/<run-id> \
  structural.analysis.preset=rome-profile-experiments
```

The matrix experiments share one versioned projection capture. The default
`neighbors` group records the unchanged score, separate-neighbor agreement,
the refined-basis control, bounded contrast, support, residual magnitude, and
numerical diagnostics:

```bash
python -m src command=structural/run \
  structural.run.models='[gpt2-large]' \
  structural.analysis.preset=rome-matrix-experiments \
  structural.run.run_id=rome-neighbor-experiments
```

Enable wider measurements explicitly. Selecting the corresponding matrix
experiment in `analysis.methods` also adds its required group automatically:

```bash
python -m src command=structural/run \
  structural.run.models='[gpt2-large]' \
  structural.analysis.preset=rome-matrix-experiments \
  structural.capture.rome_experiments.groups='[neighbors,quadratic,footprint]' \
  structural.analysis.methods.rome-matrix-experiments.experiments='[baseline-affine-mdl-v1,quadratic-neighbor-affine-mdl-v1,signed-footprint-mdl-v1]' \
  structural.run.run_id=rome-full-experiments
```

`footprint` performs full symmetric eigendecompositions and is intentionally
opt-in. Use a new run ID when capture groups change. Capture configuration is
hashed, and analysis configurations are stored under their existing config-hash
paths.

The internal-control experiment fixes the control family to attention output
projections (`o_proj`). Unsupported architectures return unavailable cases:

```bash
python -m src command=structural/run \
  structural.run.models='[gpt2-large]' \
  structural.analysis.preset=rome-control-experiment \
  structural.run.run_id=rome-control-experiment
```

Generate a model-free comparison report from any manifest-backed run:

```bash
PYTHONPATH=. python scripts/evaluate_binary_rome_presence.py \
  analysis_out/<run-id> --output-dir analysis_out/<run-id>/rome-report
```

The report treats completed ROME cases as positives and the baseline capture as
the clean control. It reports unavailable cases instead of counting them as
clean. These are exposed development comparisons; the formulas do not supply a
calibrated deployment threshold or prove which editing procedure produced a
checkpoint.

## End-to-end order

`structural run` records and analyzes the unedited baseline as well as edited
states. For each plan it captures the clean model once, applies ROME to each
selected case, captures that state, and restores the model before the next
case. The configured analyses then score the baseline and every completed ROME
attempt from those captures; renderers run only after analysis finishes. Thus
`structural.run.n_tests=N` means N independent ROME attempts, not N cumulative
edits to one checkpoint.

Detector analysis does not depend on behavioral edit efficacy: a completed
capture is analyzed even when the ROME efficacy metric is below its success
threshold. This keeps detector evaluation distinct from edit evaluation.

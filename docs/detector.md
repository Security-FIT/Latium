# Structural analyses

A structural run loads the model, saves baseline measurements, applies each
edit independently, captures its weights and restores the model. Analyses then
read saved captures; graph rendering follows analysis. `n_tests=N` means N
independent edits. A completed capture is analyzed even if ROME efficacy is low.

```bash
python -m src structural run \
  'structural.run.models=[qwen3-8b]' \
  structural.analysis.preset=ccs-composite
```

The selected analyses and renderers automatically request their required
captures and matrix columns. Explicitly disabling a required capture is an
error. The `paper` preset selects CCS for non-GPT models, norm-CV for GPT
models, and spectral analysis. Unsupported methods produce unavailable results.

| Analysis | Required capture |
|---|---|
| `spectral` | `spectral` |
| `blind` | `matrix-features` |
| `ccs-composite` | `matrix-features`, `spectral` |
| `gpt-norm-cv` | `matrix-features` (`norm_cv`) |
| `rank1-blind`, `edit-presence` | `matrix-features` |
| `bottom-rank-svd` | `bottom-rank-tokens` |
| `gram-localization` | `gram-localization` |

Every analysis ID is also a single-method preset. Matrix features are selected
by the consumer; the `paper` feature set contains `spectral_gap`, `top1_energy`,
`row_alignment`, `norm_cv` and `effective_rank`.

## Replay saved captures

```bash
python -m src structural analyze \
  structural.analyze.run_root=analysis_out/<run-id> \
  structural.analysis.preset=paper
```

Replay loads artifacts without loading a model. Required measurements must
already exist. Capture-only runs choose a profile or explicit captures:

```bash
python -m src structural capture \
  'structural.run.models=[gpt2-large]' \
  structural.capture.profile=paper
```

Analyses are stored under each plan's baseline or edited method:

```text
plans/<model>/<plan-id>/baseline/analysis/<category>/<analysis>/<config-hash>.json
plans/<model>/<plan-id>/methods/<method>/analysis/<category>/<analysis>/<config-hash>.json
```

## Gram layer localization

Use `structural=gram` for ROME and Gram capture/analysis, or follow the
[fleet guide](gram-workflow.md) for shared facts, batches and reports.

For each projection matrix, Gram uses the smaller hidden space:

```text
G = W Wᵀ / ||W||²_F   when rows <= columns
G = Wᵀ W / ||W||²_F   otherwise
```

Each eligible layer is compared with its two immediate neighbors' mean Gram
matrix. The score is the norm of the two leading residual singular values,
each divided by the neighbor Gram's support in that direction. The highest
score wins; ties select the lower layer. Eligibility trims 10% at each end
and excludes the first and last layers.

The saved score field is `diagonal_relative`. Gram uses one checkpoint's
projection weights and needs at least three layers. It localizes a suspected
edit; its argmax does not establish whether an edit exists.

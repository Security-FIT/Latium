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

Run only the composite detector:

```bash
python -m src command=structural/analyze \
  structural.analyze.run_root=analysis_out/detector-input \
  structural.analysis.preset=none \
  structural.analysis.enable='[composite]'
```

Analyses are stored under:

```text
plans/<model>/<plan-id>/methods/<method>/analysis/<category>/<analysis>/<config-hash>.json
```

| Analysis | Required captures |
|---|---|
| `spectral` | `spectral` |
| `blind` | `matrix-features` with `feature_set=blind` |
| `composite` | `matrix-features` with paper features, `spectral` |
| `gpt-norm-cv` | `matrix-features` with `norm_cv` |
| `rank1-blind` | `matrix-features` with `feature_set=rank1` |
| `edit-presence` | `matrix-features` with `feature_set=edit-presence` |
| `bottom-rank-svd` | `bottom-rank-tokens` |

Artifact studies (`ipr`, `symmetry`, `interlayer`, `attention`, and `matrix-anomaly`)
use the same contract and are stored under `analysis/artifact-study/`.

`gpt-norm-cv` is selected for GPT model families. `composite` is selected for
other model families. Unsupported selections produce an `unavailable`
artifact.

The composite and GPT norm-CV calculations live in:

- `src/structural/detectors/composite.py`
- `src/structural/detectors/gpt_norm_cv.py`

`matrix-features` is a scalar feature capture with Hydra-selected feature sets.
The `paper` set stores only `spectral_gap`, `top1_energy`, `row_alignment`,
`norm_cv`, and `effective_rank`. Bottom-rank SVD/token sweeps are not part of
`matrix-features`; they live in `bottom-rank-tokens`.

An analysis never recomputes a missing measurement from a model. For example,
`bottom-rank-svd` requires an explicitly enabled capture unless using `full`:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.capture.profile=paper \
  structural.capture.enable='[bottom-rank-tokens]'
```

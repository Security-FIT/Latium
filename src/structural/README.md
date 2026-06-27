# Structural

Structural runs are split into model-dependent capture, model-free analysis, and
manifest-backed rendering.

```text
runner.py
  -> execution/  load model, select cases, run edits
  -> capture/    write baseline and edited-state artifacts
  -> analysis/   replay saved captures without loading the model
  -> detectors/  shared detector math and resident adapters
  -> ../graphs/  render graph artifacts from manifests
```

## Add A Capture

1. Add a producer in `capture/producers.py`.
2. Register it in `capture/registry.py`.
3. Add it to a capture profile if it should run by default.
4. Add config-hash inputs in `capture/artifacts.py` when options affect output.

Captures must return JSON-serializable data. Baseline captures should store
reusable layers. Edited captures may store patches or full profiles depending
on downstream consumers. `matrix-features` stores all edited layers because
composite analyses and structural graph renderers need complete post-edit layer
profiles.

`matrix-features` is configured through Hydra:

- `structural.capture.matrix_features.feature_set`
- `structural.capture.matrix_features.features`
- `structural.capture.matrix_features.svd_top_k`

Shared matrix profile math lives in `detectors/profiles.py` as
`matrix_basic_profile`, `matrix_svd_profile`, and the composed
`matrix_profile`.

## Add An Analysis Or Study

1. Add the runner in `analysis/detector_methods.py` or `analysis/studies.py`.
2. Register it in `analysis/registry.py`.
3. Declare `required_captures`.
4. Add default method config under `src/config/structural/default.yaml` if
   needed.
5. Test missing inputs and happy-path output.

Use `category="detection"` for layer prediction and
`category="artifact-study"` for exploratory saved analyses.

## Add Detector Math

Put reusable model-free math in `detectors/`. If it also needs live model state,
keep the live adapter separate from the artifact-only analysis runner.

Use shared helpers instead of duplicating formulas:

- `src.common.arrays.local_zscore` and `curvature` for local layer transforms.
- `detectors/profiles.py` for per-layer matrix profile fields.

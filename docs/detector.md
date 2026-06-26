# Artifact-Only Structural Analyses

Structural analyses consume saved captures through `AnalysisContext`; they are
not part of model execution.

Capture reusable measurements:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.run.edit_methods='[rome]' \
  structural.capture.profile=spectral \
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
| `blind` | `matrix-features` |
| `composite` | `matrix-features`, `spectral` |
| `gpt-norm-cv` | `matrix-features` |
| `rank1-blind` | `matrix-features` |
| `edit-presence` | `matrix-features` |
| `bottom-rank-svd` | `bottom-rank-tokens` |

Artifact studies (`ipr`, `symmetry`, `interlayer`, `attention`, and `matrix-anomaly`)
use the same contract and are stored under `analysis/artifact-study/`.

`gpt-norm-cv` is selected for GPT model families. `composite` is selected for
other model families. Unsupported selections produce an `unavailable`
artifact.

The composite and GPT norm-CV calculations live in:

- `src/structural/detectors/composite.py`
- `src/structural/detectors/gpt_norm_cv.py`

An analysis never recomputes a missing measurement from a model. For example,
`bottom-rank-svd` requires a `full` capture profile or an explicitly enabled
capture:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.capture.profile=spectral \
  structural.capture.enable='[bottom-rank-tokens]'
```

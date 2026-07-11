# Artifact-Only Structural Analyses

Structural analyses consume saved captures through `AnalysisContext`; they are
not part of model execution.

Capture the legacy paper measurements explicitly:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large]' \
  structural.run.edit_methods='[rome]' \
  structural.capture.profile=paper \
  structural.capture.matrix_features.feature_set=paper \
  structural.run.run_id=detector-input
```

Run the legacy paper detector preset:

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

Run the architecture-neutral weighted-spectrum detector:

```bash
python -m src structural run \
  structural.run.models='[gpt2-large,mistral-7b-v0.1,qwen3-4b,opt-6.7b]' \
  structural.capture.profile=weighted-spectrum \
  structural.analysis.preset=weighted-spectrum \
  structural.run.run_id=weighted-spectrum
```

`weighted-spectrum` is also the default capture profile and analysis preset,
so a plain structural run uses this single detector. The named `paper` profile
and preset remain available to reproduce the older GPT/non-GPT split.

Analyses are stored under:

```text
plans/<model>/<plan-id>/methods/<method>/analysis/<category>/<analysis>/<config-hash>.json
```

| Analysis | Required captures |
|---|---|
| `weighted-spectrum` | `weighted-spectrum` |
| `rome-presence-blind-peak` | `weighted-spectrum` |
| `rome-presence-blind-footprint` | `weighted-spectrum` |
| `rome-presence-delta` | `rome-update` |
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

## Weighted-spectrum detector

`weighted-spectrum` is a blind, weight-only layer localizer. It does not route
on model names, storage layout, or model family, and its decision score has no
fitted blend weights or thresholds. The only detector settings are the allowed
first/last-layer trims (default `5/5`). Although the capture pipeline stores an
unedited artifact for patch materialization, the detector never subtracts a
clean checkpoint from the suspect weights.

For projection weight `W_l`, the capture first constructs a trace-one operator
in the shared hidden space:

```text
C_l = W_l W_l^T / ||W_l||_F^2       if rows <= columns
C_l = W_l^T W_l / ||W_l||_F^2       otherwise
```

Choosing the smaller matrix axis makes `C_l` invariant to whether a projection
is stored as a PyTorch `Linear` weight or a transposed GPT `Conv1D` weight. The
trace normalization also removes global weight scale.

Let `R_l = (C_{l-1} + C_{l+1}) / 2` and `A_l = C_l - R_l`. A rank-one weight
update induces a rank-at-most-two perturbation in the unnormalized hidden Gram,
so the capture extracts the leading two singular directions `U_l` of `A_l`.
It then measures the perturbation relative to the neighboring operator inside
that ROME-motivated subspace:

```text
B_l = U_l^T A_l U_l
G_l = U_l^T R_l U_l
E_l = G_l^(-1/2) B_l G_l^(-1/2)
score(l) = ||E_l||_F
```

The predicted layer is simply `argmax score(l)` after trimming. This affine-
relative normalization is the important new artifact: raw spectral curvature
was dominated by stable architecture-specific peaks, whereas `G_l` discounts
directions already supported by neighboring layers.

Cross-family validation used disjoint CounterFact case slices and counted only
successful ROME edits:

| Model | Cases 0-4 | Cases 5-9 | Combined |
|---|---:|---:|---:|
| GPT-2 Large | 4/4 | 4/4 | 8/8 |
| GPT-2 XL (scale holdout, layer 18) | 4/4 | — | 4/4 |
| Mistral-7B-v0.1 | 4/5 | 4/4 | 8/9 |
| Qwen3-4B | 5/5 | 5/5 | 10/10 |
| OPT-6.7B | 5/5 | 3/4 | 8/9 |
| **Total** | **22/23** | **16/17** | **38/40 (95.0%)** |

Both misses were near misses rather than family failures: Mistral layer `5`
was predicted as `7`, and OPT layer `15` as `14`. Failed edits were recorded as
`unavailable` and excluded instead of being relabeled as detector errors.
GPT-2 XL was evaluated only after the score and trimming rule were frozen.

The corresponding cluster run IDs are
`weighted-bilateral-crossfamily-n5`,
`weighted-spectrum-holdout-cases5-9`, and
`weighted-spectrum-gpt2-xl-holdout-n5`. Final-score analyses use config hash
`255ae3fd504408a57e42799558fa132697f8b7a001bee472839d8e7d09d42a6b`.

## ROME-presence decisions

The layer localizer and the binary decision are separate analyses. To capture
the inputs for all presence variants and run them together:

```bash
python -m src structural run \
  structural.capture.profile=rome-presence \
  structural.analysis.preset=rome-presence \
  structural.run.run_id=rome-presence
```

Three decisions are emitted so they can be compared before selecting one:

| Analysis | Clean checkpoint | Rule |
|---|---:|---|
| `rome-presence-blind-peak` | no | the locally whitened spectral peak exceeds the sample-size-adjusted universal extreme bound |
| `rome-presence-blind-footprint` | no | the same test after requiring the balanced, same-sign, rank-at-most-two local footprint predicted by one ROME update |
| `rome-presence-delta` | yes | exactly one canonical MLP output matrix changed and its update is rank one within a floating-point roundoff bound |

The two blind variants use only the suspect checkpoint. Their cutoff is the
universal bound `sqrt(2 log n)` over the `n` evaluated layers after robust
median/MAD normalization; there is no learned threshold or model-family
route. `blind-peak` is the more sensitive screen. `blind-footprint` is the
more ROME-specific rule because it conjoins the expected signed three-layer
shape, balanced left/right jumps, and rank-two Gram concentration without
fitted mixture weights. The reported Gaussian tail value is a diagnostic under
the universal-noise assumption, not an empirically calibrated probability;
clean and hard-negative checkpoint evaluation is still required before either
blind rule is used as a production gate.

The delta variant is the strongest attribution test when the clean checkpoint
is available. It records only scale-free update diagnostics and uses a bound
derived from the checkpoint/analysis dtype's machine epsilon. It does not use
model names, layer counts, model-family thresholds, or training. A positive result
means **ROME-like localized rank-one edit**, not proof that a particular ROME
codebase produced it: another single-rank editor can deliberately create the
same weight geometry.

All variants return `is_rome_like`, `verdict`, `anomalous_layer`, the full
evidence used by the decision, and a `threat_model` field distinguishing
suspect-only from clean-baseline analysis. The older configurable
`edit-presence` analysis remains available for reproduction, but it is not one
of these parameter-free ROME-presence variants.

They can also be called directly on canonical layer-to-weight dictionaries:

```python
from src.structural.detectors.rome_presence_resident import (
    BlindRomePresenceDetector,
    DeltaRomePresenceDetector,
)

blind = BlindRomePresenceDetector(strategy="footprint").detect(suspect_proj)
delta = DeltaRomePresenceDetector().detect(suspect_proj, clean_proj)
```

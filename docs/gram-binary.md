# GRAM binary MDL v1 (frozen experiment)

The default detector is now [GRAM binary v2](gram-binary-v2.md), implementing
the research extensions. This document describes the preserved v1 method used
by the previously submitted PBS snapshot. V1 evidence remains replayable through
the public API. V2 unit/numerical tests remain deferred; its optimized N=10
model experiment is now complete, with inadequate observed sensitivity.
See the [PBS and result record](rome-binary-pbs.md).

`gram-peel-mdl-v1` returns `edited: true/false` from one checkpoint's MLP
projection weights. A positive result means an isolated rank-one anomaly
compatible with ROME under this background code. It does not establish which
editing algorithm produced the weights. A negative result is not a proof that
the model was never edited.

The detector has no score threshold, collapse cutoff, CT input or configured
edit-layer input. It evaluates every layer eligible under the existing GRAM
interior trim (10%, with the two endpoints always excluded). No original-model
Grams or changed-layer hints enter the calculation. The ordinary GRAM
localizer and reversal acceptance rule remain available.

## Run beside ordinary GRAM

The new structural preset inherits `gram` and adds the binary analysis:

```bash
python -m src structural plan structural=gram-binary-v1
python -m src structural run structural=gram-binary-v1 \
  'structural.run.models=[gpt2-xl]' \
  structural.run.n_tests=1
```

This is the existing baseline/edit benchmark: its ROME stage still needs
covariance. Both `gram-localization` and `gram-binary` artifacts are produced.
The target layer is used only by the evaluator. Binary decisions appear in
the analysis case's `data.edited` and `data.layer`; negative results have
`layer: null`. Binary positive/negative counts are separate summary fields;
the inherited layer accuracy remains a localization metric.

For a **static scan without applying any edit, covariance or model inference**:

```bash
python scripts/gram_binary_detect.py --method v1 --model gpt2-xl \
  --out analysis_out/gram-binary/gpt2-xl.json
```

`--model` loads the checkpoint selected by that Latium model config and obtains
all projections from its architecture template. It never uses `model.layer`
to restrict the scan or decide the result. To scan exported projection weights:

```bash
# Export format: torch.save({integer_layer: projection_tensor, ...}, path).
python scripts/gram_binary_detect.py --method v1 --weights suspect-projections.pt \
  --device cuda:0 --out analysis_out/gram-binary/suspect.json
```

The command prints a small JSON verdict and writes full per-layer evidence.
It loads tensor files with `weights_only=True`. `--device` controls computation,
not the decision rule. No checkpoint weights are modified or saved by the scan.

Python API:

```python
from src.structural.detectors.gram_binary import detect, detect_from_grams

result = detect(projection_weights, method="v1")  # optional device="cuda:0"
result = detect_from_grams(normalized_hidden_grams, method="v1")
print(result["edited"], result["layer"], result["mdl_delta_bits"])
```

## Decision and exact accounting

For normalized Grams `X`, `H0` codes the uncorrected sequence. For each eligible
layer, `H1` codes a sequence with that layer replaced by its rank-one peel,
plus the layer index, correction parameters and reconstruction innovations:

```text
null_bits = common_source_bits + null_background_bits
edit_bits = common_source_bits + peeled_background_bits
          + layer_bits + parameter_bits + reconstruction_bits
edited = min(edit_bits) < null_bits
```

A tie selects `false`; equal candidate costs prefer the lower layer. The
minimum `H1` layer can differ from the original GRAM argmax. Diagnostics include
the old peak, all candidate costs, the best candidate even when rejected,
score collapse, search resolution and method/codec versions. Invalid inputs,
missing candidates and numerical failures raise errors rather than returning
`false`. Capture failures are persisted as failures by the existing pipeline.

### Fixed finite codec: `gram-rice-fp32-v1`

This first implementation uses **integer prefix-code lengths**, rather than
the optional Student-t model discussed in the original plan. This makes the
parameter costs and reconstruction explicit without estimating a probability
distribution or an effective sample size.

1. Canonicalize each symmetric normalized Gram to FP32. Code its upper
   triangle once, using a dyadic lattice with
   `fractional_bits = 23 + ceil(log2(d))`. This is a fixed numerical format,
   not a fitted classification cutoff.
2. For each signed integer block, pay for the shorter of a fixed-width code
   and a Golomb--Rice code. Both pay a mode bit and an Elias-gamma-coded
   width/parameter. Signed integers use zigzag order `0,-1,+1,-2,+2,...`.
3. The first two layers are coded directly. Each later layer can use a paid
   reset or the affine-depth predictor `2Q[l-1]-Q[l-2]`. A replacement therefore
   changes exactly blocks `l,l+1,l+2`; they are all recomputed. No stencil
   evidence is counted twice.
4. Original FP32 words are recovered by their uniform fixed-width indices
   among the float values in each lattice cell. Round-to-even boundaries and
   both signed zeros are accounted for. This source-refinement suffix is
   identical in `H0` and `H1` and cancels in the decision. FP64 input Grams are
   intentionally canonicalized to FP32 before this experiment.
5. A correction transmits two `d`-vectors and two scalars as literal FP64,
   costing `64*(2*d+2)` bits. Layer selection costs `ceil(log2(|E|))` bits.
   Quantizing the peeled Gram introduces an integer reconstruction error,
   which is separately coded. Parameters and their precision are never free.

The mode and integer-code choices are selected by their complete code costs.
There is no per-family setting or calibration on clean/edited labels. The
background predictor and codec are modeling choices; MDL is not a calibrated
probability or a guarantee of false-positive control. The module computes
code lengths and provides the reconstruction primitives; it is not a file
compression/archive utility.

### Normalization-aware candidate plane

The binary path estimates `kappa` as the median supported generalized
eigenvalue of `(A,N)`, then searches in the extreme eigen-plane of
`A-kappa*N`. Supported directions use the existing dtype/dimension numerical
tolerance. Fewer than three supported directions fall back to scale one.
This estimator is approximate when neighboring layers differ naturally.

`best_peel(..., plane_scale=kappa)` still evaluates and peels against the
original normalized reference `N`. The default `plane_scale=1` preserves the
existing reversal path. The new scale is not a positive/negative gate.

For its direction `u` and mixer `z`, the candidate is a real weight-space
rank-one correction:

```text
k_hat = W.T z
W_peeled = W - u k_hat.T
g = A z
gamma = z.T A z
H = A - u g.T - g u.T + gamma u u.T
c = tr(H),  B = H/c
A = c B + K,  K = u g.T + g u.T - gamma u u.T
```

`K` has rank at most two. The reconstruction innovations account for finite
precision. The binary path never asserts that the observed normalized
neighbor residual itself has rank two or that opposite eigenvalue signs
certify ROME.

## Validation and practical limits

The initial v1 CPU checks covered a constant clean control, a strong isolated rank-one edit,
a broad natural transition, both matrix layouts, unchanged input weights,
FP32-word and corrected-lattice reconstruction, numerical normalization,
paid integer codes, deterministic tie rules, invalid data, artifact replay,
config/CT independence, standalone CLI and preset resolution. Existing
reversal and ordinary GRAM tests were also run at that stage. These earlier
checks do not validate v2 or the later runtime optimizations. The shared codec
has since received an exact-arithmetic histogram optimization; no new tests
were executed after the user's deferral.

These are synthetic and integration checks, not validation on the real model
fleet. Weak edits can fail to pay the correction cost, and natural isolated
rank-one structure can be flagged. Some rank-one weight changes leave the
Gram unchanged and cannot be detected by this representation.

The scan caches at most five Grams and five lattice vectors, and saves compact
scalar evidence rather than every dense Gram. Peeling still uses the existing
grid/refinement search with repeated dense solves; full-size model runtime
needs measurement. No top-k screening is used to hide that cost or discard
low-ranked candidate layers. This describes the original v1 runtime. The
later v2 optimization and completed N=10 experiment are recorded separately;
see [the run history](rome-binary-pbs.md) before interpreting these decisions
as a validated ROME detector.

The follow-up [research note](rome-binary-research.md) reviews adaptive low-rank
MDL, whole-spectrum tests, noise equalization, held-out prediction and a proposed
covariance-weighted Gram extension. Their v2 implementation is documented
separately; it does not change the frozen v1 implementation or its PBS experiment.

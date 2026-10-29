# GRAM binary MDL v2

Status as of 2026-10-29: implemented and experimental, with deferred unit and
numerical tests. The optimized PBS N=10 run completed all 18 models. The binary
rule was positive on only 1/180 edited checkpoints (1/179 successful ROME edits),
with 0/18 positive clean baselines: observed sensitivity is inadequate.
See the [full PBS and result record](rome-binary-pbs.md). Job completion and
static checks do not establish a useful detector or a general false-positive
bound.

The public API and `structural=gram-binary` now select `gram-peel-mdl-v2`.
Use `method="v1"`, CLI `--method v1`, or `structural=gram-binary-v1` for the
[frozen implementation](gram-binary.md). Replay dispatches by the method and
codec recorded in an artifact, so v1 results retain their original decision.

## Implemented components

| Component | Role |
| --- | --- |
| Adaptive correction code | Selects among 8, 12, 16, 24, 32, 53 shared-exponent significant-bit formats and a literal FP64 escape. Precision index, exponents, integer parameters and reconstruction innovations are paid. |
| Competing backgrounds | Direct, constant-depth, affine-depth, quadratic-depth and adaptive predictor families. Stage changes can use paid resets. Background and block predictor choices are paid under both hypotheses. |
| Generalized full spectrum | Records the supported neighbor-whitened spectrum, its median bulk scale, all-eigenvalue squared discrepancy and Stein divergence where defined. Supplies a pulled-back candidate plane. |
| Dyson Equalizer | Reduced-SVD implementation of Algorithm 1, with row/column variance estimates, a balanced spectrum, descriptive MP distance and a candidate plane. |
| Rectangular LSS | Equations 4.16–4.20 of Jung et al., including trace correction and conditional Gaussian CLT log-likelihood ratios for a fixed subcritical SNR family. Uses a uniform mixture rather than an unpriced maximum. |
| Bi-cross-validation | Four deterministic interleaved row/column folds, each fitting its own training submatrix. Scores every rank up to the common training dimension using exact cumulative cross terms. |
| Optional covariance view | Applies the same physical left peel to normalized `WWᵀ` and `WCWᵀ`. The covariance source is coded conditionally on the ordinary view, with paid predictor choices. |
| Candidate search | One dtype-aware stabilized Cholesky factorization per layer, reused for all angular candidates. Its whole-matrix objective avoids a dense spectral score decomposition for every angle. |

Background prediction, full-spectrum discrepancy and the joint candidate
objective are modeling choices. Dyson and LSS assume particular noise models;
they are labeled `calibrated_for_llm_weights: false`. Components beyond the
asymptotic MP edge describe signal rank, which can include natural structure.
No diagnostic contributes an additional Boolean gate or a weighted vote.

BCV is a raw-matrix predictive-rank diagnostic. It does not independently
validate a peel fitted using the full matrix. It uses no full-data Gram, Dyson
scaling, whitening transform or target-layer hint. A future experiment may
compare specific candidate procedures fitted within each training split.

The literature and derivations are in [the research note](rome-binary-research.md).
Methods identified there as unsuitable final gates—such as an unqualified BBP
edge, a conformal p-value without exchangeability, or an MDL delta labeled as
an e-value—are not added as detectors. VBMF and robust decomposition remain
alternative background baselines discussed in that note, rather than additional
unvalidated presence gates.

## One binary rule

For the same fixed FP32 Gram representation in both hypotheses:

```text
null_bits = common_source_bits + min_background(paid_background_bits)
edit_bits = common_source_bits + min_layer_generator_precision_background(
    paid_background_bits + layer_bits + generator_bits
    + correction_parameter_bits + exact_reconstruction_bits
)
edited = edit_bits < null_bits
```

The common FP32 refinement suffix is inherited from v1. All source Grams,
including the optional covariance view, are restored exactly at the lattice
level by the correction and its innovations. No extra free quantization or
fitted effective sample size is introduced. Ties choose `false`; equivalent
costs use fixed background/generator/precision order and the lower layer.

The finite precision code shares the left direction across views and encodes
each view's induced moment and scalars. Each candidate's moments come from
one shared `(u,z)` weight-space peel. They are not fitted separately per view.
Quantized factors approximate that correction; the paid innovations restore
the original source. This is a description of the observed Grams, not a proof
that a unique original weight matrix or editing history was recovered.

V2 scans **every supplied contiguous layer**, including both endpoints.
Interior references average the two adjacent layers; endpoints use the one
available adjacent layer. There is no inherited 10% trim. The endpoint rule
needs empirical validation on natural early/late-layer transitions.

On valid inputs the verdict is Boolean. Invalid core inputs or failed core
candidate computation raise errors. Optional Dyson/BCV failures are persisted
with their reason as unavailable diagnostics; they are not converted into a
clean-model verdict. Every layer still has the ordinary candidate path.

## Interfaces

These are usage examples; the recorded fleet run used N=10 and precomputed ROME covariance. See the run record for exact resources and outcomes.

```bash
python scripts/gram_binary_detect.py --model gpt2-xl \
  --device cuda:0 --out analysis_out/gram-binary-v2/gpt2-xl.json

python scripts/gram_binary_detect.py --weights suspect-projections.pt \
  --device cuda:0 --out analysis_out/gram-binary-v2/suspect.json

python -m src structural run structural=gram-binary \
  'structural.run.models=[gpt2-xl]' structural.run.n_tests=10
```

`--no-bcv` disables the held-out diagnostic. `--no-dyson` disables the
equalizer diagnostic and its candidate generator, which can change the MDL
verdict. These are recorded compute variants, not score-cutoff settings.
Full diagnostics perform substantial extra linear algebra; measure a small
pilot before requesting a new fleet run or estimating PBS walltime.

Equivalent structural options:

```yaml
capture:
  gram_binary:
    method: v2
    dyson: true
    bcv: true
    covariance_manifest: null
```

Only those options are accepted. Method/codec versions and optional covariance
content signatures enter capture identity, preventing reuse of a v1 capture
as a v2 result or reuse after a metric file changes.

Python:

```python
from src.structural.detectors.gram_binary import detect, detect_from_grams
from src.structural.detectors.gram_covariance import Metric

result = detect(weights, device="cuda:0")
result = detect(weights, covariances=all_layer_covariances)
result = detect(weights, covariances={l: Metric(p, "precision") for l, p in all_layer_precisions.items()})
result = detect_from_grams(ordinary_grams, covariance_grams=weighted_grams)
```

Weights and metrics are never modified. Tensor maps use covariance `C` by
default. `Metric(..., "precision")` explicitly means `C⁻¹`. Gram-only inputs
support MDL and generalized spectra; rectangular diagnostics are unavailable.
Memory is bounded by small Gram/lattice windows plus the current layer's dense
linear algebra. Full weight maps remain owned by the caller.

## Covariance manifests

No target-layer discovery or automatic covariance computation occurs. A
manifest must cover **every supplied layer**, including endpoints. Extra or
missing layers are rejected. Paths are relative to the manifest or absolute.

For a three-layer example:

```json
{
  "version": 1,
  "kind": "precision",
  "provenance": "trusted ROME second-moment cache, matching model and neuron order",
  "layers": {
    "0": {"path": "model_0_wikipedia_100000.pt"},
    "1": {"path": "model_1_wikipedia_100000.pt"},
    "2": {"path": "model_2_wikipedia_100000.pt"}
  }
}
```

Each entry can override `kind`, specify a tensor-dictionary `key`, and include
an expected `sha256`. Raw upstream `.npz` files normally use
`{"kind":"covariance", "key":"mom2.mom2", ...}`. The repository's ROME
`.pt` cache stores the inverse moment and needs `kind: precision`. Semantics
are never inferred from file extension alone.

```bash
python scripts/gram_binary_detect.py --weights suspect-projections.pt \
  --covariance-manifest all-layer-metrics.json \
  --out analysis_out/gram-binary-v2/joint.json
```

Metrics are loaded lazily, checked for compatible shape/symmetry/positivity,
and hashed for provenance. A precision view uses a Cholesky solve rather than
forming a dense covariance inverse. Covariance loading/factorization can still
be expensive. The currently available one-edit-layer cache is not sufficient
for this all-layer mode; do not fabricate coverage or start recomputation as
a side effect of requesting detection.

## Validation state

Deferred tests cover exact adaptive reconstruction in two views, local versus
full recoding, Dyson scale equivariance, held-out rank-one prediction,
covariance/precision parity, complete manifest coverage/content identity, v2
replay and tampered parameter prices. The existing v1 tests explicitly target
the preserved implementation.

These tests have not been executed. The initial v2 PBS jobs all exceeded their
2–4 hour limits before completing 100 cases. The 32–48 hour replacement
submissions were cancelled while queued. A subsequent N=100 run used disjoint
batches capped at 24 hours; 12 batches completed before the remaining 47 queued
jobs were cancelled for the optimized N=10 experiment.

The N=10 experiment completed all 18 models, with 180 complete captures and
analyses. It produced only one positive edited-checkpoint verdict and no
positive clean baselines. Of the 180 edit attempts, 179 passed the recorded ROME
success criterion. The resulting 1/179 observed successful-edit sensitivity
is inadequate; see [per-model/per-case results](experiments/rome-binary-n10-results-20261029.yaml).
No new unit/numerical tests were run to prepare these records.

Interrupted benchmark runs save each edited case atomically under
`methods/<method>/case-checkpoints/<identity hash>/`. Each checkpoint includes the
execution record, all requested captures, and a content hash. Rerunning the same
plan reuses complete cases only when the execution configuration, capture
configuration, baseline inputs, and case content match. Failed or invalid cases
are retried; `force=true` recomputes every case. Final aggregate artifacts are
written when the full batch finishes. Checkpoints remain available if PBS kills
a job before that point.

## Runtime optimization (2026-10-26)

The earlier implementation evaluated 98 angles per generator by building a
full dense peeled matrix and solving a dense linear system on every angle.
With three active generators this was 294 dense candidate evaluations per
layer per edit, in addition to the generalized eigendecompositions, one full
Dyson SVD and four BCV SVDs. The original GRAM localizer did substantially less
work. Existing PBS logs show GPT-2 large finished 100 edits in 5:06:51, whereas
DeepSeek needed 8:06–12:21 for 25 edits depending on GPU hardware.

For a plane P and H=(A+ridge I)^-1, both the direction u and constrained mixer
live in span(P, H P, H B P). The angle search now uses precomputed matrices of
at most 6x6 with the same grid and refinement count. Its three finalists are
reconstructed and scored with the original dense formula. Candidate search
rounding can change the selected direction; MDL code lengths and the binary
decision rule remain unchanged.

Dyson and BCV diagnostics are memoized with SHA256 hashes of the actual weight
bytes, shape, dtype, device and compute flags. The bounded cache accepts no
editing-layer hint or experiment baseline. A changed matrix is recomputed.
Rice-code quotient sums use a bounded histogram and an exact recurrence;
triangular indices and source lattices are reused, and the rolling Gram cache
covers the full local predictor window. No detection threshold was added.

Evidence now records wall-clock stage timings and diagnostic cache hits. The
compute implementation is part of capture configuration hashes, so old
checkpoints and captures do not silently satisfy the new capture request.
New unit/numerical tests remain deferred. The optimized PBS run used only
ROME, N=10 per model, 2–6 hour limits, and precomputed ROME covariances; it
completed on all 18 models by 2026-10-29. The late four jobs used 4 CPU,
64 GiB RAM and a 40 GiB GPU minimum. Recorded runtimes and scheduler resource
adjustments are in the [run record](rome-binary-pbs.md). Different GPUs and
batch sizes prevent a controlled end-to-end speedup claim. The poor observed
binary sensitivity remains a separate unresolved research problem.

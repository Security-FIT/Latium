# ROME detector mathematical ablation

Status: **implementation complete; empirical selection blocked on recapture**

Schema: `rome-math-ablation-evaluation-v1`
Threat-model claim status: **no production binary ROME claim**

This report is intentionally incomplete as a result report. The scorer,
versioned evaluator, fixed family split, synthetic invariants, and recapture
job exist, but the required model checkpoints do not exist on this host. No
candidate has been selected and the production weighted-spectrum detector has
not been changed.

## Five-equation experiment

For an editable projection \(W_l\), orient the Gram into the smaller shared
hidden space using tensor shape, never a model or family name:

\[
G_l=\operatorname{hidden\_gram}(W_l)\in\{W_lW_l^\top,\;W_l^\top W_l\}. \tag{1}
\]

The normalized branch uses one scale removal:

\[
C_l=G_l/\max(\operatorname{tr}(G_l),\tau_l), \tag{2}
\]

where \(\tau_l\) is derived from computation dtype, reduction dimension, and
matrix scale. The raw diagnostic branch uses \(G_l\) directly.

For each fractionally trimmed interior layer:

\[
D_l=C_l-\tfrac12(C_{l-1}+C_{l+1}),\qquad
N_l=\tfrac12(C_{l-1}+C_{l+1}). \tag{3}
\]

With \(U_l\) and \(\sigma_i\) from the same top-two SVD, the three simple
candidates are:

\[
\begin{aligned}
M0_l&=\lVert D_l\rVert_F,\\
M1_l&=(\sigma_1(D_l)^2+\sigma_2(D_l)^2)^{1/2},\\
M2_l&=\lVert U_l^\top D_lU_l\rVert_F/
       \max(\lVert U_l^\top N_lU_l\rVert_F,\tau_l).
\end{aligned} \tag{4}
\]

M3 is the unchanged weighted-spectrum control. B0 applies dtype-derived
magnitude and rank-two-tail bounds to the clean-reference Gram delta. B1 uses
the winning simple score and one frozen cutoff; B2 is the current footprint
control:

\[
z_l=(s_l-\operatorname{median}(s))/
     \max(\operatorname{MAD}(s),\tau),\qquad
B1=\left[\max_l z_l>t_{\mathrm{global}}\right]. \tag{5}
\]

Equation (1)'s unnormalized Gram delta has rank at most two after a rank-one
weight update. That is a theorem. The same statement is not made after
Equation (2): trace rescaling can introduce a full-rank term. Top-two
concentration after normalization is therefore a measured hypothesis.

## Predeclared split and selection

The split was written before any M0--M2 real-model scores existed:

| Split | Families |
|---|---|
| Development | GPT-J, GPT-2, Llama, Mistral, Qwen |
| Held-out family | DeepSeek, Falcon, OPT |
| Final frozen gate | restored exact 40-case protocol only |

All cases from a family stay in one selection split. The versioned source of
truth is `manifests/rome_math_ablation_recapture.yaml`.

Candidate comparison and selection will use development cases only, with
leave-one-development-family-out diagnostics, paired cases, and equal family
weighting. The predeclared non-inferiority margin is 2.5 macro-accuracy
percentage points. A simpler candidate is non-inferior when the lower endpoint
of a 95% paired hierarchical bootstrap interval for its development macro
exact-accuracy difference from the best candidate is greater than -0.025.
Development families are resampled as equal clusters and cases are resampled
within family.

Selection order is M0, M1, M2, then M3. M2 must demonstrate a scale failure
avoided relative to M0/M1 in a leave-one-development-family-out fold. M3 must
demonstrate a material cross-development-family gain for every retained
complex feature. DeepSeek, Falcon, and OPT play no role in selection. Once
selected, the formula is frozen. A later regression greater than 10 percentage
points on one of those families is reported as a possible catastrophic
regression and a validation failure; it does not trigger formula reselection.

For B1, one cutoff will be chosen using development families and development
hard negatives only, then frozen. The rule is to choose the smallest cutoff
whose development specificity has a 95% Wilson lower bound of at least 0.95;
among ties, choose the cutoff with greatest sensitivity. If no cutoff meets
that condition, blind binary identification is not validated.

## Recapture inventory

| Model key | Family | Split | Local checkpoint |
|---|---|---|---|
| `gpt-j-6b` | GPT-J | development | missing |
| `gpt2-medium` | GPT-2 | development | missing |
| `gpt2-xl` | GPT-2 | development | incomplete cache |
| `llama2-7b` | Llama | development | missing |
| `mistral-7b-v0.1` | Mistral | development | missing |
| `qwen3-8b` | Qwen | development | missing |
| `deepseek-7b-base` | DeepSeek | held out | missing |
| `falcon-7b` | Falcon | held out | missing |
| `opt-6.7b` | OPT | held out | missing |

The smoke recapture is fixed to cases 0 and 1 on two development families:
`gpt2-medium` and `mistral-7b-v0.1`. It is run twice under distinct run IDs,
then selected layers and scores are compared within the recorded numerical
tolerances. The development recapture requests cases 0--99 only after that
repeatability gate passes.

DeepSeek, Falcon, and OPT are not present in the executable recapture job.
Their manifest entries record only the frozen split and checkpoint audit.
They remain unseen until both the candidate mathematics and global B1 cutoff
are frozen. Because the required development hard negatives are currently
absent, the B1 cutoff cannot yet be calibrated and the held-out release gate
remains closed.

Audit on 2026-07-28:

- `analysis_out` contained 7,932 files and no `.pt`, `.pth`, `.bin`,
  `.safetensors`, or `.ckpt` model tensors;
- the Hugging Face cache contained complete `sshleifer/tiny-gpt2` weights, but
  that model has only two repeated layers and cannot form an interior
  second difference;
- the only other cached model was an incomplete 2.1 GB `gpt2-xl` download;
- several second-moment tensors existed, but second moments are not model
  checkpoints and cannot reconstruct M0--M3;
- the three original 40-case roots were still absent.

The recapture command is:

```bash
jobs/run_rome_math_ablation.sh smoke-a
jobs/run_rome_math_ablation.sh smoke-b
jobs/run_rome_math_ablation.sh development
```

The first two commands must pass and agree before the development command is
run. There is deliberately no held-out or all-model mode in this pre-freeze
job.

## Saved schema and reproducibility

Each capture has schema `rome-math-ablation-capture-v1` and stores:

- source dtype, computation dtype, weight shape, and shape-derived orientation;
- trace and dtype/dimension-derived trace tolerance;
- eligible and excluded layer metadata using a 10% fractional trim;
- normalized M0, M1, M2, and unchanged M3 control scores;
- raw-Gram M0/M1 diagnostics;
- top two singular values, full Frobenius norm, and top-two concentration;
- M3 rank-two energy, bilateral coherence, and bilateral balance controls;
- B0 clean-reference magnitude, tail, and numerical bounds;
- runtime and an explicit peak working-set estimate.

Full Gram matrices are not saved. A three-Gram cache bounds the capture's Gram
working set. The evaluator enumerates even ignored artifacts using
`rg --no-ignore --files`, merges baseline and three-layer patches without
modifying them, and writes a new versioned JSON plus a compact CSV.

The randomized recapture SVD has a fixed per-layer seed. Determinism is judged
within recorded numerical tolerance, not by byte equality of floating-point
outputs.

## Existing local evidence: peak versus footprint

The committed inventory contains 12 paired N=100 run/model variants. A direct
case join, rather than an inference from the aggregate README, found:

| Quantity | Count |
|---|---:|
| Requested paired cases | 1,200 |
| Both analyses complete | 1,162 |
| Peak/footprint verdict disagreements | **0** |
| Peak/footprint selected-layer disagreements | 208 |

Selected-layer disagreements were concentrated in Falcon (84), Qwen (63),
OPT (36), GPT-2 Medium (21), Llama (2), and Mistral v0.3 (2). The remaining
paired variants had none. Thus the morphology path changed many reported
presence peaks without correcting a single available verdict. This is
case-level evidence against retaining morphology, but it is not held-out
hard-negative validation.

The committed ten-case fixture remains marked `scientific_baseline: false`.
It is only a mechanical parity guard and does not contribute to candidate
selection.

## Pending localization results

| Candidate | Development macro exact | Held-out macro exact | Within one | Runtime | Peak memory | Decision |
|---|---:|---:|---:|---:|---:|---|
| M0 trace-normalized Frobenius | pending | pending | pending | pending | pending | pending |
| M1 trace-normalized top two | pending | pending | pending | pending | pending | pending |
| M2 scalar support ratio | pending | pending | pending | pending | pending | pending |
| M3 current control | pending | pending | pending | pending | pending | control |
| M0 raw Gram diagnostic | pending | pending | pending | pending | pending | diagnostic |
| M1 raw Gram diagnostic | pending | pending | pending | pending | pending | diagnostic |

The evaluator will also emit per-family accuracy, unavailable/error counts,
best-versus-second score margins, candidate disagreement counts, and named
failures by target depth. Missing cases remain in denominator accounting and
are never silently dropped from inventory totals.

## Pending M3 feature decisions

Feature decisions are made on cross-development-family evidence and frozen
before the held-out-family release gate opens.

| Feature | Cross-development-family gain | Failure prevented | Keep/remove |
|---|---:|---|---|
| whitening | pending | pending | pending |
| rank-two energy multiplier | pending | pending | pending |
| bilateral coherence | pending | pending | pending |
| bilateral balance | pending | pending | pending |
| morphology product | no verdict gain in 1,162 available paired cases; development recapture pending | none observed locally | pending development decision |
| `log1p` transform | pending | pending | pending |

## Binary threat models and validation status

| Proposal | Threat model | Boolean available | Validated claim |
|---|---|---|---|
| B0 | clean reference plus suspect | yes | no; only “ROME-compatible low-rank edit” |
| B1 | suspect only | yes after supplying one global cutoff | no; hard negatives absent |
| B2 | suspect only, current control | yes | no; current universal bound is not empirically calibrated |

No adequate independent clean or hard-negative corpus exists locally. In
particular, there are no non-ROME low-rank edits, other editing methods,
ordinary fine-tunes, quantized/merged checkpoints, dtype-only conversions, or
pruned/compressed checkpoints. Sensitivity without these negatives cannot
support a binary forensic attribution claim.

## Stop decision

Production replacement is stopped. The blocking evidence is not a software
failure: the required checkpoint tensors, frozen 40-case manifests, and
binary hard negatives are absent. The experimental scorer, invariant tests,
versioned evaluator, fixed family split, and reproducible recapture job are
ready. The next authorized step is to run the smoke recapture on a host with
the declared checkpoints, confirm deterministic output, and only then run the
development-only recapture. DeepSeek, Falcon, and OPT remain unopened until
the formula and global B1 cutoff have been frozen.

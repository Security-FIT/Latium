# ROME mathematical detector: nine-model N=50 development iteration

## Status and claim boundary

The run completed all 450 requested CounterFact cases (50 per model). All
structural captures completed; 435 ROME edits met the edit-success criterion.
The current M3 control was the only candidate that passed the predeclared
localization non-inferiority rule. Production detector mathematics was not
changed during this evidence-generating run.

The subsequent evidence-preserving consolidation retained only the M3
localizer and clean-reference B0 rule: commit `af2b770` froze the compact
golden evidence, `420bcd8` consolidated the production detector, and
`c0f4222` removed the completed ablation runtime. Full raw evidence remains
recoverable from Git at commit `693a949`.

This is not a scientific baseline or a held-out evaluation. All nine models,
including the formerly held-out DeepSeek, Falcon, and OPT families, were
exposed as development data. The run cannot establish blind specificity,
calibrate B1, attribute a low-rank footprint uniquely to ROME, or reproduce
the historical 38/40 result.

## Execution provenance

- Host: `150-136-40-217` (`150-136-40-217`)
- GPU: NVIDIA A100-SXM4-40GB
- Execution backend: persistent `tmux`, direct on a fresh dedicated GPU host
- Scheduler: unavailable (`qsub` was not installed), so there are no PBS job
  IDs; the terminal-stage ledger records
  `direct:150-136-40-217:complete`
- Main checkout commit used for dependency recovery:
  `29ea56819d0f1f9f2a51fc41bb0f76292aa92ada`
- Detector checkout commit used for smoke and N=50:
  `e1ab618`
- Capture profile: `rome-math-ablation`
- Edit methods: `rome` only
- Normal analysis preset: disabled
- Rendering: disabled; no graph directories were produced

The lack of graph artifacts is intentional. Graph rendering is a visualization
stage and is not part of M0-M3 detection. The detector capture, localization,
and B0 evaluation all ran.

## Dependency audit

Exact local second moments were copied and checksum-verified. No covariance was
renamed or used for a different layer. Four dependencies had already completed
the trace/recompute path before the instruction to stop matrix calculation;
the other five used exact transferred matrices. No additional matrices were
calculated during the N=50 continuation.

| Model | Revision | Layer | Dependency | Shape | SHA-256 |
|---|---|---:|---|---:|---|
| gpt2-xl | `15ea56dee5df4983c59b2538573817e1667135e2` | 16 | exact local transfer | 6400² | `5c55443efb8a2596163db5e862ac2e5f81907aa14a5e80c0bea3a2e4aacefb9b` |
| gpt-j-6b | `47e169305d2e8376be1d31e765533382721b2cc1` | 5 | trace-confirmed recompute | 16384² | `9f9982efe5dd33093f9ad0d5e0371aa9e6d8eaeca15a045e3104d20e894faa27` |
| mistral-7b-v0.1 | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | 5 | trace-confirmed recompute | 14336² | `59a65c8713595f63ef3d46d8df6f53374f6437a79683ee561ee9fdd5787d194d` |
| mistral-7b-v0.3 | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | 6 | exact local transfer | 14336² | `b04f75f5d08ae86d45ced2e25d62681927af21a39225de3c679d447f450ce82c` |
| deepseek-7b-base | `7683fea62db869066ddaff6a41d032262c490d4f` | 5 | exact local transfer | 11008² | `6b5bd52eab30964e9b6303028f11d3cb77e2f8574dbaf63be7c9cbaa5dbc26e0` |
| falcon-7b | `ec89142b67d748a1865ea4451372db8313ada0d8` | 5 | trace-confirmed recompute | 18176² | `78e89bc08487da87124a69e7025ad54cf6fb3eb2ab3ff6748f86be785236845a` |
| opt-6.7b | `a45aa65bbeb77c1558bc99bedc6779195462dab0` | 14 | trace-confirmed recompute | 16384² | `7d92d2631bca823d1ef35d6bb4daf5baefbe58acc74917f6e6bae365b38a7116` |
| llama2-7b | `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` | 5 | exact local transfer | 11008² | `f177d57e09e857657528f8bcf0fe2b82a54d0867a1964372ea93b709264b383e` |
| granite4-micro | `56111ae135df9c53a78c99028e7bc24035a9e979` | 10 | exact local transfer | 8192² | `8d8df9d6022b6fda3ae3b77476becfc9c27cb99dc93176d6384fa01660b02b34` |

All matrices are float32, use the Wikipedia method, and record 100,000
samples. Full dependency paths, trace summaries, confirmation intervals, and
timestamps are in
`manifests/rome_math_n50_causal_dependencies.json`.

## Smoke repeatability

Cases 0-1 were run twice on gpt2-xl and Mistral-7B-v0.1. All four paired cases
completed in both runs. Across 16 M0-M3 score comparisons:

- selected layers agreed exactly;
- maximum absolute score difference was `0.0`;
- every comparison was inside its recorded numerical tolerance;
- B0 was true for every successful smoke edit;
- B1 remained `not_evaluated_uncalibrated`;
- no normal detector or graph-rendering artifacts were created.

## ROME execution

| Model | Requested | Capture complete | Successful ROME edits |
|---|---:|---:|---:|
| gpt2-xl | 50 | 50 | 48 |
| gpt-j-6b | 50 | 50 | 49 |
| mistral-7b-v0.1 | 50 | 50 | 50 |
| mistral-7b-v0.3 | 50 | 50 | 46 |
| deepseek-7b-base | 50 | 50 | 49 |
| falcon-7b | 50 | 50 | 49 |
| opt-6.7b | 50 | 50 | 50 |
| llama2-7b | 50 | 50 | 49 |
| granite4-micro | 50 | 50 | 45 |
| **Total** | **450** | **450** | **435** |

No unavailable or detector-error cases were removed. Metrics below retain all
450 requested cases, and successful-edit-only metrics use the 435 explicitly
successful ROME edits.

## M0-M3 localization

| Candidate | Correct / 450 | Exact | Within one | Correct / 435 successful | Successful-edit exact | Equal-family macro |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 174/450 | 38.7% | 38.7% | 166/435 | 38.2% | 42.4% |
| M1 | 222/450 | 49.3% | 60.4% | 212/435 | 48.7% | 46.3% |
| M2 | 314/450 | 69.8% | 77.6% | 304/435 | 69.9% | 70.5% |
| M3 | 386/450 | 85.8% | 89.1% | 375/435 | 86.2% | 85.9% |

### Exact localization by model (all 50 requested)

| Model | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| gpt2-xl | 28 | 2 | 1 | 47 |
| gpt-j-6b | 0 | 0 | 50 | 49 |
| mistral-7b-v0.1 | 0 | 34 | 48 | 46 |
| mistral-7b-v0.3 | 9 | 40 | 16 | 39 |
| deepseek-7b-base | 50 | 50 | 50 | 50 |
| falcon-7b | 0 | 0 | 0 | 9 |
| opt-6.7b | 0 | 0 | 49 | 50 |
| llama2-7b | 37 | 46 | 50 | 46 |
| granite4-micro | 50 | 50 | 50 | 50 |

The failures reveal stable architecture-level confounders in the simple
scores. M0 selects Falcon layer 14 in all cases, GPT-J layer 2 in all cases,
Mistral-v0.1 layer 13 in all cases, and OPT layer 3 in all cases. M2 improves
substantially but selects Falcon layer 13 in all cases and GPT-2 XL layer 42 in
49/50 cases. M3 also has a serious Falcon failure: it selects layer 9 in 41/50
cases instead of target layer 5.

### Provisional selection

Selection used equal family weighting, a paired hierarchical
family-then-case bootstrap (10,000 iterations, seed 20260728), a 95% interval,
and a 2.5 percentage-point non-inferiority margin. Candidate order was fixed
as M0, M1, M2, M3.

| Candidate | Macro accuracy | Difference from M3 | Paired 95% interval | Non-inferior |
|---|---:|---:|---:|---|
| M0 | 42.4% | -43.5 pp | [-71.5, -17.8] pp | no |
| M1 | 46.3% | -39.6 pp | [-72.7, -12.3] pp | no |
| M2 | 70.5% | -15.4 pp | [-39.5, +0.8] pp | no |
| M3 | 85.9% | 0.0 pp | [0.0, 0.0] pp | yes |

The development selection is therefore **M3**. This is evidence against
replacing the localizer with M0-M2 as defined in the experiment. The
production consolidation retained the score actually selected by M3:
normalized hidden Gram, two-neighbor residual, top-two subspace, 2x2
neighbor-support whitening, and its Frobenius norm. It did not retain the
rank-two multiplier, bilateral coherence/balance, morphology, `log1p`, or
blind-presence rules because those quantities were not part of the winning
localization score. No model-specific correction was added for Falcon.

## B0 clean-reference result

B0 returned `ROME-compatible low-rank edit` for 434 of 435 successful ROME
edits: sensitivity 99.77%. The only false negative was Llama-2 case 26 at the
correct target/selected layer 5.

| Model | B0 true / successful |
|---|---:|
| gpt2-xl | 48/48 |
| gpt-j-6b | 49/49 |
| mistral-7b-v0.1 | 50/50 |
| mistral-7b-v0.3 | 46/46 |
| deepseek-7b-base | 49/49 |
| falcon-7b | 49/49 |
| opt-6.7b | 50/50 |
| llama2-7b | 48/49 |
| granite4-micro | 45/45 |

This is positive sensitivity only. With no independent clean checkpoints,
non-ROME rank-one edits, other editing methods, fine-tunes, quantized or merged
checkpoints in this corpus, no specificity or ROME-attribution claim is
supported.

B1 was not calibrated and is recorded as
`not_evaluated_uncalibrated` in all 450 historical outputs. No fake blind
cutoff was introduced. B2 was only an experimental control; neither B1 nor B2
is part of the minimal production API.

## Runtime and memory

The isolated M0-M3/B0 capture took 389.3 aggregate detector seconds across 450
cases. Per-model aggregate detector time ranged from 10.3 seconds (gpt2-xl) to
67.6 seconds (Falcon). The maximum estimated peak working memory was
578,142,208 bytes (about 551 MiB), observed for Falcon. These measurements
exclude model loading and ROME editing.

## Preserved evidence and next scientific gate

- Compact 450-case golden evidence:
  `tests/fixtures/rome_detector_n50_golden.json`
- Golden integrity and aggregate assertions:
  `tests/test_rome_detector_n50_golden.py`
- Covariance and causal provenance:
  `manifests/rome_math_n50_causal_dependencies.json`
- Full raw captures, evaluation outputs, and execution logs:
  Git commit `693a949`

The next binary gate needs independent clean checkpoints and hard negatives,
especially non-ROME low-rank edits and another knowledge-editing method. A
final localization claim also needs new unseen model families/checkpoints.
The historical 40-case roots must be restored or exactly rerun before claiming
that any future simplification preserves the reported 38/40 result.

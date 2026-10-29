# ROME binary detection: implementation and PBS record

Last verified: **2026-10-29T13:27:02.925866+01:00**. Branch: `ROME-reversal`.

## Current result

**All 18 models completed N=10**, with 180 complete edited-state captures and analyses, plus 18 clean baselines.
Completion is an execution result, not evidence of adequate detection quality. The ROME evaluator recorded 179 successful edits out of 180 attempts.
**The binary detector returned positive for only 1/180 edited checkpoints, or 1/179 successful edits (about 0.56%).**
The positive was Gemma 4 12B, case `8091`, detected layer `11`, with `mdl_delta_bits=6664209`.
All 18 clean baselines returned negative. That small control set does not establish a general false-positive bound.
**The current MDL detector therefore has inadequate observed sensitivity and is still experimental.** No threshold was fitted to improve this result.

Machine-readable records:
- [Per-model and per-case N=10 results](experiments/rome-binary-n10-results-20261029.yaml)
- [Every submission period, job ID and resource adjustment](experiments/rome-binary-pbs-history-20261029.yaml)

## What changed

- Added a versioned single-checkpoint Boolean MDL detector beside ordinary GRAM and reversal. V1 remains replayable; v2 is the default.
- Added adaptive paid precision, competing depth backgrounds, generalized spectra, Dyson/LSS diagnostics, raw-matrix BCV and an optional all-layer covariance view.
- The detector scans supplied contiguous layers, including endpoints, without accepting a configured editing layer or CT result. The benchmark still uses its layer configuration to apply ROME and evaluate localization.
- Added CLI/Hydra presets, capture/analysis registry integration, configuration validation and evidence replay.
- Added atomic per-case checkpoints with configuration, baseline-input and case-content identity; interrupted runs can resume complete matching cases.
- Replaced dense angle-by-angle peeling with at-most-6D search and dense finalist scoring; memoized pure research diagnostics by actual weight content; accelerated Rice lengths and reused triangular indices/source lattices.
- Added stage timings and compute-implementation identity so old captures/checkpoints cannot silently satisfy the changed capture implementation.
- Preserved reversal defaults. `best_peel(plane_scale=...)` can change the candidate plane; its default remains the legacy behavior.

The mathematical derivation and limitations are in [the research note](rome-binary-research.md),
the [initial plan](rome-binary-plan.md), [v1 reference](gram-binary.md) and [v2 implementation](gram-binary-v2.md).

## Deployment chronology

| Date | Period | Outcome |
|---|---|---|
| Oct 24 | 18 v1 jobs, 100 edits/model | Cancelled for the v2 replacement; outputs retained. |
| Oct 24–25 | 18 v2 jobs, 2–4h limits | All hit walltime. Baselines survived; edited captures were held in memory until the full batch and were lost on timeout. |
| Oct 25 | 33 replacement jobs, 4–48h limits | Cancelled while all were queued. These limits were too high for the user's requested workflow. |
| Oct 25–26 | 59 disjoint batches, each at most 24h, total 100/model | 12 completed with 575 persisted edited cases; remaining 47 queued jobs cancelled for N=10. |
| Oct 26–29 | Optimized v2, 18 GPU jobs, N=10/model, 2–6h limits | All 18 complete. A five-minute CPU job checked the ten pilot captures before releasing other models. |

The user policy is **at most 24 hours per job**. The server advertising a higher queue maximum is not authorization to request a longer walltime.
Cancelled snapshots and outputs were preserved; the late four-job fixes used `qalter`, keeping IDs and the original `qtime`.

## Resource corrections and measured usage

The initial 8 CPU / 96 GiB RAM / 80 GiB GPU request was overly restrictive for several models.
Scheduler inspection used `qstat -sw`, `qstat -f`, `qstat -T`, `pbsnodes -av -F json` and `pbs-get-job-history` for records older than 24h.
The reported first missing resource changed from `gpu_mem` to `queue_list` or `ncpus`; these comments do not guarantee an immediate start once one constraint is relaxed.

- Oct 28: GPU minimum 80→40 GiB for all nine still-queued 4–8B jobs, based on comparable 7–8B peak usage around 20–23 GiB.
- Oct 28: CPU request 8→4 for OPT and the three Qwen jobs; comparable completed runs averaged about one CPU.
- Oct 28: host RAM 96→64 GiB for those same four jobs, with observed completed-run peaks up to approximately 33 GiB including the larger Gemma 12B model.
- Their final common request was 4 CPU, 64 GiB host RAM, 1 GPU with at least 40 GiB VRAM, 50 GiB local scratch and a four-hour walltime.

| Model | PBS ID | Complete cases | ROME successes | Binary positive | Recorded walltime |
|---|---|---:|---:|---:|---|
| gpt2-medium | 24988653.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 00:11:40 |
| deepseek-7b-base | 24988655.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| falcon-7b | 24988656.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| gemma-4-12b | 24988657.pbs-m1.metacentrum.cz | 10 | 10 | 1 | expired from qstat history |
| gpt-j-6b | 24988658.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| gpt2-large | 24988659.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| gpt2-xl | 24988660.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| granite-4.1-8b | 24988661.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| granite4-micro | 24988662.pbs-m1.metacentrum.cz | 10 | 10 | 0 | expired from qstat history |
| llama2-7b | 24988663.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 02:45:27 |
| ministral-3-8b | 24988664.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 02:37:10 |
| mistral-7b-v0.1 | 24988665.pbs-m1.metacentrum.cz | 10 | 9 | 0 | 02:09:19 |
| mistral-7b-v0.3 | 24988666.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 02:06:59 |
| olmo-3-1025-7b | 24988667.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 02:14:28 |
| opt-6.7b | 24988668.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 03:35:18 |
| qwen3-4b | 24988669.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 01:06:16 |
| qwen3-8b | 24988670.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 02:46:35 |
| qwen3.5-4b | 24988671.pbs-m1.metacentrum.cz | 10 | 10 | 0 | 00:53:39 |

For comparison, the earlier full v2 implementation ran GPT-2 large N=100 in 5:06:51,
while DeepSeek N=25 took 8:06–12:21 depending on GPU hardware. The optimized DeepSeek N=10 run took 2:28:42,
Gemma N=10 took 2:49:11 and Granite 8B N=10 took 2:21:27 according to archived PBS records.
These different batch sizes and GPUs are not a controlled end-to-end speedup measurement.

## Source and artifact provenance

The optimized deployment was exported from base commit `54a829e1e8d997fc431344a771a3d5cea250da45` plus the then-uncommitted branch changes.
It contained 315 source files. Its tarball SHA256 was `af634b7ecc8d11d5aadb1db059b909747abb0848adf7a07243eb2f3b25e9e8e3`.
The deployed source files passed a later hash audit. New documentation in this commit was written after that deployment.

Cluster snapshots under the account's shared home:
```text
Latium-gram-binary-20261024
Latium-gram-binary-v2-20261024
Latium-gram-binary-v2-rerun-20261025/
  relaunch-24h/
  analysis_out_24h/
Latium-gram-binary-v2-n10-20261026/
  source-files.sha256.json
  submitted-jobs.json
  pilot-gate-job.json
  analysis_out/<model>/run/
  push-status-20261029.json
  binary-outcomes-20261029.json
```

Local working records and helper scripts are under ignored `analysis_out/gram-binary-*` directories.
This tracked note and the two YAML records preserve their relevant operations and results. Large captures, model weights,
archives, transient SSH helpers and credential-bearing `jobs/local.env` are not source artifacts.
Each full run keeps its artifact manifest, baseline/method captures, analysis results and atomic `case-checkpoints/` files.

## Reproduce a small run

Use the existing shared model, CounterFact, prefix and ROME covariance caches. The optional detector covariance view
remained disabled because the available ROME cache covers only editing layers, not every scanned layer.
Missing ROME covariance must fail rather than trigger computation.

```bash
bash jobs/submit.sh custom --queue gpu --name mdl10-opt \
  --ncpus 4 --mem 64gb --gpu-mem 40gb --gpu-cap compute_80 \
  --scratch 50gb --walltime 04:00:00 -- \
  structural run structural=gram-binary \
  'structural.run.models=[opt-6.7b]' 'structural.run.edit_methods=[rome]' \
  structural.run.n_tests=10 structural.run.start_idx=0 \
  structural.run.case_index_file=manifests/counterfact_seed42_n1000.json \
  structural.run.output_dir=analysis_out/rome-binary-n10-opt \
  structural.run.run_id=run structural.run.fail_on_missing_second_moment=true \
  runtime.second_moment_allow_autocompute=false structural.render.enabled=false \
  structural.capture.gram_binary.method=v2 \
  structural.capture.gram_binary.dyson=true structural.capture.gram_binary.bcv=true \
  structural.capture.gram_binary.covariance_manifest=null
```

This is a new-run example, not an instruction to rerun the completed fleet. Choose resources from actual model requirements;
Gemma 12B was not tested on a 40 GiB card. Never automatically extend beyond the 24h policy.

## Validation state

Static Ruff checks, whitespace checks, configuration/import preflights and production artifact-completion checks passed.
The earlier v1 synthetic/integration checks were run before the user deferred further tests.
**The v2 unit/numerical test suite and optimization-equivalence tests have not been executed.** No new tests were run to prepare this push.
Production N=10 execution and result summaries are kept separate from that deferred validation.
Remaining research work includes explaining the poor sensitivity, checking optimization equivalence and evaluating independent controls.

Official operational references:
- [GPU resources and usage metrics](https://docs.metacentrum.cz/en/docs/computing/gpu-comput/gpu-job)
- [Job history](https://docs.metacentrum.cz/en/docs/computing/jobs/finished-jobs)
- [PBS attributes and qalter](https://docs.metacentrum.cz/en/docs/computing/advanced)

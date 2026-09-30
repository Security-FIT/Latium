# Gram workflow validation — 2026-09-30

Branch: `simplify-code`, worktree `/home/metju/Latium-ccs-simplify`.
Core fixes and model settings were integrated from `exp/rome-layer-fleet` at
`3feeb33`. On 2026-10-01, history was rewritten to exclude the optional Gram
experiments, their configurations, registrations, reports and tests. The
normal Gram detector is byte-for-byte identical to its prior `simplify-code`
version. The original validation below records evidence gathered before that
cleanup; removed experimental tests are not part of the current suite.

## Automated checks

**175 distinct tests passed** across the targeted validation suites, using
the existing MetaCentrum `latium` Python environment on an isolated copy:

- Registry, jobs/PBS, ROME evaluation, graph renderers, structural planning,
  per-case editing/restoration, CCS report and W&B tests: 101 existing tests.
- Detector, saved Gram regression, analysis runtime, artifact writer,
  relative graph renderers, CLI smoke and model configs: 65 tests.
- New shared-cohort/range/append/report/fleet gates: 9 tests.

The new gates cover deterministic 1000-fact sampling, disjoint `[0:100]`
and `[100:200]`, dataset/case content mismatch, identical model selections,
Gram-only dependency resolution with plots enabled, overlap/config rejection,
100 → 200 unique cases, unchanged first-batch artifact bytes, graph input
invalidation and missing-file repair, baseline/error denominators, harmonic
ROME score aggregation, and complete-batch retries with zero additional edits.

The detector reproduces the full localization objects for **600/600** saved
cases from the measured 12-model runs. This is parity, not 100% exact accuracy.
Basic Gram accuracy on those saved cases remains 453/600.

`ruff check` passed on source, jobs, generator and the new tests.
`bash -n` passed for the PBS fleet submit script; `git diff --check` passed.

## Frozen facts

`manifests/counterfact_seed42_n1000.json` contains 1000 unique row indices and
CounterFact case IDs, seed 42, train split of `azhx/counterfact`.
Dataset revision: `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`.
Manifest hash: `7ce8d8044134b28347a92ff0d078c6d1ae00748b6450f2f42b7f81604c18c8a9`.
The manifest also freezes dataset fingerprint and selected case content hashes.

## GPU smoke

Isolated remote source staging:
`/storage/brno2/home/olexamatej/Latium-gram-smoke-20260930`.
Experiment root: `analysis_out/smoke` beneath that directory.

Two configured models, GPT-2 XL and Granite 4 Micro, use the same frozen cohort
in batches `[0:3]` and `[3:6]`, reusing existing covariance. No causal tracing,
CCS or experimental Gram capture is selected.

First batch finished for both models, all 3 cases complete/evaluated each.
GPT-2 XL: Gram exact 3/3, detected layer 18; Granite: 0/3, detected layer 6
with target layer 12. ROME success 3/3 each. These are small smoke samples,
not new benchmark accuracy estimates.

Both batches finished for both models: **6 unique selected/completed/evaluated
facts per model**, the same IDs and positions 0..5. Only baseline/ROME execution,
Gram captures and Gram analyses were produced; the Gram report was rendered.

| Model | ROME success | Gram exact | ROME target | Gram detected layers |
|---|---:|---:|---:|---|
| GPT-2 XL | 6/6 | 6/6 | 18 | 18 × 6 |
| Granite 4 Micro | 6/6 | 2/6 | 12 | 6 × 4, 12 × 2 |

All **12 first-batch artifact files** kept identical SHA-256 hashes after append.
Retrying both batches on the frontend took about 13 seconds, issued **zero new
model commands**, and kept all **24 plan artifacts** byte-identical. The report
contains 12 unique `(model, case_id)` rows, with common evaluated coverage 6.
JSON/CSV reports and PNG graphs are generated and the plots were visually checked.

Local evidence is retained in `.tmp/gram-smoke/validation.json` and
`.tmp/gram-smoke/report/` in this worktree. PBS jobs were
`24125907`, `24125908` (first batch), `24126046`, `24126047` (append), all complete.

The only GPU work submitted is this small two-model smoke, not a 100/1000-fact
fleet. Resume is at batch/artifact granularity; partial per-case edits are not
checkpointed. The full repository test suite was not run.

## Complexity review and cleanup

After the GPU smoke, the append guard was narrowed from all `src`/`jobs` files
to computation code plus each selected model's config. Plotting/PBS changes
and adding an unrelated model config no longer force a new experiment root.
The manifest is loaded once when planning instead of once per model/run.
Unused imports were removed. Detector/edit computations were unchanged.

**88 targeted tests passed after this cleanup**, including a regression that
accepts reporting changes and rejects computation changes. Lint, shell syntax
and diff whitespace checks also passed. The GPU smoke above predates this
small guard/planning cleanup; no additional GPU run was needed.

See [the design review](gram-design-review.md) for scope and commit grouping.

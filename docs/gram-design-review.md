# Gram workflow: complexity review

Reviewed on `simplify-code`, 2026-09-30.

## Verdict

The new workflow is proportionate to the requested fleet, shared 1000-fact
cohort, range append, resume and cumulative graphs. One broad compatibility
check was unnecessary and has been narrowed. The implementation uses existing
ROME execution/restoration, capture/analysis registries, artifact manifests,
writer locks and graph input lineage. It adds no scheduler, database, custom
checkpoint engine, plugin framework or new ROME layer override API.

## Simplifications made

- **Compatibility checks:** hash computation code and each selected model's
  configuration. The original whole-tree check also rejected plotting/PBS
  changes and adding unrelated model YAMLs, which would obstruct normal work.
  The new regression accepts these changes and rejects computation changes.
- **Planning:** read the fact manifest once, then use it for every model/run.
- **Cleanup:** remove unused imports; retain small procedural helpers.

## Why the remaining pieces exist

| Piece | Requirement it serves |
|---|---|
| Fixed cohort plus revision/content checks | Same random facts and ordering across models and future batches. |
| Existing plan/artifact layout | Append without replacing the first batch or duplicating retries. |
| One experiment catalog | Expected model/fact coverage, including cases whose workers have not produced results yet. |
| Per-batch state | Reuse completed ranges and distinguish another equal-sized range. |
| Catalog and batch locks | Parallel PBS workers and duplicate submissions cannot race on shared state. |
| One artifact-only report reader/renderer | Cumulative per-case statistics and graphs without rerunning models. |

No per-case resume mechanism is added. Interrupted batches use the existing
artifact cache and otherwise retry at batch granularity. Reports are rebuilt
from saved cases; there is no separate incremental aggregation/cache engine.
Overlapping different ranges are rejected instead of introducing retry
resolution rules.

## Size and preserved source

The Gram feature commit adds **538 net lines of production code/configuration**,
plus 217 test lines and the fixed manifest (3016 JSON lines). Its large apparent
diff is mostly that cohort data. Separately, the first commit preserves the
measured model settings, runtime fixes, fleet support and tracking. Optional
Gram experiments were removed from that commit during the history cleanup on
2026-10-01. The normal Gram detector remains identical to its prior
`simplify-code` implementation; experimental registrations, configurations,
implementations, reports and tests are absent.

## Commit grouping

1. Core fixes — measured MetaCentrum model settings, runtime fixes, fleet
   support and tracking from `3feeb33`, with optional Gram experiments excluded.
2. Gram workflow — minimal Gram preset, shared cohort/ranges, fleet append/resume,
   cumulative reports and regression tests.
3. Documentation — usage guide, validation evidence and this complexity review.

## Evidence

88 targeted tests passed after cleanup, including disjoint ranges, 100 → 200
append, unchanged first artifacts, zero duplicate edits on resume, graph
invalidation, denominator handling and the narrower compatibility check.
Prior evidence includes 600/600 saved-result parity and a real two-model
3 + 3 GPU smoke. Details are in [the validation report](gram-validation.md).

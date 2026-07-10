# Current Handoff

## Goal

- Restore/verify structural graph rendering for Qwen3-8B runs.
- Ensure `matrix-features` captures are available and complete enough for composite/blind detector graphs.
- Run a small Qwen3-8B structural job on the cluster and copy outputs locally.

## What Was Done

- Ported/verified legacy-style structural graph renderers earlier in this branch.
- Fixed `matrix-features` capture so edited-state captures include all layers, not only changed layers.
- Diagnosed failed Qwen3-8B runs where `matrix-features` was missing or unusable.
- Ran fixed N=1 Qwen3-8B job on cluster.
- Rendered graph bundle manually after capture/analysis completed.
- Copied completed run back locally.

## Successful Run

- Remote run: `/home/ubuntu/Latium/analysis_out/20260627_001042`
- Local copy: `/home/metju/Latium/analysis_out/20260627_001042`
- Graph files rendered: 50
- Key detector output:
  - `analysis_out/20260627_001042/graphs/structural-detector-layer-window/final_n500_detector_layer_window_radius1.json`
  - target layer: `10`
  - detected layer: `10`
  - strict accuracy: `1.0`
  - +/-1 window accuracy: `1.0`
- Verified method `matrix-features` capture contains all 36 projection layers: `0..35`.

## Issues Found

- Old run `analysis_out/qwen3-8b-fast` had `matrix-features` explicitly disabled, so composite/blind analyses were unavailable.
- Earlier fixed run had edit-side `matrix-features` containing only changed layer `10`, causing graph detector output to be empty.
- Full-layer `matrix-features` on Qwen3-8B initially died before writing manifest records.
- This was not a manifest deadlock:
  - no structural process remained;
  - `.manifest.lock` was zero bytes and old, consistent with a normal lock file;
  - no Python traceback was written;
  - process disappeared during expensive baseline `matrix-features` work.
- Likely failure mode was native/library crash or resource kill during full SVD / full PCS work.
- Graph rendering did not run automatically in the successful N=1 structural command; `render` was `null`, so graphs were rendered with a separate `graphs run` command.

## Uncommitted Code Changes Made

### `src/structural/capture/producers.py`

- `capture_matrix_features` now calls:
  - `BlindMSDDetector().compute_layer_features(weights, top_k=context.options["spectral_top_k"])`
- It writes the returned blind/profile feature dict directly instead of recomputing `matrix_profile(weights[layer])` for each layer.
- Purpose:
  - avoid duplicate SVD/profile work;
  - keep all per-layer profile keys available for analyses/graphs;
  - use the same `spectral_top_k` budget as spectral capture.

### `src/structural/detectors/blind_resident.py`

- Removed full SVD path from `compute_layer_features`.
- Replaced `gpu_svd(...)` / full singular vectors with `gpu_svd_topk(..., k=top_k, niter=4)`.
- Added `top_k: int = 50` argument to `compute_layer_features`.
- Replaced `src.rome.common.pcs(W)`, which materializes a full row similarity matrix, with an equivalent row-normalized sum formula:
  - normalize rows;
  - sum normalized rows;
  - dot the sum vector with itself;
  - divide by the same PCS denominator.
- `compute_layer_features` now returns `**profile` plus `pcs` and `row_alignment`, instead of only a subset of profile keys.

## Remote Cluster State

- The same optimized files were copied to the cluster repo at `~/Latium` and compiled there.
- Successful N=1 run used those uncommitted remote changes.
- Remote worktree is expected to be dirty in the same two files.

## Current Local Git Status

Uncommitted modified files from the Qwen3-8B fix:

- `src/structural/capture/producers.py`
- `src/structural/detectors/blind_resident.py`

Unresolved merge-conflict files currently present:

- `src/graphs/registry.py`
- `src/graphs/runtime.py`
- `tests/test_graph_renderers.py`

Important: I did not resolve or modify the conflicted graph files in this last step. They appear as `UU` in `git status` and should be handled separately.

## Verification Performed

- Local compile check passed:
  - `python3 -m compileall src/structural/detectors/blind_resident.py src/structural/capture/producers.py`
- Remote compile check passed with cluster venv:
  - `.venv/bin/python -m compileall src/structural/detectors/blind_resident.py src/structural/capture/producers.py`
- Remote N=1 capture/analysis completed successfully:
  - baseline execution complete
  - baseline spectral capture complete
  - baseline matrix-features capture complete
  - ROME execution complete
  - method spectral capture complete
  - method matrix-features capture complete
  - composite analysis complete
  - spectral analysis complete
  - blind analysis complete
  - `gpt-norm-cv` unavailable as expected for Qwen3-8B
- Graph render completed separately:
  - `MPLCONFIGDIR=/tmp/matplotlib_cache .venv/bin/python -m src graphs run analysis_out/20260627_001042 graphs.renderer_preset=structural-full graphs.force=true`

## Next Steps

- Resolve the three graph merge conflicts before committing anything.
- Decide whether the top-k approximation for blind `matrix-features` is acceptable as the permanent Qwen3-8B behavior.
- If acceptable, commit the two matrix-feature performance fixes after conflict resolution.
- If exact full singular values are required, add a separate configurable capture mode rather than using full SVD by default for large models.

## Review of Bad Commit `c8f21558d8ff12efa370931212795d069c38dfe9`

Commit reviewed: `c8f21558d8ff12efa370931212795d069c38dfe9` / `feat: port legacy graph renderers to structural artifact pipeline`.

High-level verdict: this commit made graphs render, but it is not a clean port onto the refactored artifact pipeline. It copied old bundle-era assumptions into new code, built a synthetic legacy payload inside renderer code, and mixed graph rendering, artifact adaptation, detector replay, bundle export, and legacy script archival into one commit. It should not be treated as a good architectural foundation.

Scale of the commit:

- 22 files changed.
- Around 7,069 inserted lines.
- Added `src/graphs/structural_renderers.py` with 1,282 lines in one module.
- Added around 5,200+ lines of copied legacy scripts under `graphs/scripts/` and `graphs/render_artifacts/`.
- Added new graph runtime behavior, registry entries, command aliases, config, tests, and copied scripts all at once.

### Core Architectural Problems

- The renderer layer constructs a fake old-style `metadata` plus `tests` structural payload from executions, captures, and analyses. That compatibility adapter belongs in a small typed materialization layer, not inside `src/graphs/structural_renderers.py`.
- Renderers do not declare required inputs. They search `context["executions"]`, `context["captures"]`, and `context["analyses"]` manually by producer strings. This bypasses the repo contract that components should declare dependencies instead of adding orchestration/search branches.
- Missing inputs are not represented as `unavailable` with a reason. Most functions silently produce empty data, empty summaries, or panels saying `No data`.
- `render_run` passes all execution, capture, and analysis artifacts to every renderer. Every renderer has the same input dependency list, so unrelated artifact changes can invalidate all renders and missing renderer-specific inputs are not visible.
- Render config hashes only include `{"renderer": renderer_id}`. They do not include renderer options, feature lists, transform windows, trim policy, style preset, legacy-compat version, or renderer code/data schema version.
- `render_bundle` is output-directory based and not manifest-owned. That may be acceptable for ad hoc exports, but it violates the stronger refactor direction where renderers consume manifest-indexed artifacts and do not infer relationships from filenames.
- `GraphRendererContext` is nominally typed but immediately converted back to a plain dict with shallow-copied payloads. It does not provide typed artifact lookup, required-input validation, or stable run/case indexing.
- `src/graphs/structural_renderers.py` is a monolith. It contains artifact lookup, legacy payload conversion, detector replay, scoring, data aggregation, plotting, CSV writing, JSON export, and style constants in one file.
- The copied scripts under `graphs/scripts/` still assume the old `final_n500_bundle` layout, old `metadata/tests` JSON files, hard-coded naming conventions, and often mutate `sys.path`. They were not integrated into the refactored graph runtime.
- Some copied scripts include hard-coded local/Windows paths, for example `C:/Users/flexo/...` in `graphs/render_artifacts/render_artifacts_grid.py`. That should not be committed into a clean code path.
- The commit preserves old scripts as runnable files but does not clearly mark them as archival/legacy-only. Future developers will not know which graph path is authoritative.

### Contract And Data-Model Problems

- `_find_capture` returns the first capture matching `(model, plan_id, edit_method, producer)`. It ignores capture config hash, analysis variant, artifact ID, input content hash, and producer config. This can pick the wrong capture when multiple variants exist.
- `_analysis_by_case` has the same issue for analyses. It picks by producer only and ignores analysis variant/config hash.
- Case matching is done only by stringified `case_id`. There is no validation that execution cases and capture cases are complete, one-to-one, or drawn from the same input references.
- `target_layer` is taken from execution summary and applied to all cases. That is fragile if target layer ever becomes per-case, if a mixed plan exists, or if an analysis artifact carries more precise target metadata in `case["accuracy"]`.
- Baselines are matched by model only in `render_structural_artifact_grid`: `baselines_by_model = {model: payload}`. This is wrong for bundles with multiple plans, capture configs, seeds, model variants, or analysis variants.
- Output names collide for bundles. `artifact_grid_{model}.png` overwrites or collapses multiple plans/variants for the same model.
- Renderer-time data adaptation makes it hard to know whether a graph reflects stored analysis results or freshly replayed logic.

### Detector And Analysis Problems

- The graph code re-runs detector logic in the renderer via `detect_composite` and `detect_gpt_norm_cv`. Renderers should generally consume analysis artifacts, not recompute analyses.
- Replaying detectors in renderers can produce results that differ from stored analysis artifacts if detector defaults, trim windows, capture approximations, or analysis configs change.
- Detector choice is hard-coded from model names: GPT-2/GPT-J use `gpt`, everything else uses `composite`. This should come from registry/config/model-family metadata or the stored analysis artifact that was actually run.
- The renderer default parameters for composite/GPT detector replay are hard-coded. They are not linked to the analysis config used during `structural analysis`.
- The graph code can show detector metrics even when the corresponding analysis artifact was unavailable, stale, or never run, as long as `matrix-features` exists.
- `gpt-norm-cv` unavailable for Qwen3-8B is expected, but the graph code hides that distinction by selecting `composite` itself.

### `structural-artifact-grid` Problems

- The end goal was to render `structural-artifact-grid` and related plots from the newest refactored JSONs. The commit technically does that for simple cases, but through a legacy shim rather than a clean artifact API.
- `render_structural_artifact_grid` consumes synthetic `blind_detection.layer_features`, not a typed `matrix-features` artifact view.
- It renders only edited payloads. Baseline-only rendering is not supported except as an overlay when an edited payload exists.
- It overlays baseline by model only, not by matching plan/cases/capture profile/config.
- It emits only a PNG. The legacy script emitted PNG/PDF and had richer figure metadata. The new renderer does not emit a graph-specific data JSON with the exact layer means/stds used for the plot.
- It changed the legacy visual identity. Legacy artifact grid used wine edited line `#9f1239`, slate baseline `#475569`, near-black target line, shared legend, model labels, and paper-specific title text. The port uses different edited/baseline/target colors and omits the shared legend.
- It hard-codes the trim shading to two edge layers via `layers.min() - 0.5` to `layers.min() + 1.5` and symmetric max-side shading. Legacy used model/family-specific trim values, for example 5 for GPT-family models and 2 for others.
- It does not read trim settings from analysis config or model config.
- It uses `src.common.arrays.local_zscore` and `curvature`, whose boundary behavior differs from the copied legacy grid script. Legacy transforms used `NaN` for insufficient local neighborhoods and curvature boundaries; the shared helper returns zero at boundaries. That changes plotted aggregates.
- `_feature_matrix` in the port fixes the layer set to the first valid case and does not union/pad later cases. The legacy `collect_feature_matrix` handled differing layer sets by unioning/padding. The port can silently drop later-case layers or misrepresent partial captures.
- The figure title is generic (`{model} spectral artifacts`) and omits useful legacy context like `n` successful edits, baseline case count, model display name, and edited layer.
- The port does not preserve model display labels from legacy (`Qwen3-8B`, `GPT-J-6B`, etc.). It uses slugs directly.
- The port has no explicit paper/appendix output distinction and no legacy filename compatibility except approximate names.

### Other Renderer Problems

- `render_structural_windowed_detector_summary` treats baselines as expected negatives but still runs `_valid_tests` with `require_rome_success=False`, requiring layer features and detector replay. Baseline false-positive behavior depends on synthetic detector replay, not a stored edit-presence analysis.
- `render_structural_fleet_summary` takes only the first detection for a model/plan to fill target/detected fields. That loses case-level detail and is wrong for multi-case/multi-plan bundles.
- `render_structural_paper_export` invents a new schema `latium.structural_paper_graph_export.v1`, but it is not registered, documented, or tied to stable artifact IDs.
- `render_structural_spectral_profiles` consumes spectral analysis artifacts directly, while other renderers consume replayed detector results from captures. The commit mixes artifact-only and replayed behavior without a clear rule.
- Many outputs keep legacy filenames like `final_n500_detector_layer_window_radius1.*` even when the input run is not N=500. This is misleading for N=1 or other small verification runs.
- `_safe_float` and similar helpers can hide malformed/missing data by defaulting to `0.0`. That can turn missing data into plotted zeros rather than an unavailable renderer result.
- Exception handling in `render_run` catches all renderer exceptions and writes status `error`, but the command returns success. That can hide broken graph generation in automation unless callers inspect artifacts manually.

### Copied Legacy Script Problems

- `graphs/render_artifacts/render_artifacts_grid.py` still has hard-coded `BUNDLE_ROOT` and paper image output directories pointing to a Windows user path.
- `graphs/scripts/finalize_n500_bundle.py` is a large archival bundle builder tied to `final_n500_bundle`, old manifests, old raw JSON layout, and paper packaging concerns. It is not a refactored renderer.
- `graphs/scripts/export_qwen_paper_graph_jsons.py` is hard-coded around `qwen3-8b` and old bundle paths.
- `graphs/scripts/render_bundle_prefixtest_graphs.py` contains hard-coded prefixtest variant slugs and old bundle paths.
- Several scripts mutate `sys.path` to import repo modules. This should not be part of the clean graph runtime.
- These scripts may be useful as historical reference, but they should live under something like `scripts/legacy_graphs/` or `docs/legacy_graph_scripts/`, not be mixed with first-class graph code.

### Test Coverage Problems

- The structural renderer test mostly asserts output files exist. It does not validate the data in `structural-artifact-grid`, plotted series, target-layer lines, baseline overlays, transform behavior, or detector metrics.
- There is no golden comparison against the legacy artifact-grid aggregated data.
- There is no test for missing `matrix-features` producing a clear `unavailable` result.
- There is no test for multiple analysis variants/capture configs where producer-only lookup can select the wrong artifact.
- There is no test for multi-plan or bundle output filename collisions.
- There is no test for baseline matching by plan/config/case set.
- There is no test that graph renderers consume stored analysis artifacts instead of silently replaying detectors with defaults.
- There is no real-manifest fixture from the newest refactored JSON layout. The synthetic fixture is too simple and can pass while real runs fail.

## How To Improve The Graph Port

Recommended direction: keep the 1:1 legacy visual style as a compatibility goal, but do not keep the legacy data model in renderer internals.

1. Introduce a typed structural graph materialization layer.

- Add a small module such as `src/graphs/structural_inputs.py` or `src/graphs/structural/materialize.py`.
- It should read manifest records and produce typed objects like `GraphRun`, `GraphCase`, `LayerSeries`, and `DetectorResult`.
- It should index by artifact ID, model, plan ID, edit method, case ID, producer, config hash, and input refs.
- It should expose explicit missing-input reasons.

2. Make renderer dependencies declarative.

- Extend `RendererSpec` or add renderer metadata so each renderer declares required artifact kinds/producers.
- `structural-artifact-grid` should declare `execution` and `matrix-features` captures for edited runs, plus optional baseline `matrix-features` captures.
- Detector summary graphs should declare the relevant analysis artifacts instead of recomputing detector outputs.

3. Split `structural_renderers.py`.

- Move materialization out of renderer functions.
- Move plotting primitives/style into a small module.
- Move detector-summary renderers, artifact-grid renderer, spectral-profile renderer, and paper-export renderer into separate files.
- Keep each renderer small and focused.

4. Rebuild `structural-artifact-grid` first.

- Treat it as the primary target and make it clean before expanding to the rest.
- Inputs: edited `matrix-features`, optional matched baseline `matrix-features`, execution target metadata, and trim/model metadata.
- Outputs: PNG, PDF, data JSON with exact layer means/stds/counts, and `artifact.json` through the normal render runtime.
- Matching: model + plan ID + case selection/capture config/input refs, not model alone.
- Style: add a `legacy` style preset that exactly reproduces the old colors, labels, legend, trim shading, dpi, line widths, title, and transform behavior.
- Data: use legacy-compatible transforms if the goal is 1:1. Do not use shared helpers when their boundary semantics differ.

5. Separate visual compatibility from architecture.

- It is fine to copy visual constants and transform formulas from legacy code.
- It is not fine to copy old bundle paths, old `metadata/tests` contracts, or filename inference into the new runtime.
- Add compatibility functions with tests proving they reproduce legacy aggregate arrays.

6. Stop replaying detectors by default in graph renderers.

- Detector graphs should consume `analysis/detection/*` artifacts.
- If replay is needed for old visualizations, name it explicitly, include replay config in the render config hash, and mark outputs as replayed.
- Use the same detector config as the analysis artifact, not hard-coded defaults.

7. Make unavailable states explicit.

- If `matrix-features` is missing, `structural-artifact-grid` should return an unavailable render artifact with `missing captures: matrix-features`.
- If a baseline overlay is missing, the output should say baseline overlay unavailable but still render edited data if allowed.
- Avoid silent `No data` figures unless the graph artifact also records the missing reason.

8. Clean up legacy scripts.

- Either remove copied legacy scripts from the commit or move them to an explicitly archival location.
- Delete hard-coded Windows paths and bundle-specific defaults from production paths.
- If scripts must remain, mark them as historical references and exclude them from the normal graph registry/runtime.

9. Improve tests.

- Add a real minimal manifest fixture using the newest refactored artifact schema.
- Add golden tests for `structural-artifact-grid` aggregate data JSON.
- Add tests for missing captures, multiple variants, multiple plans, and baseline matching.
- Add tests that renderer config hashes change when feature lists/windows/style presets change.
- Add tests that detector renderers prefer stored analysis artifacts and do not recompute by default.

10. Commit strategy.

- Do not keep `c8f21558...` as one giant port commit if it can be avoided.
- Replace it with smaller commits: graph context/runtime cleanup, typed structural input materialization, artifact-grid renderer, detector-summary renderers, legacy script archival/removal, tests.
- Each commit should have its own verification and should not mix copied scripts with production runtime changes.

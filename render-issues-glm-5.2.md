# Render Issues & Rebuild Plan — GLM-5.2 analysis

Analysis author: GLM-5.2 (einfra/glm-5.2)
Date: 2026-06-27
Branch: `refactor-clean`
Base commit targeted: `5687ea7` (Fix second moment config and artifact preservation)

## Purpose

This document records (1) the issues found in the two commits to be removed
(`c8f2155` and `82428f6`), (2) the findings that drive the rebuild, and
(3) the full, finalized execution plan. It is the single reference for the
graph-renderer rebuild and the matrix-features/spectral split.

---

## 1. Git state and commits to remove

Current HEAD is `82428f6`. The history under it:

```
82428f6 fix: capture all layers in matrix-features, not just changed ones   (REMOVE)
c8f2155 feat: port legacy graph renderers to structural artifact pipeline   (REMOVE)
5687ea7 Fix second moment config and artifact preservation                 (KEEP — new base)
```

There is also a **stale cherry-pick** of `5687ea7` in progress, leaving
conflict markers in `src/graphs/registry.py`, `src/graphs/runtime.py`, and
`tests/test_graph_renderers.py`. Resetting to `5687ea7` (mixed) drops both
unwanted commits, clears the stale cherry-pick, and keeps all working-tree
changes.

### Commit `c8f2155` — what it added (22 files, ~7069 lines)

- `src/graphs/structural_renderers.py` — 1282-line monolith, 12 renderers.
- `src/graphs/context.py` — shallow-dict context shim.
- `src/graphs/registry.py` / `runtime.py` — 12 structural renderer registrations,
  bundle command, execution+capture+analysis inputs passed to every renderer.
- `src/command_handlers/graphs.py`, `src/command_aliases.py`,
  `src/config/command/graphs/bundle.yaml`, `src/config/graphs/default.yaml` —
  bundle plumbing.
- `tests/test_graph_renderers.py`, `tests/test_cli_smoke.py` — existence-only tests.
- 12 copied legacy scripts under `graphs/scripts/` and `graphs/render_artifacts/`
  (~5311 lines): Windows paths, `sys.path` mutations, bundle-specific assumptions,
  hard-coded `final_n500_bundle` roots.

### Commit `82428f6` — what it changed (1 file, 5 lines)

`src/structural/capture/producers.py`: `capture_matrix_features` now captures
all layers (not just changed ones) because the composite detector needs the full
post-edit layer profile. This behavior is correct and is preserved in the rebuild,
but the commit is dropped because it sits on top of `c8f2155` and will be
re-introduced cleanly inside the split commit.

### Uncommitted working-tree changes (the perf fix — preserved)

From `current.md`. Two files:

- `src/structural/detectors/blind_resident.py` —
  `BlindMSDDetector.compute_layer_features` switched from full `gpu_svd` to
  `gpu_svd_topk(k=top_k, niter=4)`; `pcs()` replaced with an equivalent
  row-normalized sum formula; returns `**profile` plus `pcs` and
  `row_alignment`.
- `src/structural/capture/producers.py` —
  `capture_matrix_features` passes `top_k=context.options["spectral_top_k"]`
  into `compute_layer_features` and writes the returned features directly.

These survive the reset (mixed reset keeps the working tree) and become part
of Commit 1, refined: `niter` aligned to `2` (matching `capture_spectral`) so
the shared SVD cache hits across both captures.

---

## 2. Issues found

### 2.1 SVD cache miss across captures (perf)

Both `capture_spectral` and `capture_matrix_features` call `gpu_svd_topk` on
the same per-layer weight matrices, but with different `niter`
(spectral: `niter=2` via `spectral_primitives.py:53`; matrix-features:
`niter=4` via `blind_resident.py:62`). The SVD cache key in
`src/common/linalg.py:403` is `("svd_topk", q, niter) + tensor_key`, so the
two captures never share a cached result — every layer's top-k SVD is computed
twice per case.

**Fix (Commit 1):** align both to `niter=2`. The cache also supports slicing
a larger-`q` cached result down to a smaller `q` (linalg.py:410-421), so once
`niter` matches, the second capture hits the cache.

Note: `gpu_svd` (full) at `src/common/linalg.py:326` has **zero callers** in
`src/structural/` — the earlier "full SVD hang" diagnosis in the prior
handoff was based on the pre-fix `blind_resident.py` which did call `gpu_svd`.
The perf fix in the working tree already removed that call.

### 2.2 `local_zscore` / `curvature` boundary divergence (1:1 blocker)

The legacy artifact-grid script (`graphs/render_artifacts/render_artifacts_grid.py`)
defines transforms that return **NaN at boundaries**:

```python
def transform_local_z(matrix, window):
    out = np.full_like(matrix, np.nan, dtype=float)
    ...
    if nbrs.shape[1] < 2:
        continue   # leaves NaN
    ...

def transform_curvature(matrix):
    out = np.full_like(matrix, np.nan, dtype=float)
    if n_layers >= 3:
        out[:, 1:-1] = np.abs(...)   # boundaries stay NaN
    return out
```

`src/common/arrays.local_zscore` / `curvature` return **0.0 at boundaries**
(arrays.py:22, 38). The `c8f2155` port imported the shared helpers, so its
artifact-grid aggregates differ from legacy at the edge layers. This breaks
1:1 output.

**Fix (Commit 2):** `src/graphs/structural/style.py` ships
legacy-compatible `transform_local_z` / `transform_curvature` (NaN-at-boundary,
`EPS=1e-8`) used only by the artifact-grid renderer. The shared
`src.common.arrays` helpers remain unchanged for detector code.

### 2.3 Missing `stored_top_k` gate on matrix-features

`capture_spectral` persists `stored_top_k` (producers.py:138/149) and the
spectral analysis validates `top_k <= stored_top_k` (spectral.py:218-223),
raising `unavailable` with a recapture reason when violated.
`capture_matrix_features` borrows `spectral_top_k` at capture time but
`capture/artifacts.py:92` writes `{}` into its capture config — so analyses
consuming `matrix-features` cannot gate on the captured rank.

**Fix (Commit 1):** matrix-features gets its own option `svd_top_k`, persisted
as `stored_top_k` in its capture config; `blind` / `composite` / `gpt-norm-cv`
analyses validate against it.

### 2.4 matrix-features / spectral coupling

`CAPTURE_PROFILES["spectral"] = ("spectral", "matrix-features")` (capture/registry.py:59)
bundles both producers under one name. `matrix-features` borrows
`options["spectral_top_k"]` as its rank (producers.py:207). They cannot be
enabled or swept independently.

**Fix (Commit 1):** split profiles — `"spectral": ("spectral",)`,
`"matrix": ("matrix-features",)`, `"paper": ("spectral","matrix-features")`.
matrix-features reads `options["svd_top_k"]`.

### 2.5 Hard-coded model names in detector-kind selection

`structural_renderers.py:287` (`_detector_kind`) special-cases `gpt2`* and
`gpt-j-6b` to pick the GPT detector. This duplicates the registry's
`model_families` mechanism (`src/registry.py:75`).

**Fix (Commit 2):** detector-kind comes from `RendererSpec.model_families` /
analysis `model_families` (`analysis/registry.py` already declares
`composite` → `("non-gpt",)`, `gpt-norm-cv` → `("gpt",)`). Renderers consume
the stored analysis artifact, so no model-name branching is needed in graph
code.

### 2.6 Monolith architectural problems (from `current.md` review, confirmed)

- No input declarations: renderers search `context["executions"]` /
  `["captures"]` / `["analyses"]` by producer string, bypassing the repo
  contract that components declare dependencies.
- Missing inputs silently produce empty data / "No data" panels instead of
  `unavailable` with a reason.
- `render_run` passes all artifacts to every renderer; config hash is only
  `{"renderer": renderer_id}` (no options/style/schema version).
- `_find_capture` matches by `(model, plan_id, edit_method, producer)` only —
  ignores config hash, variant, input content hash. Can pick the wrong capture.
- Baselines matched by model alone in `render_structural_artifact_grid` —
  wrong for multi-plan/multi-variant bundles.
- Renderer re-runs detector logic (`detect_composite` / `detect_gpt_norm_cv`)
  with hard-coded hyperparameters instead of consuming stored analysis
  artifacts.
- `final_n500_` / `n500` filename prefixes bake a 500-case sample size into
  outputs for N=1 runs.
- Cross-renderer file coupling: `render_structural_cohort_fp` re-reads JSON
  from disk that another renderer wrote.
- Dead imports: `Counter`, `scipy.stats`, `REFERENCE_COLOR`.

**Fix (Commit 2):** the new framework addresses all of the above — declarative
`required_captures` / `required_analyses` / `requires_execution` on
`RendererSpec`; `render_run` passes only declared inputs; missing inputs →
`unavailable` artifact with reason; config hash includes options + style +
schema version; typed `RenderContext` with artifact-ID-indexed lookup;
detector graphs consume analysis artifacts (no replay); filenames use run_id.

### 2.7 Copied legacy script problems

- `render_artifacts_grid.py` has hard-coded Windows paths
  (`C:/Users/flexo/Downloads/Temp/Esorics_petr/...`).
- 7 of 10 scripts mutate `sys.path`.
- All assume the old `final_n500_bundle` layout / `metadata`+`tests` JSON
  contract / `run_summary_latest.json`.
- `finalize_n500_bundle.py` (913 lines) is a bundle indexer, not a renderer.
- No clear marking that they are archival.

**Fix (Commit 2):** drop entirely. They remain in git history at `c8f2155` as
reference. The new framework is designed to recreate every legacy graph;
artifact-grid is the reference implementation, the rest follow the same
pattern.

### 2.8 Open question: top-k approximation for blind features on large models

The perf fix uses `gpu_svd_topk(k=svd_top_k, niter=2)` instead of full SVD for
`matrix-features`. For Qwen3-8B (36 layers, large proj matrices) this is
required for the capture to complete. The blind features (`spectral_gap`,
`top1_energy`, `effective_rank`, `spectral_entropy`, `norm_cv`, `pcs`,
`row_alignment`) are all computable from top-k singular values + vectors.

Open: is top-k (default 50) sufficient for `effective_rank` / `spectral_entropy`
estimates used by the blind IsolationForest detector? Full-spectrum versions
would be more accurate but OOM on large models. Decision: keep top-k as the
default; if exact full-spectrum is ever required, add a separate configurable
capture mode rather than defaulting to full SVD. This is noted but not blocking.

---

## 3. Full execution plan

### Step 0 — Git reset

```
git reset 5687ea7
```

Mixed reset. Drops `82428f6` + `c8f2155`, clears the stale cherry-pick of
`5687ea7`, keeps the working-tree perf fix in `producers.py` +
`blind_resident.py`. `current.md` stays untracked.

Result: clean base at `5687ea7` with two modified files in the working tree.

### Commit 1 — matrix-features / spectral full split + perf fix

**Files:**
- `src/structural/capture/producers.py`
- `src/structural/capture/registry.py`
- `src/structural/capture/artifacts.py`
- `src/structural/detectors/blind_resident.py`
- `src/structural/analysis/registry.py`
- `src/structural/analysis/runtime.py`
- `src/structural/detectors/blind.py` (add `stored_top_k` gate)
- `src/structural/detectors/detector_methods.py` (gate composite/gpt-norm-cv)
- `src/config/structural/default.yaml` (add `svd_top_k`)

**Changes:**

1. **Perf fix (refined).** `BlindMSDDetector.compute_layer_features` uses
   `gpu_svd_topk(k=svd_top_k, niter=2)` — aligned with `capture_spectral`'s
   `niter=2` so the shared SVD cache hits. `pcs` via row-normalized sum
   formula (kept from working tree).

2. **Own rank option.** `capture_matrix_features` reads
   `options["svd_top_k"]` (default 50), not `spectral_top_k`. Persists
   `stored_top_k` in its output (mirroring `capture_spectral`).

3. **Capture config persistence.** `capture/artifacts.py` writes
   `{"svd_top_k": int(options["svd_top_k"])}` into the matrix-features
   capture config record (currently writes `{}` at line 92).

4. **Split profiles.** `CAPTURE_PROFILES`:
   ```python
   "none": (),
   "spectral": ("spectral",),
   "matrix": ("matrix-features",),
   "paper": ("spectral", "matrix-features"),
   "full": CAPTURES.identifiers(),
   ```

5. **stored_top_k gate.** `blind` / `composite` / `gpt-norm-cv` analyses
   validate `top_k <= stored_top_k` and raise `unavailable` with recapture
   reason when violated (mirror `spectral.py:218-223`). Analysis variant
   field mapping adds `("svd_top_k", "top_k")` projection in
   `analysis/registry.py`.

6. **All-layers capture.** Preserve the `82428f6` behavior:
  `capture_matrix_features` captures all layers for the edit case (composite
  needs the full post-edit profile).

**Verification:**
- `python -m compileall` on changed files.
- Run `tests/test_analysis_runtime.py`, `tests/test_capture*.py`,
  `tests/test_structural_planning.py`.
- Confirm `CAPTURE_PROFILES` split via a quick import check.

### Commit 2 — new modular graph framework + artifact-grid renderer

**A new framework, not a port of the c8f2155 shim.** Adding a graph = write
one renderer function + register one `RendererSpec` with declared inputs.
Scalable and modular.

#### `src/graphs/registry.py` — declarative renderer specs

```python
@dataclass(frozen=True)
class RendererSpec(RegistryEntry):
    runner: str = ""
    model_families: tuple[str, ...] = ("all",)
    required_captures: tuple[str, ...] = ()      # NEW
    required_analyses: tuple[str, ...] = ()      # NEW
    requires_execution: bool = False             # NEW
    options: tuple[str, ...] = ()                # NEW: option keys hashed into config
```

Keep the pre-port clean renderers (`paper`, `detector`, `run-summary`).
Register `structural-artifact-grid` with `required_captures=("matrix-features",)`,
`requires_execution=True`.

#### `src/graphs/runtime.py` — manifest-driven, per-renderer inputs

`render_run` reads **only declared inputs** per renderer from the manifest via
`RunArtifactReader`. Missing declared inputs → writes `unavailable` artifact
with reason, skips the renderer. Config hash =
`{renderer, options, style_preset, schema_version}`. No bundle command.

#### `src/graphs/context.py` — typed RenderContext

```python
@dataclass(frozen=True)
class RenderContext:
    run_root: Path
    output_dir: Path
    manifest: dict[str, Any]
    executions: tuple[dict, ...]        # only if requires_execution
    captures: dict[str, tuple[dict, ...]]   # keyed by producer, only declared
    analyses: dict[str, tuple[dict, ...]]   # keyed by producer, only declared
```

Replaces the shallow-dict shim. Renderers receive typed lookup, not a plain
dict to search.

#### `src/graphs/structural/` (new package)

**`materialize.py`** — typed views from raw artifact JSON:
- `GraphCase`, `LayerSeries`, `DetectorResult` dataclasses.
- Indexed by case_id / layer.
- `Unavailable` exception with reason string.
- No legacy `metadata` / `tests` shape — reads the refactored artifact schema
  directly (`run`, `cases`, `data.families`, `data.layers`, `summary`).
- Baseline matching by model + plan_id + case-set (not model alone).

**`style.py`** — `LegacyStyle` preset:
- Exact wine palette: `EDITED_COLOR="#9f1239"`, `EDITED_BAND="#fda4af"`,
  `BASELINE_COLOR="#475569"`, `TRIM_SHADE="#f1f5f9"`, `ELL_STAR="#0f172a"`.
- Panel size `4.1 x 2.55`, dpi 200, rcParams (white facecolor, edge `#222222`,
  grid `#d0d0d0` alpha 0.30, legend framealpha 0.92).
- **Legacy-compatible transforms** (NaN at boundaries, `EPS=1e-8`):
  `transform_local_z(matrix, window)`, `transform_curvature(matrix)`.
- Trim resolver: 2 default, 5 GPT-family, from model config / `model_families`.
- Model display labels: `Qwen3-8B`, `GPT-J-6B`, etc. (lookup table, not
  hard-coded inside renderer logic).

**`artifact_grid.py`** — the 5x4 per-layer signal grid (Figs 2/3), 1:1 legacy:
- Declares `required_captures=("matrix-features",)`, `requires_execution=True`,
  optional baseline `matrix-features`.
- Inputs: edited `matrix-features` (all layers), optional matched baseline
  `matrix-features`, execution target metadata, trim/model metadata.
- Outputs: PNG + PDF + data JSON (exact layer means/stds/counts) + `artifact.json`.
- 5 features (SG, TE, RA, CV, ER) x 4 transforms (raw, lz5, lz7, curv).
- Wine edited line +/-1 sigma band, slate dashed baseline, dotted ell*, trim
  shading, shared legend, model display labels.
- Returns `unavailable` with reason if `matrix-features` missing.
- Baseline matched by model + plan_id + case-set.
- Filenames use run_id, not `final_n500_`.

#### `src/graphs/renderers.py`

Keep existing `paper` / `detector` / `run-summary` (pre-port clean ones),
adapted to the typed `RenderContext`.

#### Dropped

- `src/graphs/structural_renderers.py` (monolith) — not carried over.
- `src/graphs/context.py` shallow-dict shim — replaced by typed `RenderContext`.
- `graphs/scripts/` and `graphs/render_artifacts/` (12 legacy scripts) —
  dropped. Recoverable from git history at `c8f2155` if needed as reference.
- `src/command_handlers/graphs.py` bundle command, `bundle.yaml`,
  `command_aliases.py` bundle alias — dropped.
- `tests/test_graph_renderers.py` — rewritten for the typed context.

**Verification:**
- `python -m compileall src/graphs`.
- Run `tests/test_graph_renderers.py` (rewritten).
- Render `structural-artifact-grid` against the local run
  `analysis_out/20260627_001042` and visually verify 1:1 with legacy
  `render_artifacts_grid.py` output.
- Note: that run was captured pre-split (matrix-features borrowed
  `spectral_top_k`, no `stored_top_k` in config). The artifact-grid renderer
  reads raw capture data, not the config gate, so it renders fine. Full
  end-to-end split verification (capture + analysis + render) needs a fresh
  remote run — deferred until remote access returns.

### Renderers deferred (framework supports them; added later via the same pattern)

These are not ported in this round. The framework is designed so each is one
renderer file + one `RendererSpec` registration:

- `structural-detector-layer-window` — consumes `composite` / `gpt-norm-cv`
  analysis artifacts (no replay). CSV/JSON/MD tables, strict vs +/-1 window.
- `structural-spectral-profiles` — consumes `spectral` analysis artifacts.
- `structural-cohort-envelope` (Figs 5/6) — wine palette, 5-95 percentile band.
- `structural-cohort-fp` (Fig 4) — false-positive audit, wine colormap.
- `structural-rome-success-metrics` — ROME metric heatmaps/bars.
- `structural-windowed-detector-summary` — confusion matrix, rates.
- `structural-fleet-summary`, `structural-prefixtest-summary`,
  `structural-detector-stacked-variants`, `structural-bundle-index`,
  `structural-paper-export`.

Each will declare its required captures/analyses and consume stored artifacts
(no detector replay, no model-name branching, no `final_n500_` filenames).

---

## 4. Verification summary

| Step | What | How |
|------|------|-----|
| Commit 1 | capture/analysis split | compileall + existing tests + import check |
| Commit 2 framework | registry/runtime/context | compileall + rewritten renderer tests |
| Commit 2 artifact-grid | 1:1 legacy match | render against local run `20260627_001042`, visual compare |
| Full split end-to-end | capture + analysis + render | deferred — needs fresh remote run |

---

## 5. Notes

- The local run `analysis_out/20260627_001042` (N=1 Qwen3-8B) has 50 graph
  files from the `c8f2155` port and verified detector output (target layer 10,
  detected layer 10, strict accuracy 1.0, +/-1 window accuracy 1.0). It serves
  as the visual regression baseline for the rebuilt artifact-grid.
- `gpt-norm-cv` is always `unavailable` for Qwen3-8B (not a GPT model) —
  expected. The framework must represent this as an explicit `unavailable`
  artifact, not hide it.
- `MPLCONFIGDIR=/tmp/matplotlib_cache` is required on remote for matplotlib
  (not relevant locally).
- Remote cluster access (ubuntu@64.247.196.81 via metju@kubapc) is unavailable
  during this round. All verification is local.

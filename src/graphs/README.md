# Graphs

Renderers consume completed analysis artifacts and write render artifacts under
`<run-root>/graphs/<renderer-id>/`.

The renderer runtime passes manifest-backed executions, captures, analyses, and
the manifest itself to renderer functions. New renderers declare the producers
they need in `RendererSpec`; missing required inputs produce an `unavailable`
render artifact with a reason. Render artifacts depend on their selected inputs,
so re-rendering skips current outputs and stale graphs are invalidated when
upstream artifacts change.

Built-in renderers:

- `paper`: machine-readable paper analysis data.
- `detector`: aggregate detector accuracy summary.
- `run-summary`: one JSON row per analysis artifact.
- `rome-success`: execution success-rate CSV/JSON/PNG.
- `detector-window`: exact and +/-1 layer detector accuracy.
- `detector-signals`: per-case layer signal profile plots.
- `structural-artifact-grid`: fixed 5x4 matrix artifact grid from
  current `matrix-features` artifacts.
- `rome-relative-profile-grid`: eight per-layer relative ROME methods with
  edited spread, a matched unedited baseline, selected B0 fits, overview
  figures, case diagnostics, and machine-readable JSON.
- `structural-ccs-lines`: six panels of spectral gap and top-1 energy at raw,
  window-5, and window-7 scale. Each panel shows individual edited cases,
  their mean and spread, and the matched unedited mean.

`ccs-report` selects `paper`, `detector`, `rome-success`, `detector-window`,
and `structural-ccs-lines` together. It requires all report figures and their
inputs to be present; missing output makes the command fail. The old
aggregate-only renderer preset `paper` is retired. The `paper` renderer itself
still writes the report's machine-readable summary.

For a new CCS run, use `python -m src structural run structural=ccs-report`
with the desired model and case overrides. For saved artifacts, use
`python -m src graphs run <run-root>`; this also defaults to `ccs-report`.

## Add A Renderer

1. Add a function in `renderers.py` or a focused module under
   `src/graphs/structural/`.
2. Register it in `registry.py`, declaring required captures/analyses.
3. Add it to `RENDERER_PRESETS` if useful.
4. Test it in `tests/test_graph_renderers.py`.

Renderer functions receive `src.graphs.context.RenderContext` with typed access
to `output_dir`, `manifest`, `executions`, captures/analyses grouped by
producer, renderer `options`, and `style_preset`. Return the list of files
written; the graph runtime stores those paths in the render artifact summary.
Older generic renderers may still call `context.as_mapping()` internally as a
compatibility bridge, but new graph makers should use `RenderContext` directly.

Renderer options are configured through Hydra under
`graphs.renderers.<renderer-id>` for re-rendering and
`structural.render.renderers.<renderer-id>` for end-to-end runs. For example:

```bash
python -m src graphs run analysis_out/run-id \
  graphs.renderer_preset=structural-paper \
  graphs.style_preset=default \
  graphs.renderers.structural-artifact-grid.formats='[png,pdf,json]'
```

The structural artifact-grid transforms use shared `src.common.arrays`
functions, not graph-local math. `local_zscore(..., axis=1, fill_value=nan,
absolute=true)` preserves the established grid's layer-wise absolute local-z behavior;
`curvature(..., axis=1, pad_value=nan)` preserves NaN boundaries in aggregate
plots.

Use `graphs.renderer_preset=rome-relative-paper` to render saved relative
analyses. The end-to-end structural preset is
`structural.analysis.preset=rome-relative` together with
`structural.render.renderer_preset=rome-relative-paper` and
`structural.render.enabled=true`. Configure `case_pages` (`errors`, `all`, or
`none`), `case_traces` (`auto`, `true`, or `false`), and `formats` under
`graphs.renderers.rome-relative-profile-grid`. Matching requires the same
model, plan, analysis producer, and config hash; absent baselines are annotated
and ambiguous matches are rejected.

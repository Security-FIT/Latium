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
- `structural-artifact-grid`: legacy-compatible 5x4 matrix artifact grid from
  current `matrix-features` artifacts.
- `rome-detector-explainer`: per-case overview, all 15 weighted-spectrum layer
  profiles, blind-presence cutoffs, clean-delta evidence, CSV exports, and an
  aggregate outcome graph.

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
`graphs.renderers.<renderer-id>`. For example:

```bash
python -m src graphs run analysis_out/run-id \
  graphs.renderer_preset=structural-paper \
  graphs.style_preset=default \
  graphs.renderers.structural-artifact-grid.formats='[png,pdf,json]'
```

The structural artifact-grid transforms use shared `src.common.arrays`
functions, not graph-local math. `local_zscore(..., axis=1, fill_value=nan,
absolute=true)` preserves the legacy grid's layer-wise absolute local-z behavior;
`curvature(..., axis=1, pad_value=nan)` preserves NaN boundaries in aggregate
plots.

The complete ROME detector view can be rendered from a structural run that used
the `rome-presence` capture/analysis presets:

```bash
python -m src graphs run analysis_out/RUN_ID \
  graphs.renderer_preset=rome-presence
```

Outputs under `graphs/rome-detector-explainer/` include a detector walkthrough
and an all-statistics grid for every complete case, `detector-data.json` beside
each case, an all-layer CSV, a case summary CSV, aggregate outcome charts, and
a manifest-linked index. Boundary shading comes from the analysis artifact's
actual candidate set; it does not route on model family.

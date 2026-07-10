# Graphs

Renderers consume completed analysis artifacts and write render artifacts under
`<run-root>/graphs/<renderer-id>/`.

The renderer runtime passes manifest-backed executions, captures, analyses, and
the manifest itself to renderer functions. New renderers declare the producers
they need in `RendererSpec`; missing required inputs produce an `unavailable`
render artifact with a reason. Render artifacts depend on their selected inputs,
so re-rendering skips current outputs and stale graphs are invalidated when
upstream artifacts change.

Every renderer must declare its artifact inputs. Analysis configuration hashes
are preserved in summary rows, plot labels, CSV columns, and per-case output
paths, so multiple variants never overwrite or aggregate into one another.

Built-in renderers:

- `paper`: machine-readable paper analysis data.
- `detector`: aggregate detector accuracy summary.
- `run-summary`: one JSON row per analysis artifact.
- `rome-success`: execution success-rate CSV/JSON/PNG.
- `detector-window`: exact and +/-1 layer detector accuracy.
- `detector-signals`: per-case layer signal profile plots.
- `structural-artifact-grid`: legacy-compatible 5x4 matrix artifact grid from
  current `matrix-features` artifacts.

## Add A Renderer

1. Add a function in `renderers.py` or a focused module under
   `src/graphs/structural/`.
2. Register it in `registry.py`, declaring required captures/analyses.
3. Add it to `RENDERER_PRESETS` if useful.
4. Test it in `tests/test_graph_renderers.py`.

Renderer functions receive `src.graphs.context.RenderContext` with typed access
to `output_dir`, `manifest`, `executions`, captures/analyses grouped by
producer, and renderer `options`. Return the list of files written; the graph
runtime stores those paths in the render artifact summary.

Unexpected renderer exceptions are saved as `status=error` artifacts and then
raised by default so CLI automation receives a non-zero exit. Set
`graphs.continue_on_error=true` only for an explicitly best-effort batch.

Renderer options are configured through Hydra under
`graphs.renderers.<renderer-id>`. For example:

```bash
python -m src graphs run analysis_out/run-id \
  graphs.renderer_preset=structural-paper \
  graphs.renderers.structural-artifact-grid.formats='[png,pdf,json]'
```

The structural artifact-grid transforms use shared `src.common.arrays`
functions, not graph-local math. `local_zscore(..., axis=1, fill_value=nan,
absolute=true)` preserves the legacy grid's layer-wise absolute local-z behavior;
`curvature(..., axis=1, pad_value=nan)` preserves NaN boundaries in aggregate
plots.

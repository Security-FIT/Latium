# Graphs

Renderers consume completed analysis artifacts and write render artifacts under
`<run-root>/graphs/<renderer-id>/`.

The renderer runtime passes manifest-backed executions, captures, analyses, and
the manifest itself to renderer functions. Render artifacts depend on those
inputs, so re-rendering skips current outputs and stale graphs are invalidated
when upstream artifacts change.

Built-in renderers:

- `paper`: machine-readable paper analysis data.
- `detector`: aggregate detector accuracy summary.
- `run-summary`: one JSON row per analysis artifact.
- `rome-success`: execution success-rate CSV/JSON/PNG.
- `detector-window`: exact and +/-1 layer detector accuracy.
- `detector-signals`: per-case layer signal profile plots.

## Add A Renderer

1. Add a function in `renderers.py`.
2. Register it in `registry.py`.
3. Add it to `RENDERER_PRESETS` if useful.
4. Test it in `tests/test_graph_renderers.py`.

Renderer functions receive a context dict with `output_dir`, `manifest`,
`executions`, `captures`, `analyses`, and run metadata. Return the list of files
written; the graph runtime stores those paths in the render artifact summary.

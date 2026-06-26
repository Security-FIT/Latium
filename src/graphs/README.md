# Graphs

Renderers consume completed analysis artifacts and write render artifacts under
`<run-root>/graphs/<renderer-id>/`.

## Add A Renderer

1. Add a function in `renderers.py`.
2. Register it in `registry.py`.
3. Add it to `RENDERER_PRESETS` if useful.
4. Test it in `tests/test_graph_renderers.py`.

Renderer functions receive a context dict with `output_dir`, `analyses`, and run
metadata. Return the list of files written; the graph runtime stores those paths
in the render artifact summary.

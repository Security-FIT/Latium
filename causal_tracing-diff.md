# Causal Tracing: Production vs Notebook

The full current method is documented in `causal_tracing.md`. This file only
records the implementation differences that matter when comparing runs.

| Aspect | Production command | Audited long notebook |
|---|---|---|
| Entry point | `python3 -m src causal-trace` | `notebooks/causal-tracing-auto-v2.ipynb` |
| Method | Paired subject-last MLP-window trace | Same |
| Window selection | Discovery argmax, then test that exact held-out center | Same |
| Eligible windows | Full width only | Same |
| Module mapping | Selected model YAML's exact MLP projection template | Built-in presets with adapter fallbacks |
| Subject mapping | Offset mappings, then token-ID fallback | Offset mappings required |
| Output root | `analysis_out/causal_trace/` | `analysis_out/causal_tracing_auto_v2/` |
| Optional model-config update | Explicit `overwrite_model_config_layer=true` | No source YAML update |
| Covariance and ROME | Not part of the command | Optional downstream notebook stages |

`notebooks/causal-tracing-auto.ipynb` is the older exploratory notebook. It
uses middle-band, near-peak, adjacency, and noninferiority heuristics that are
not part of the audited production selector. It should not be used as the
parity reference for current results.

The retired “standard” and “alternative” source workflows previously described
here no longer represent the active code.

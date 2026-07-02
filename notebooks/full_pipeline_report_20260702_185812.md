# Full Pipeline Causal Trace + ROME Report

Run timestamp: `20260702_185812`

Causal tracing runs for every model. Covariance and ROME run only when a strict confirmed graph-selected layer differs from the existing config/reference layer. Diagnostic candidates are report-only by default.

## Settings

- Models: `['mistral-7b-v0.3']`
- Causal facts requested: `100`
- Noise samples: `10`
- Window mode/size: `canonical_rome` / `10`
- ROME benchmark edits: `100`
- Covariance target samples: `100000`

## Model Summary

| model | strict layer | diagnostic layer | region centers | config ref | strict failure | facts | downstream skip | cov saved | raw cov saved | ROME n | ES | PS | NS | overall | plot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral-7b-v0.3 | 5 |  | 5,6,7 | 17 |  | 20 |  | yes | yes | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812/early_site_causal_region_with_config_reference.png |

## Output Files

### mistral-7b-v0.3

- Causal trace output dir: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812`
- PNG graph: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812/final_selection.json`
- Selection diagnostics JSON: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812/selection_diagnostics.json`
- Inverse covariance: `data/second_moment_stats/mistralai_Mistral-7B-v0.3_5_SM_Method.WIKIPEDIA_100000_2.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/mistralai_Mistral-7B-v0.3_5_raw_covariance_SM_Method.WIKIPEDIA_100000_2.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260702_185812/rome_benchmark`


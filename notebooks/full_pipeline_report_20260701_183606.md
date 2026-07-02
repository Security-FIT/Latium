# Full Pipeline Causal Trace + ROME Report

Run timestamp: `20260701_183606`

This run uses the causal graph representative center as the benchmark layer. The config/reference layer is shown only as a graph marker and is not used for layer selection.

## Settings

- Models: `['gpt2-xl', 'qwen3-4b', 'mistral-7b-v0.1', 'gpt-j-6b', 'llama2-7b', 'granite4-micro', 'qwen3-8b', 'mistral-7b-v0.3', 'opt-6.7b', 'falcon-7b', 'deepseek-7b-base']`
- Causal facts requested: `100`
- Noise samples: `10`
- Window mode/size: `canonical_rome` / `10`
- ROME benchmark edits: `100`
- Covariance target samples: `100000`

## Model Summary

| model | graph layer | region centers | config ref | facts | downstream skip | cov saved | raw cov saved | ROME n | ES | PS | NS | overall | plot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2-xl | 17 | 13,14,15,16,17,18,19 | 18 | 100 |  | yes | yes | 100 | 0.9800 | 0.9400 | 0.7740 | 0.8268 | analysis_out/full_pipeline_causal/gpt2-xl_20260701_183606/early_site_causal_region_with_config_reference.png |
| qwen3-4b | 6 | 5,6,7,8,9 | 12 | 100 |  | yes | yes | 100 | 1.0000 | 0.9700 | 0.8150 | 0.8738 | analysis_out/full_pipeline_causal/qwen3-4b_20260701_183606/early_site_causal_region_with_config_reference.png |
| mistral-7b-v0.1 | 5 | 5,6,7,8,9 | 5 | 100 | graph_selected_layer_matches_existing_config:5 | skipped | skipped | 0 |  |  |  |  | analysis_out/full_pipeline_causal/mistral-7b-v0.1_20260701_183606/early_site_causal_region_with_config_reference.png |
| gpt-j-6b | 5 | 5,6,7 | 5 | 100 | graph_selected_layer_matches_existing_config:5 | skipped | skipped | 0 |  |  |  |  | analysis_out/full_pipeline_causal/gpt-j-6b_20260701_183606/early_site_causal_region_with_config_reference.png |
| llama2-7b | 6 | 5,6,7,8 | 19 | 100 |  | yes | yes | 100 | 0.9500 | 0.9250 | 0.7950 | 0.8122 | analysis_out/full_pipeline_causal/llama2-7b_20260701_183606/early_site_causal_region_with_config_reference.png |
| granite4-micro |  |  | 9 | 100 | no_confirmed_robust_graph_layer | skipped | skipped | 0 |  |  |  |  | analysis_out/full_pipeline_causal/granite4-micro_20260701_183606/early_site_causal_region_with_config_reference.png |
| qwen3-8b | 7 | 5,6,7,8,9,10 | 10 | 100 |  | yes | yes | 100 | 1.0000 | 0.9850 | 0.8240 | 0.8999 | analysis_out/full_pipeline_causal/qwen3-8b_20260701_183606/early_site_causal_region_with_config_reference.png |
| mistral-7b-v0.3 | 5 | 5,6,7,8 | 17 | 100 |  | yes | yes | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260701_183606/early_site_causal_region_with_config_reference.png |
| opt-6.7b | 15 | 13,14,15,16,17,18 | 15 | 100 | graph_selected_layer_matches_existing_config:15 | skipped | skipped | 0 |  |  |  |  | analysis_out/full_pipeline_causal/opt-6.7b_20260701_183606/early_site_causal_region_with_config_reference.png |
| falcon-7b |  |  | 3 | 100 | no_confirmed_robust_graph_layer | skipped | skipped | 0 |  |  |  |  | analysis_out/full_pipeline_causal/falcon-7b_20260701_183606/early_site_causal_region_with_config_reference.png |
| deepseek-7b-base | 20 | 18,19,20,21 | 6 | 100 |  | yes | yes | 98 | 0.9796 | 1.0000 | 0.6245 | 0.7079 | analysis_out/full_pipeline_causal/deepseek-7b-base_20260701_183606/early_site_causal_region_with_config_reference.png |

## Output Files

### gpt2-xl

- Causal trace output dir: `analysis_out/full_pipeline_causal/gpt2-xl_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/gpt2-xl_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/gpt2-xl_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/gpt2-xl_17_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/gpt2-xl_17_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/gpt2-xl_20260701_183606/rome_benchmark`

### qwen3-4b

- Causal trace output dir: `analysis_out/full_pipeline_causal/qwen3-4b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/qwen3-4b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/qwen3-4b_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/Qwen_Qwen3-4B_6_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/Qwen_Qwen3-4B_6_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/qwen3-4b_20260701_183606/rome_benchmark`

### mistral-7b-v0.1

- Causal trace output dir: `analysis_out/full_pipeline_causal/mistral-7b-v0.1_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/mistral-7b-v0.1_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/mistral-7b-v0.1_20260701_183606/final_selection.json`
- Inverse covariance: ``
- Raw covariance: ``
- ROME benchmark output dir: `None`

### gpt-j-6b

- Causal trace output dir: `analysis_out/full_pipeline_causal/gpt-j-6b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/gpt-j-6b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/gpt-j-6b_20260701_183606/final_selection.json`
- Inverse covariance: ``
- Raw covariance: ``
- ROME benchmark output dir: `None`

### llama2-7b

- Causal trace output dir: `analysis_out/full_pipeline_causal/llama2-7b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/llama2-7b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/llama2-7b_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/NousResearch_Llama-2-7b-hf_6_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/NousResearch_Llama-2-7b-hf_6_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/llama2-7b_20260701_183606/rome_benchmark`

### granite4-micro

- Causal trace output dir: `analysis_out/full_pipeline_causal/granite4-micro_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/granite4-micro_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/granite4-micro_20260701_183606/final_selection.json`
- Inverse covariance: ``
- Raw covariance: ``
- ROME benchmark output dir: `None`

### qwen3-8b

- Causal trace output dir: `analysis_out/full_pipeline_causal/qwen3-8b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/qwen3-8b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/qwen3-8b_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/Qwen_Qwen3-8B_7_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/Qwen_Qwen3-8B_7_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/qwen3-8b_20260701_183606/rome_benchmark`

### mistral-7b-v0.3

- Causal trace output dir: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/mistralai_Mistral-7B-v0.3_5_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/mistralai_Mistral-7B-v0.3_5_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/mistral-7b-v0.3_20260701_183606/rome_benchmark`

### opt-6.7b

- Causal trace output dir: `analysis_out/full_pipeline_causal/opt-6.7b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/opt-6.7b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/opt-6.7b_20260701_183606/final_selection.json`
- Inverse covariance: ``
- Raw covariance: ``
- ROME benchmark output dir: `None`

### falcon-7b

- Causal trace output dir: `analysis_out/full_pipeline_causal/falcon-7b_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/falcon-7b_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/falcon-7b_20260701_183606/final_selection.json`
- Inverse covariance: ``
- Raw covariance: ``
- ROME benchmark output dir: `None`

### deepseek-7b-base

- Causal trace output dir: `analysis_out/full_pipeline_causal/deepseek-7b-base_20260701_183606`
- PNG graph: `analysis_out/full_pipeline_causal/deepseek-7b-base_20260701_183606/early_site_causal_region_with_config_reference.png`
- Final causal selection JSON: `analysis_out/full_pipeline_causal/deepseek-7b-base_20260701_183606/final_selection.json`
- Inverse covariance: `data/second_moment_stats/deepseek-ai_deepseek-llm-7b-base_20_SM_Method.WIKIPEDIA_100000.pt`
- Raw covariance: `data/second_moment_stats/raw_covariance/deepseek-ai_deepseek-llm-7b-base_20_raw_covariance_SM_Method.WIKIPEDIA_100000.pt`
- ROME benchmark output dir: `analysis_out/full_pipeline_causal/deepseek-7b-base_20260701_183606/rome_benchmark`


# Compact ROME Benchmark Summary

Source: `rome_benchmark_summary_20260701_183606.csv`

`current_config_layer -> new_layer` means the layer currently in the model config mapped to the layer selected by causal tracing and used for the ROME benchmark. If they matched, the benchmark was skipped by design.

`NS_specificity` is the neighborhood preservation/specificity metric from `mean_neighborhood_score`.

| model            | current_config_layer -> new_layer          | current_config_layer | new_causal_layer_tested | n_evaluated | ES_efficacy | EM_efficacy_magnitude | PS_paraphrase | NS_specificity | overall | skip_reason                                     |
| ---------------- | ------------------------------------------ | -------------------- | ----------------------- | ----------- | ----------- | --------------------- | ------------- | -------------- | ------- | ----------------------------------------------- |
| gpt2-xl          | 18 -> 17                                   | 18                   | 17                      | 100         | 0.980       | 0.925                 | 0.940         | 0.774          | 0.827   |                                                 |
| llama2-7b        | 19 -> 6                                    | 19                   | 6                       | 100         | 0.950       | 0.732                 | 0.925         | 0.795          | 0.812   |                                                 |
| qwen3-4b         | 12 -> 6                                    | 12                   | 6                       | 100         | 1.000       | 0.988                 | 0.970         | 0.815          | 0.874   |                                                 |
| qwen3-8b         | 10 -> 7                                    | 10                   | 7                       | 100         | 1.000       | 0.992                 | 0.985         | 0.824          | 0.900   |                                                 |
| deepseek-7b-base | 6 -> 20                                    | 6                    | 20                      | 98          | 0.980       | 0.607                 | 1.000         | 0.624          | 0.708   |                                                 |
| falcon-7b        | ? -> none (no robust new layer)            |                      |                         | 0           |             |                       |               |                |         | no_confirmed_robust_graph_layer                 |
| gpt-j-6b         | 5 -> 5 (same as current config; skipped)   |                      |                         | 0           |             |                       |               |                |         | graph_selected_layer_matches_existing_config:5  |
| granite4-micro   | ? -> none (no robust new layer)            |                      |                         | 0           |             |                       |               |                |         | no_confirmed_robust_graph_layer                 |
| mistral-7b-v0.1  | 5 -> 5 (same as current config; skipped)   |                      |                         | 0           |             |                       |               |                |         | graph_selected_layer_matches_existing_config:5  |
| mistral-7b-v0.3  | 17 -> 5                                    | 17                   | 5                       | 0           | 0.000       | 0.000                 | 0.000         | 0.000          | 0.000   |                                                 |
| opt-6.7b         | 15 -> 15 (same as current config; skipped) |                      |                         | 0           |             |                       |               |                |         | graph_selected_layer_matches_existing_config:15 |

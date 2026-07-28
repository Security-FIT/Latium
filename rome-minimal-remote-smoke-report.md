# Minimal ROME detector: exact-covariance remote smoke

## Scope

This execution smoke ran CounterFact cases 0-1 on every remote model whose
current `origin/main` configuration had an exact model-, layer-, and
sample-count-matched 100,000-sample Wikipedia second moment.

- Host: `150-136-40-217`
- GPU: NVIDIA A100-SXM4-40GB
- Minimal detector commit: `d6ec938`
- Current-main runtime/config commit: `29ea568`
- Edit method: ROME only
- Capture/analysis: `rome-presence` only
- Artifact schema: `rome-detector-minimal-v1`
- Rendering: disabled; no graph directories were created

The smoke used a disposable remote integration worktree so the current-main
model runtime/configuration could exercise the minimal detector without
modifying either `main` or `detector-simplification`.

This is an execution and compatibility smoke, not a held-out statistical
evaluation. All cases and models are exposed.

## Results

| Model | Target | ROME success | Selected layer by case | Exact / successful | B0 true / successful |
|---|---:|---:|---|---:|---:|
| gpt2-xl | 16 | 2/2 | 16, 16 | 2/2 | 2/2 |
| gpt-j-6b | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| mistral-7b-v0.1 | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| mistral-7b-v0.3 | 6 | 2/2 | 6, 7 | 1/2 | 2/2 |
| deepseek-7b-base | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| falcon-7b | 5 | 2/2 | 9, 9 | 0/2 | 2/2 |
| llama2-7b | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| granite4-micro | 10 | 2/2 | 10, 10 | 2/2 | 2/2 |
| deepseek-r1-llama3-8b | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| olmo-3-1025-7b | 6 | 2/2 | 3, 3 | 0/2 | 2/2 |
| granite-4.1-8b | 16 | 2/2 | 16, 16 | 2/2 | 2/2 |
| ministral-3-8b | 5 | 2/2 | 5, 5 | 2/2 | 2/2 |
| gemma-4-12b | 11 | 1/2 | 11, unavailable | 1/1 | 1/1 |
| **Total** | — | **25/26** | — | **20/25** | **25/25** |

Every successful edit produced a complete minimal capture and analysis. B0
selected the configured changed layer in all 25 available cases. Gemma case 1
failed the ROME edit-success criterion and was retained as unavailable rather
than being counted as a detector result.

The two Falcon misses reproduce the known N=50 failure without a family
exception. The two OLMo misses are a newly observed localizer limitation and
need a larger, separately designed evaluation before any generalization
claim. The Mistral-v0.3 miss was adjacent to the target.

## Excluded covariance files

Two remote files were not exact current-main dependencies and were not used:

- OPT-6.7B: current main selects layer 15; only a layer-14 matrix exists.
- Qwen3-8B: current main selects layer 7; only a 5,000-sample layer-10 matrix
  exists.

No covariance was renamed, substituted across layers, or recalculated.


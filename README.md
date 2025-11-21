# Reimagined Framework

## TODO:
- Finalize the ROME main module files
    - Implement saving of the computed model
    - Implement the main file
    - (Optional) Implement the console UI
- Refactor the weight intervention module
    - Move hyperparams into config
- Refactor the handler class
    - Remove unnecessary code
    - Account for the multitoken targets

- Alter the original approach to minimize the increased cosine distance between the edited vectors (in case of precomputed covariance matrix C, implement this in the v* optim)
- Add autoselect of the most relevant layer for information storage
- Move causal trace support functions into common.py
- Create a handler template
- Implement more handlers for modern models
- Add precomputable params caching
- Optimize the cuda memory usage
- Detection idea - ROME perhaps lowers the cost of adversarial attacks?
- Detection of knowledge conflicts - linear probing
- Add precision reduction into config

## Changes to the original ROME implementation & ambiguities in the paper
- The layer selection for the causal tracing
- The subject tokenation problems ("Rome" vs " Rome")
- The multi-token subject prediction in causal tracing
- The delta matrix magnitude regularization/normalization in weight intervention

## Ideas:
- Covariance matrix generation from random sequence of tokens
- Evaluation of the wikipedia dataset performance for the covariance matrix generation
- Sampling for ROME - how does ROME performace persists when sampling is present

## Models roadmap
---

| Supported Models  | Causal Trace       | Weight intervention | Notes |
|-------------------|--------------------|---------------------|-------|
| gpt2-medium       | :heavy_check_mark: | :heavy_check_mark:  |       |
| gpt2-large        | :heavy_check_mark: | :heavy_check_mark:  |       |
| gpt2-xl           | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-0.6b        | :heavy_check_mark: | :heavy_check_mark:  |       |
| qwen3-1.7b        | :heavy_check_mark: |                     |       |
| qwen3-4b          | :heavy_check_mark: |                     |       |
| qwen3-8b          | :heavy_check_mark: |                     |       |
| granite4-micro    | :heavy_check_mark: |                     |       |

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `2`           | Resource already exists   | Trying to create a resource that already exists.                            |
| `-1`          | Unknown           | An unknow error. Contact the developer with instruction to reproduce the behavior.  |

---

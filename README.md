# Reimagined Framework

## TODO:
- Finalize the ROME main module files
    - Implement saving of the computed model
    - Implement the main file
    - Implement the console UI
- Refactor the weight intervention module
    - Move hyperparams into config
    - Remove duplicit functions and move them into utils
    - Add progress bar for covariance matrix computation
    - Add TensorBoard
- Refactor the handler class
    - Remove unnecessary code
    - Parallelize?
    - Add the MLP layer name/location inside of the model
    - Reimplement as much as possible to utilize torch.nn.Module hooks
- Add autoselect of the most relevant layer for information storage
- Move causal trace support functions into common.py
- Create a handler template
- Implement more handlers for modern models
- Add precomputable params caching
- Alter the original approach to minimize the increased cosine distance between the edited vectors (in case of precomputed covariance matrix C, implement this in the v* optim)
- Optimize the cuda memory usage
- Detection idea - ROME perhaps lowers the cost of adversarial attacks?

## Changes to the original ROME implementation & ambiguities in the paper
- The layer selection for the causal tracing
- The subject tokenation problems ("Rome" vs " Rome")
- The multi-token subject prediction in causal tracing
- The delta matrix magnitude regularization/normalization in weight intervention

## Models roadmap
---

| Model Family (type) | Handler Class     | Example Tested Models                              | Notes                        |
|---------------------|-------------------|----------------------------------------------------|------------------------------|
| gpt2                | GPT2Handler       | <ul><li>[x] gpt2-medium</li><li>[x] gpt2-large</li><li>[x] gpt2-xl</li></ul>                   | All GPT-2 variants supported |
| llama               | LlamaHandler      | <ul><li>[ ] microsoft/Phi-3-mini-4k-instruct</li></ul>                   | Includes Llama, Phi-3, etc.  |
| gpt-j               | GPTJHandler       | <ul><li>[ ] EleutherAI/gpt-j-6b</li></ul>                   | Includes GPT-J, GPT-J-6B, etc.  |

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `2`           | Resource already exists   | Trying to create a resource that already exists.                            |
| `-1`          | Unknown           | An unknow error. Contact the developer with instruction to reproduce the behavior.  |

---
# Reimagined Framework

## TODO:
- reimplement handlers -- separate the model handler from other stuff (tokenizer, dataset, ...)
- implement more handlers for modern models
- make handlers a standalone module within the framework

## Models roadmap
---

| Model Family (type) | Handler Class     | Example Tested Models                              | Notes                        |
|---------------------|-------------------|----------------------------------------------------|------------------------------|
| gpt2                | GPT2Handler       | <ul><li>[x] gpt2-medium</li><li>[x] gpt2-large</li><li>[x] gpt2-xl</li></ul>                   | All GPT-2 variants supported |
| llama               | LlamaHandler      | <ul><li>[x] microsoft/Phi-3-mini-4k-instruct</li></ul>                   | Includes Llama, Phi-3, etc.  |

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `2`           | Resource exists   | Trying to create a resource that already exists.
| `-1`          | Unknown           | An unknow error. Contact the developer with instruction to reproduce the behavior.  |

---
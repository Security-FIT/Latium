# Reimagined Framework

## Implemented model families

---

| Model Family (type) | Handler Class     | Example Tested Models                              | Notes                        |
|---------------------|-------------------|----------------------------------------------------|------------------------------|
| gpt2                | GPT2Handler       | gpt2-medium                                        | All GPT-2 variants supported |
| llama               | LlamaHandler      | microsoft/Phi-3-mini-4k-instruct                   | Includes Llama, Phi-3, etc.  |

---

## Error codes:

---

| Error code    | Name of the error | Description                                                                         |
|---------------|-------------------|-------------------------------------------------------------------------------------|
| `1`           | Help              | Help invoked. Typically caused by incorrect script usage.                           |
| `-1`          | Unknown           | An unknow error. Contact the developer with instruction to reproduce the behavior.  |

---
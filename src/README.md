# Source Tree

`src/` is the application package. The normal entrypoint is:

```bash
python3 -m src <command> [hydra-overrides...]
```

Runtime flow:

```text
main.py -> commands.py -> command_handlers/ -> domain package
```

## Main Areas

| Path | Owns |
|---|---|
| `main.py`, `command_aliases.py`, `commands.py` | CLI aliases and Hydra dispatch. |
| `command_handlers/` | Command handlers that call domain runtimes. |
| `config/` | Hydra config groups. |
| `editing/` | Edit method protocol and registry. |
| `structural/` | Model execution, capture, model-free analysis, detectors. |
| `results/` | Artifact IDs, manifest writes, cache invalidation. |
| `graphs/` | Graph/render artifact production. |
| `causal_trace/` | Standard and alternative causal tracing workflows. |
| `common/` | Shared loading, config, IO, linalg, plotting helpers. |
| `evaluation/` | CounterFact and edit evaluation helpers. |
| `rome/`, `handlers/` | ROME implementation and model handler glue. |

## Adding A Command

1. Add a command config under `src/config/command/`.
2. Add or reuse a handler in `src/command_handlers/`.
3. Register the command in `src/commands.py` or `src/command_handlers/operations.py`.
4. If it needs a top-level alias, update `COMMAND_OVERRIDE_MAP` in `src/main.py`.
5. Add a smoke test or CLI help check.

Prefer putting domain logic in a domain package, not directly in the command
handler.

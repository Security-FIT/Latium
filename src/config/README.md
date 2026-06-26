# Config

Hydra starts from `src/config/latium.yaml`.

Use nested overrides:

```bash
python3 -m src structural plan \
  'structural.run.models=[gpt2-large]' \
  structural.run.n_tests=5
```

## Important Groups

| Group | Purpose |
|---|---|
| `command/` | Command entry configs. |
| `runtime/` | Seed, HF token, logging, cache behavior. |
| `model_base/` | Shared model defaults. |
| `model/` | Concrete model definitions and layer templates. |
| `model_fleet/` | Named model collections. |
| `dataset_facts/`, `dataset_sm/` | Dataset configs. |
| `edit_method/` | Editing method registry entries. |
| `structural/` | Structural run/capture/analysis/render config. |
| `graphs/` | Renderer presets. |

## Adding Config

- Add a YAML file in the relevant group.
- Keep public names stable and lowercase with hyphens.
- Do not add flat compatibility keys for new structural settings.
- If a config changes artifact content, include it in the relevant config hash.

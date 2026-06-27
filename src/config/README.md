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
| `runtime/` | HF token and logging toggles. |
| `model_base/` | Shared model defaults. |
| `model/` | Concrete model definitions and layer templates. |
| `model_fleet/` | Named model collections. |
| `dataset_facts/`, `dataset_sm/` | Dataset configs. |
| `edit_method/` | Editing method registry entries. |
| `structural/` | Structural run/capture/analysis/render config. |
| `graphs/` | Graph render command config (preset, style, renderer options). |

Common structural overrides:

```bash
structural.capture.profile=paper
structural.capture.matrix_features.feature_set=paper
structural.render.renderer_preset=structural-paper
graphs.renderer_preset=structural-paper
graphs.renderers.structural-artifact-grid.formats='[png,pdf,json]'
```

Use `structural.capture.profile=spectral` for spectral-only primitives,
`structural.capture.profile=matrix` for matrix-features only, and
`structural.capture.profile=paper` for the paper graph/analysis path.

## Adding Config

- Add a YAML file in the relevant group.
- Keep public names stable and lowercase with hyphens.
- Do not add flat compatibility keys for new structural settings.
- If a config changes artifact content, include it in the relevant config hash.

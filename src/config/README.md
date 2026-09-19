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
| `pipeline/` | Cross-process workflow defaults used by cluster launchers. |

Common structural overrides:

```bash
python3 -m src structural plan structural=ccs-report 'structural.run.models=[qwen3-4b]'
python3 -m src structural run structural=ccs-report 'structural.run.models=[qwen3-4b]'
python3 -m src graphs run analysis_out/run-id
```

Use `structural.capture.profile=spectral` for spectral-only primitives,
`structural.capture.profile=matrix` for matrix-features only, and
`structural.capture.profile=paper` to explicitly request both paper captures.
The `ccs-report` workflow enables the paper analysis and all five report
renderers, and capture planning adds their required inputs. The graph command
defaults to the same bundle; `graphs.renderer_preset=paper` is retired so it
cannot produce only aggregate summaries by mistake.
For an end-to-end `structural run`, selecting an analysis preset automatically
adds its required captures and matrix columns, so a detector does not require a
matching capture-profile override. Every analysis ID, including
`ccs-composite`, `blind`, and `rank1-blind`, is a single-method preset.

## Adding Config

- Add a YAML file in the relevant group.
- Keep public names stable and lowercase with hyphens.
- Do not add flat compatibility keys for new structural settings.
- If a config changes artifact content, include it in the relevant config hash.

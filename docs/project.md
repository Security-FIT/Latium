# Project Structure

Latium is a Hydra-driven research pipeline for editing model knowledge, saving
structural measurements, replaying model-free analyses, and rendering outputs
from manifest-owned artifacts.

## Runtime Flow

```text
python -m src
  -> src/main.py
  -> src/commands.py
  -> src/command_handlers/
  -> domain runtime
  -> src/results manifest artifacts
```

The structural workflow is staged:

```text
structural capture: load model, run edits, save execution/capture artifacts
structural analyze: load saved artifacts, run model-free analyses
graphs run: load analysis artifacts, write graph/render artifacts
```

Use nested Hydra keys only:

```bash
python -m src structural plan \
  'structural.run.models=[gpt2-large]' \
  structural.run.n_tests=5
```

## Important Packages

| Path | Purpose |
|---|---|
| `src/main.py` | CLI entrypoint and shortcut handling. |
| `src/commands.py` | Thin Hydra command dispatcher. |
| `src/command_handlers/` | Domain-specific command handlers. |
| `src/common/` | Shared config, loading, linalg, plotting, and IO helpers. |
| `src/editing/` | Editing method interface, registry, and ROME adapter. |
| `src/evaluation/` | CounterFact and edit evaluation utilities. |
| `src/results/` | Artifact IDs, paths, manifest reads/writes, hashes. |
| `src/structural/execution/` | Model loading, case selection, edit execution, weight extraction. |
| `src/structural/capture/` | Capture registry, producers, baseline/method artifact writing. |
| `src/structural/analysis/` | Analysis registry, runtime, case helpers, detector/study runners. |
| `src/structural/detectors/` | Shared detector cores and model-resident adapters. |
| `src/graphs/` | Renderer registry, runtime, and renderers. |
| `src/config/` | Hydra app, command, runtime, model, dataset, and method configs. |

Package-local README files live under the main extension points:

- `src/README.md`: source tree and command wiring.
- `src/config/README.md`: Hydra config groups and override rules.
- `src/editing/README.md`: edit method contract and registration.
- `src/structural/README.md`: structural capture, analysis, and detector flow.
- `src/graphs/README.md`: renderer contract.
- `src/results/README.md`: artifact manifest rules.
- `src/causal_trace/README.md`: early-site causal tracing.

## Notebooks

Notebooks are for visual inspection, not production runs.

- `notebooks/causal-tracing.ipynb`: canonical standalone early-site causal
  tracing reference; it imports no Latium modules and mirrors the pipeline's
  intervention, calibration, and held-out selector.
- `notebooks/analysis.ipynb`: thin wrapper around graph rendering for an
  existing run root.

## Config Layout

Hydra starts from `src/config/latium.yaml`.

Common groups:

- `command/`: command entry configs, such as `structural/run.yaml`.
- `runtime/default.yaml`: seed, HF token, logging, cache behavior.
- `model/`: concrete model configs.
- `model_base/`: shared model defaults.
- `model_fleet/`: named model collections.
- `dataset_facts/`: CounterFact-style fact datasets.
- `edit_method/`: editing method registry entries.
- `structural/default.yaml`: structural run/capture/analysis/render defaults.
- `graphs/default.yaml`: renderer defaults.
- `prefix_experiment/default.yaml`, `rome_benchmark/default.yaml`, etc.

Structural config sections:

- `structural.run`: models, edit methods, case ranges, output, force.
- `structural.capture`: capture profile and explicit enable/disable lists.
- `structural.analysis`: analysis preset, variants, per-method config.
- `structural.render`: renderer preset and enable/disable lists.
- `structural.analyze`: existing run root for analysis-only commands.
- `structural.validate_cov`: covariance validation behavior.

## Adding An Editing Method

Implement `src.editing.base.EditMethod`:

```python
class MyMethod:
    identifier = "my-method"
    description = "..."

    def apply(self, handler, case):
        return EditOutcome(
            success=True,
            metrics={},
            metadata={},
            modified_weights={"proj": (handler._layer,)},
            restorations={...},
        )

    def evaluate(self, handler, case, outcome):
        return {"efficacy_score": 1.0}
```

Then add `src/config/edit_method/my-method.yaml`:

```yaml
identifier: my-method
description: My edit method.
factory: src.editing.my_method:MyMethod
model_families:
  - all
```

Important contracts:

- `apply()` runs while the model is loaded and may mutate weights.
- `EditOutcome.restorations` must contain tensors needed to restore mutated
  modules.
- `modified_weights` tells structural capture which families/layers changed.
- `probe_vector` is required only by captures such as `bottom-rank-tokens`.
- The structural runtime restores the handler after each case.

Add tests around the adapter and registry loading.

## Adding A Capture Producer

Capture producers are model-dependent and run during `structural capture`.

1. Add a function in `src/structural/capture/producers.py`.
2. Register it in `src/structural/capture/registry.py`.
3. Add it to a capture profile if it should be selected by default.
4. If it has options, add relevant config hashing in
   `src/structural/capture/artifacts.py`.

Producer shape:

```python
def capture_my_features(context: CaptureContext) -> dict[str, Any]:
    return {
        "mode": "baseline" if context.is_baseline else "patch",
        "families": {"proj": {...}},
    }
```

Rules:

- Return JSON-serializable data.
- Baseline captures should include all reusable layers.
- Method captures may write patches or full profiles depending on downstream
  consumers. `matrix-features` writes all layers for edited cases because
  composite and paper graphs compare the full post-edit depth profile.
- Use `context.changed_layers(family, layers)` for patch selection.
- Declare `requires_probe=True` if the capture needs an edit probe vector.
- Hydra-owned capture options belong under `structural.capture`. Matrix feature
  columns are selected with `structural.capture.matrix_features.feature_set` or
  `structural.capture.matrix_features.features`.

## Adding An Analysis Or Artifact Study

Analyses are model-free and consume saved captures. Detection methods and
artifact studies use the same runtime; the registry `category` decides where
the artifact is stored.

1. Implement a runner in `src/structural/analysis/detector_methods.py` or
   `src/structural/analysis/studies.py`.
2. Register it in `src/structural/analysis/registry.py`.
3. List required capture IDs in `required_captures`.
4. Validate required matrix feature columns with `require_matrix_features` when
   consuming `matrix-features`.
5. Add default config under `structural.analysis.methods.<analysis-id>` if the
   analysis has config fields.
6. Add tests for unavailable inputs and happy-path outputs.

Simple study shape:

```python
def analyze_my_study(context: AnalysisContext) -> dict[str, Any]:
    def analyze(data, case_id):
        return {"anomalous_layer": None, "detection_score": 0.0}

    return run_case_analysis(context, "matrix-features", analyze)
```

Registry fields:

- `identifier`: public analysis ID.
- `category`: `detection` or `artifact-study`.
- `runner`: `module:function` path.
- `required_captures`: capture IDs that must exist.
- `config_fields`: accepted config keys.
- `variant_fields`: maps `AnalysisVariantConfig` fields into method config.
- `model_families`: optional support filter, such as `gpt` or `non-gpt`.

Use `AnalysisUnavailableError` when a case cannot be analyzed because the saved
capture is insufficient and recapture is required.

## Adding A Renderer

Renderers consume completed analysis artifacts and write files under
`<run-root>/graphs/<renderer-id>/`.

1. Add a renderer function in `src/graphs/renderers.py` or a focused module
   under `src/graphs/structural/`.
2. Register it in `src/graphs/registry.py` with declared executions, captures,
   and analyses.
3. Add it to a renderer preset if useful.
4. Test it in `tests/test_graph_renderers.py`.

Renderer shape:

```python
from src.graphs.context import RenderContext


def render_my_output(context: RenderContext) -> list[str]:
    output_dir = context.output_dir
    analyses = context.flat_analyses
    output_dir.mkdir(parents=True, exist_ok=True)
    path = write_json(output_dir / "my-output.json", {"analyses": analyses})
    return [str(path)]
```

The graph runtime records returned paths in a render artifact and hashes the
declared input artifacts as inputs. New renderer options are configured through
Hydra under `graphs.renderers.<renderer-id>`.

## Artifact Rules

Artifacts are owned by `src/results/`:

- Execution artifacts record edit status and evaluation results.
- Capture artifacts record baseline measurements or edited-state patches.
- Analysis artifacts record model-free detector/study results.
- Render artifacts record generated graph/data outputs.

Do not infer relationships from filenames. Use `RunArtifactReader`,
`ArtifactWriter`, and manifest records.

Config hashes decide whether an artifact is current. Input content hashes drive
transitive invalidation. Use `force=true` to rewrite current artifacts.

## Testing Checklist

For most changes, run the narrow tests first:

```bash
python3 -m pytest tests/test_registry.py tests/test_structural_planning.py
python3 -m pytest tests/test_model_cases.py tests/test_analysis_runtime.py
python3 -m pytest tests/test_graph_renderers.py tests/test_results_artifacts.py
```

Useful smoke checks:

```bash
python3 -m src command=methods
python3 -m src structural plan 'structural.run.models=[gpt2-large]' structural.run.n_tests=5
```

Before merging structural changes, search for stale public names and deleted
facades:

```bash
rg "src\.structural\.(posthoc|model_cases|analysis_spectral|cases|runs)"
```

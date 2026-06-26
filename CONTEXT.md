# Latium Architecture Context

## Purpose

Latium separates model-dependent knowledge editing from repeatable,
model-free analysis. Expensive model work happens during capture; detectors,
artifact studies, and visualization operate on saved JSON measurements.

## Domain Vocabulary

- **Edit method**: applies one knowledge edit and evaluates its result.
- **Execution artifact**: case inputs, edit status, evaluation metrics, changed
  weight families, and target-layer metadata.
- **Capture producer**: computes reusable measurements while weights are
  available.
- **Capture artifact**: baseline data or an edited-state patch.
- **Analysis method**: consumes declared captures without loading a model.
- **Renderer**: consumes manifest-indexed analysis artifacts.
- **Run root**: one directory owned by one manifest.
- **Analysis variant**: one normalized analysis configuration.
- **Artifact ID**: stable logical identity independent of filesystem path.

## Module Boundaries

- `src/editing/`: edit method interface, adapters, and registry.
- `src/structural/execution/`: model lifecycle, edit execution, structural
  CounterFact case loading, covariance lookup, and weight/model helpers.
- `src/structural/capture/`: capture metadata, capture producers, baseline
  capture, execution/capture configs, manifest inputs, and artifact writes.
- `src/structural/analysis/`: analysis registry, artifact loading,
  materialization, selective reruns, persistence, case helpers, trim policy,
  detector analyses, and artifact studies.
- `src/structural/detectors/`: reusable detector implementations, resident
  adapters, matrix profiles, local scores, and spectral primitives.
- `src/common/linalg.py`: shared CUDA-aware SVD, VRAM, and device helpers.
- `src/common/loading.py`: model, dataset, and small inference helpers.
- `src/common/model_config.py`: model YAML/fleet config resolution.
- `src/evaluation/counterfact.py`: historical CounterFact evaluation utilities.
- `src/model_config.py`: compatibility facade for historical model config
  imports; repo code should use `src/common/model_config.py`.
- `src/utils.py`: compatibility adapter for external historical imports; repo
  code should import from named modules.
- `src/registry.py`: shared registry primitives, lazy object loading, preset
  resolution, and model-family support.
- `src/results/`: IDs, paths, hashing, atomic writes, manifest ownership, and
  artifact reads.
- `src/graphs/`: renderer registry and runtime.
- `src/main.py`: public Hydra-first command surface and top-level command
  aliases.
- `src/command_aliases.py`: shortcut-to-Hydra override translation for
  structural, graph, and prefix experiment aliases.
- `src/commands.py`: thin Hydra command dispatcher.
- `src/command_handlers/`: focused Hydra command handlers by domain.
- `src/structural/config.py`: normalized structural runtime config and
  structural value objects.
- `src/structural/hydra_config.py`: compact Hydra-to-structural-config
  adapter; analysis-variant sweep parsing belongs in structural planning.
- `src/config/latium.yaml`: Hydra app defaults for the command surface.
- `src/runtime.py`: typed runtime options derived from Hydra config for
  non-protected code.

## Protected ROME Boundary

`src/rome/`, `src/handlers/rome.py`, `src/config/config.yaml`, and
`src/config/model/` keep ROME math and model defaults stable. `RomeEditMethod`
adapts the structural runtime to that code without changing ROME equations,
optimization, insertion, or evaluation semantics. These paths must match
`main` exactly. New runtime switches should use Hydra config in non-protected
code; avoid OS environment variables for runtime configuration.

## Runtime Contracts

The model runtime calls an `EditMethod`. Each method reports changed weight
families and restoration tensors.

Capture producers receive `CaptureContext` and return JSON-serializable data.
The `spectral` profile supplies the common detector inputs.

Model-resident detectors and artifact-only analyses should share one detector
implementation where a capture can represent the same inputs. Resident classes
are adapters for experiments that still have weights loaded; artifact analyses
consume manifest captures. Shared detector cores live under
`src/structural/detectors/`.

Analysis methods receive `AnalysisContext` and declare required capture IDs.
Missing inputs produce an `unavailable` result with a recapture reason.

The manifest owns artifact discovery and dependencies. Config hashes determine
whether work is current. Input content hashes drive transitive invalidation.
Manifest writes use one lock and artifact writes use atomic replacement.

Renderers consume manifest records and do not infer relationships from
filenames.

Registries share `RegistryEntry`-shaped metadata and optional
`model_families`. Registry entries may be loaded from YAML when extension by
configuration matters, or defined inline when the implementation surface is
owned by code; both styles should use the shared registry helpers.

## Data Strategy

Latium stores numerical measurements, not full model checkpoints. Baseline
captures are shared per plan. Method captures overlay edited-state patches on
the baseline. Analysis variants reuse those captures and do not rerun model
work.

## Extension Rules

New components must:

1. use a Hydra-backed registry entry as their public selection surface;
2. declare inputs instead of adding orchestration branches;
3. emit JSON-serializable data;
4. use stable artifact IDs and manifest input references;
5. report unsupported work as `unavailable`.

Editing methods are isolated by module and registered through
`src/config/edit_method/*.yaml`.

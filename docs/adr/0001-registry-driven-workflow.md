# ADR 0001: Registry-Driven Workflow

## Status

Accepted

## Decision

Use four lazy internal registries:

- Edit methods
- Capture producers
- Analysis methods
- Renderers

`python -m src` is the public Hydra-first entrypoint. Structural execution is
split into:

- `structural capture` for editing, evaluation, and model measurements;
- `structural analyze` for model-free analysis;
- `graphs run` for visualization;
- `structural run` for capture followed by analysis.

Analysis entries declare required capture IDs. Registry entries use lazy import
paths so planning and command discovery do not load PyTorch or plotting
libraries. Edit-method registry entries are read from
`src/config/edit_method/*.yaml`.

ROME is exposed through an adapter and its protected implementation is not
modified.

## Consequences

New components have a small registration surface and do not add branches to
the benchmark loop. Capture and analysis can evolve independently. The
registries are internal Python APIs backed by repository Hydra config rather
than a dynamic plugin system.

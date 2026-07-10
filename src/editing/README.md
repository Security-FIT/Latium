# Editing

Edit methods implement `EditMethod` from `base.py` and are discovered through
YAML files in `src/config/edit_method/`.

## Add A Method

1. Implement `apply(handler, case) -> EditOutcome`.
2. Implement `evaluate(handler, case, outcome) -> dict`.
3. Add `src/config/edit_method/<method>.yaml`.
4. Add registry/adapter tests.

`EditOutcome.modified_weights` tells structural capture which matrix families
and layers changed. `EditOutcome.restorations` must contain enough state for
the structural runtime to restore the model after each case.

The registry loads factories from `factory: module:ClassName`; keep constructors
argument-free unless the registry is changed deliberately.

Lazy `module:attribute` paths are the registry extension contract. They keep
command discovery from importing model and plotting stacks; in-tree capture,
analysis, and renderer entries use the same contract as external additions.

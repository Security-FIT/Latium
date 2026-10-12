# MetaCentrum runner provenance

The October 2026 integration uses the final own-covariance campaign source
archive, SHA256
`5b8cb2b0b5ff123aab08452a82cccd2d8f71212dcddcc656b5ba308eade969e5`,
as the source for the latest finetune runners, strict loading checks, saved
tokenizer recovery, fixed fact assignments, covariance receipts, and Hydra
path quoting. Earlier deployed checkouts were on `fix/finetune-tokens`,
including commits `77cd18a` and `a9223f4`; the final archive additionally
contained deployment patches. A branch name alone does not identify that code.

The deployed files under `src/structural`, `src/rome`, `src/editing`,
`src/evaluation`, `src/results`, and `src/graphs` matched `simplify-code`
commit `097ce88` byte for byte. This integration retains those files and the
branch's model configurations. Deployment-specific materialized model YAMLs,
account paths, selected checkpoints, old outputs, and batch plans are not
copied into the source tree. All 18 classic families remain selectable.

The retained tracing implementations are `causal-kuba-fix` (corrected
token-by-block restoration) and `causal-trace` (MLP-window tracing). The
historical `causal-kuba` implementation is removed from the supported tree.
Corrected Kuba settings retain the `command.legacy_trace` namespace so
existing experiment arguments remain recognizable.

The production wrappers are adapted to accept ordinary frozen manifests,
configurable covariance sample counts and worker budgets. Report recovery
uses explicit validation even with Python optimization, verifies registered
artifact hashes, rejects nonfinite/incomplete science, and retains original
failure states and exit codes. These adaptations have CPU regression coverage;
the integration itself does not launch new cluster or GPU experiments.

Changed loading or tracing implementations invalidate old run identities.
Use a fresh run root when moving an older campaign to this code; keep original
artifacts for provenance and analyze each frozen cohort separately. Completion
of a workflow and success of a ROME edit remain distinct outcomes.

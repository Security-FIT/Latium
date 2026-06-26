# ADR 0002: Manifest-Owned Artifacts

## Status

Accepted

## Decision

Use one manifest-owned run root containing independently addressable JSON
artifacts:

- Edit executions
- Baseline and per-method captures
- Per-method analyses grouped by category
- Render outputs

Each artifact records its stable ID, kind, producer, run identity, status,
config hash, input artifact hashes, cases, summary, and optional error.

Writes use temporary files followed by atomic replacement. Manifest mutations
use one file lock and reload current state before writing. When an artifact
changes, descendants that reference its previous content hash are removed.
Unrelated artifacts remain reusable.

Baseline captures are stored once per plan. Method captures may store patches
over that baseline. Analyses materialize baseline plus patch without loading a
model. Analysis configurations produce independent artifact identities while
sharing the same captures.

## Consequences

Analyses can be rerun without inference or edit application. Replacing one
artifact does not rewrite unrelated outputs. Consumers use
`RunArtifactReader`, and missing required measurements require recapture.

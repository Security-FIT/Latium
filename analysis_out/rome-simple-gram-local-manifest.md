# Local ROME simple-Gram evidence manifest

Saved from `ubuntu@150.136.40.217` on 2026-07-28.

## Source code

- Branch: `detector-simplification`
- Commit: `88c0f8aac6e636cce33d96a0e81a4ae92354cba0`
- Git bundle: `analysis_out/detector-simplification-88c0f8a.bundle`
- Bundle SHA-256:
  `126a6b7823a3126e26c7717a209901006d60e609b2275a77e8d21a1e03649282`

The Git bundle was verified with `git bundle verify` and records complete
history for the branch.

## Evidence

- Portable archive:
  `analysis_out/rome-simple-gram-evidence-20260728.tar.gz`
- Archive SHA-256:
  `21571e709cb27b630dc2d75563cca1de17c13b3101ec9d7ea3d6980f555ce961`
- Files synchronized and compared against the cluster: 153
- Full-tree SHA-256 comparison result: identical

The unpacked local evidence includes:

- all five N=2 smoke run roots and their smoke ledger;
- all thirteen N=20 raw run roots;
- N=20 driver log, ledger, and offline aggregates;
- all five hard-negative bundles, logs, ledger, and aggregate;
- all five magnitude-generation source bundles.

Important aggregate hashes:

- N=20 all-model localization:
  `f119e82ccba128be21970cf4f730243adc5ddcc9078362a2abd2d5f6f4aefe88`
- Binary hard-negative evaluation:
  `eeb7c5608f0f8971eb47b6229a0e8d140af86dd039c2220fd01eb9125cbf326e`

The human-readable experiment report is
`rome-simple-gram-simplification-report.md` at commit `88c0f8a`.

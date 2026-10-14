# CSF preparation validation — 14 October 2026

The submission branch starts at `d75afbf973a78ffcef0ff059e0f5094d6148bbcf`,
the fetched `origin/main` containing the simplify-code merge. Preparation used
a separate checkout; the existing `feat/paper-improvements` checkout, its tracked
changes and untracked reports/artifacts were left intact. Author names and the
license are retained.

## Local checks passed

- 178 CPU tests passed, including real tiny-checkpoint loading and LoRA merging,
  rejection of incomplete weights, padding/subject tokenization, covariance
  batching and non-padding accumulation, ROME update/restoration in float32 and
  bfloat16, GRAM cache parity, frozen assignments, repair selection, tamper
  detection, and rendering without source-byte changes.
- Ruff, Python compilation, Bash syntax and `git diff --check` passed.
- Frozen manifests, all 1,100 exported CounterFact records, checkpoint/adapter
  pins and the three original external prefix caches were checked. The 747
  own-covariance candidates exactly equal the previous 747 efficacy successes;
  all retain their original assigned fact. All evaluated execution seeds are 0;
  the original cohort sampling seeds and tracing settings remain separate.
- All four requested archive folders were downloaded/read. 38,954 indexed files
  were verified, and 26,711 indexed JSON files parsed. There were no missing files
  or experimental content/configuration hash mismatches.
- Source indices contain four subsequently changed README files and one CSV
  whose CRLF line endings became LF. The CSV hash matches its recorded hash after
  restoring CRLF in memory. These differences are explicitly recorded in
  [artifact-validation.json](artifact-validation.json); source files were not
  rewritten to hide them.
- All selected campaign run manifests, six primary artifacts, identities,
  configuration/content hashes and dependency references were read with the
  cleaned code. The three campaigns rendered to external directories in
  PNG/PDF/SVG, with outcome and layer-statistics tables. Representative linear
  and log profiles were visually inspected.

| Campaign | Evaluated edits/checkpoints | ROME efficacy success | GRAM exact target | Baseline FP |
|---|---:|---:|---:|---:|
| Classic | 18,000 edits / 18 base models | 17,644 | 14,623 | 4 / 18 |
| Base covariance | 790 checkpoints | 747 | 618 | 186 / 790 |
| Own covariance | 744 checkpoints | 736 | 586 | 159 / 744 |

The three own-covariance failed downloads are preserved separately. Efficacy
failures remain in the evaluated populations. Baseline FP is target coincidence
on the unedited checkpoint, using its future execution-configured edit layer.
Classic baseline replicas are provenance, not additional independent controls.

Causal tracing records and unavailable statuses remain in the archive and are
exported by the renderer. Classic tracing has 1,700 profiles across 17 families;
the comparable DeepSeek 7B base reference remains explicitly unavailable.
Checkpoint campaigns retain 766 and 727 accepted pre-ROME traces. No replacement
trace or layer selection was invented.

The local validation environment used Python 3.10.12, torch 2.9.1+cpu,
transformers 4.57.1, datasets 4.4.1, peft 0.21.2, hydra-core 1.3.7, numpy 2.1.2,
matplotlib 3.10.8, pytest 9.1.1 and ruff 0.17.0. It had no CUDA device. This is a
record of local validation, not a claim about the unrecorded campaign environment
or support for every modern model architecture in this older Transformers build.

## Remaining fresh-cluster check

After the user provides the fresh cluster, verify a clean installation and CUDA,
load the pinned GPT-2 Medium adapter and its pinned base, run the two documented
`smoke` commands, and inspect saved covariance, ROME, baseline/edited GRAM and
rendered outputs. Smoke uses one original assigned fact and 32 covariance samples;
it preserves the edit settings and marks the reduced protocol. Record package
versions, device, snapshot pins, tensor hashes, efficacy and exact-layer outcomes.
This GPU/model-download validation has **not** been run locally.

Exact classic replay and newly computed paper base covariance still need the
original base-weight commit SHAs: the archived classic/base-covariance execution
records did not record them. The runner requires those pins and never resolves a
current Hub tip as a substitute. Adapter base revisions are recorded and permit
the separate smoke validation. The original own-covariance tensors were deleted;
their computation/deletion receipts survive. The archived covariance corpus
revision and package environment were not pinned, so bitwise replay cannot be
promised. These gaps do not prevent saved-artifact analysis or rendering.

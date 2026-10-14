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

## Fresh-cluster checks completed

The user-provided host `root@203.57.40.186:10173` was tested on 14 October 2026.
It runs Ubuntu 22.04.5, Python 3.11.11 and an NVIDIA RTX 6000 Ada with 48 GB VRAM
and driver 595.91.07. A fresh HTTPS clone received the exact local CSF commits
through Git bundles. Installation used an isolated venv without replacing system
packages. The CUDA 13 PyTorch build, CUDA matrix operations, BF16 availability,
and package dependency checks passed. The environment uses torch 2.14.1,
transformers 5.19.0, datasets 5.1.0 and peft 0.21.2; the full package lock is
[requirements-cluster.txt](../requirements-cluster.txt).

Eight smoke workflows completed: base and own covariance for the frozen GPT-2
Medium DPO adapter, MiniLLM GPT-2 Large full checkpoint and Vietnamese-legal
Qwen3.5 4B adapter, plus the first fact in each classic GPT-2 Medium batch.
All eight achieved ROME efficacy 1.0 and exact configured-layer GRAM matching;
all baseline FP indicators were false. Each preserved its original fact,
checkpoint/adapter revision, seed and ROME settings. Fresh own covariance was
computed on the actual finetuned/merged checkpoint; base covariance was computed
on the pinned base. Matrices were finite float32, their hashes matched their
receipts, and base/own hashes differed for all three families.

A further two edits ran on one resident GPU model. After both edits, original
projection bytes, the parameter object and its frozen flag were restored exactly.
Their restored SHA256 hashes were identical. Rendering left source JSON bytes
unchanged. Saved primary artifacts passed strict identity, configuration/content
hash and dependency checks. The initial 178-test suite passed; after the download
regression test was added, the updated suite passed 179 tests. Ruff, compilation,
Bash syntax and frozen-input checks passed as well.

The fresh installation exposed the old GPT-2 Hub alias failing at Xet downloads;
the downloader now resolves the canonical repository at the requested immutable
revision and verifies that revision. Visual inspection exposed long per-run
titles overlapping; complete labels now wrap. The export audit corrected unique
checkpoint counts for split classic batches and retained supplied base revisions
in aggregate rows. Regression checks cover these cases.

Commands, exact models, fact IDs and metrics are in [command.md](../command.md).
The durable [cluster-smoke-results.json](cluster-smoke-results.json) contains
provenance, covariance identities/hashes, metrics, source revisions, GPU restoration
evidence and figure hashes. PNG files were decoded, SVG parsed, PDF pages/text
read with a strict parser, and representative linear/log figures inspected.
In total, 48 primary smoke artifacts and 102 figure files passed these checks.

Smoke covariance used **32 eligible texts**, not the paper protocol's 100,000.
This validates execution and saved outputs, not paper accuracy or bitwise replay.
MiniLLM neighborhood scores were 0.1 and the classic smoke scores were 0.4/0.5;
these are retained outcomes, separate from efficacy success. Qwen3.5 used working
PyTorch reference convolution/linear-attention kernels; optional accelerated
packages were absent. Only these three families were tested on this host.

Exact classic replay and newly computed paper base covariance still need the
original base-weight commit SHAs: the archived classic/base-covariance execution
records did not record them. The runner requires those pins and never resolves a
current Hub tip as a substitute. Adapter base revisions are recorded and permit
the separate smoke validation. The original own-covariance tensors were deleted;
their computation/deletion receipts survive. The archived covariance corpus
revision and package environment were not pinned, so bitwise replay cannot be
promised. These gaps do not prevent saved-artifact analysis or rendering.

## Read-only provenance recovery on kubapc

At the user's direction, `/data/CSF-final-papers`,
`/data/Latium-artefacts-merged/classic-gram`, the two checkpoint archive roots,
the Latium model directories, three Hugging Face caches and six covariance
directories were inspected over SSH on `metju@kubapc`. The evidence is saved in
[kubapc-provenance.json](kubapc-provenance.json), including cache revisions,
snapshot file inventories, config hashes and original log references.

Ten campaign-family base repositories have surviving immutable cache revisions.
The converted local model configs have no recorded commit hash or download
metadata. The 40 classic support logs contain no immutable model download URL;
they identify original covariance storage under the MetaCentrum campaign path.
The surviving covariance candidates on kubapc predate the October campaigns,
and several have different layers or sample counts. No evidence recovered here
ties these cache revisions or tensors to the exact CSF execution inputs.
They remain explicitly unconfirmed and do not replace the missing campaign pins
or original covariance. No remote files, environments or GPU jobs were changed.
This inspection was separate from the completed fresh-cluster smoke validation
above, which used recorded adapter-base pins without a new historical revision map.

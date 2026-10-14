# Latium — CSF submission

Latium evaluates ROME model editing and normal GRAM layer localization. It is
developed by the Security@FIT group in collaboration with Red Hat Research.

Main contributors: **Matej Olexa** and **Jakub Res**. Names and attribution are
preserved; submission anonymization is handled separately.

## Installation and authentication

Use Linux, Bash and Python 3.11 or newer on a GPU machine. An ordinary SSH
session works; no MetaCentrum paths, PBS installation or scheduler is required.
PyTorch must support the machine's CUDA driver and GPU. Install its appropriate
build before installing the remaining requirements if your cluster provides one.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # local checks
python -m pytest -q
python -m ruff check src tests scripts
```

Public checkpoints need no token. For gated/private models, obtain access to
the original repositories and set `HF_TOKEN` in the SSH environment, or use
`hf auth login`. Do not put tokens in commands committed to this repository.
`HF_HOME` controls the Hugging Face cache. `LATIUM_COUNTERFACT_CACHE` may point
to an existing `datasets.save_to_disk` CounterFact cache; selected case IDs and
content hashes are checked even when its Arrow representation differs.

Install `requirements-monitoring.txt` only to enable optional W&B tracking.
Tracking is disabled by default. `LATIUM_PYTHON` selects the interpreter used by
the Bash entry point.

[command.md](command.md) records the fresh-cluster clone, installation,
multi-model smoke tests, rendering and validation commands. The tested Python
3.11/CUDA 13 environment is pinned in `requirements-cluster.txt`; other drivers
may require another PyTorch build.

## Frozen experimental inputs

The authoritative archive is
[LLM Security on Nextcloud](https://nextcloud.fit.vutbr.cz/s/FcezyKRbbwdAbtX).
Use `classic-gram-merged`, `finetune-no-cov`, `finetune-covariance`, and the
`CSF-final-papers` export. At the time of this preparation, the live share places
`finetune-no-cov` inside `ROME - CSF` and calls the paper export `CSF-graphs`.
The downloader maps those locations to the four requested local names.

| Campaign | Frozen population and protocol |
|---|---|
| `classic` | 18 base models, the same 1,000 CounterFact facts; one conceptual unedited control per model and 1,000 independent ROME edits with edited GRAM. Original first/second 500 batches and the 24 repaired Qwen3.5 cases remain identified. |
| `finetune-base` | 790 clean evaluated workflows from the original candidates, up to 100 per family. Each actual finetuned/merged checkpoint has its own baseline, one assigned fact/edit, and edited GRAM; covariance comes from its family base model. |
| `finetune-own` | The frozen 747 checkpoints that previously passed ROME efficacy; 744 evaluated and three recorded download failures. Each evaluated checkpoint uses freshly computed covariance from that finetuned/merged checkpoint, with 100,000 eligible text samples. |

`manifests/CSF-final/` contains the original 1,000- and 100-fact selection
manifests, the separate 900-fact checkpoint tracing pool, exact checkpoint/fact
assignments, pinned checkpoint and adapter-base revisions, original resolved
checkpoint configs, recorded execution seeds, and the three external prefix
caches verified against their recorded hashes. Convenience exports include
actual CounterFact records, dataset indices and model IDs. `checksums.json`
protects the packaged inputs. The original 100-fact manifest was copied from
`CSF-final-papers-preview/manifests/finetuned-own-covariance/rome-facts-100.json`.

These files are the selections. Do not resample facts, rediscover models, infer
layers from current defaults, or mix the two checkpoint populations. Every
checkpoint baseline is measured on its actual full or safely merged LoRA
checkpoint. ROME restores the original projection weights between edits,
including failures. Self/template prefix generation and caching remain in the
ROME runtime; the separate prefix-test study has been removed.

The saved classic and base-covariance records do **not** identify the commit
SHAs of their original base-weight snapshots. Checkpoint and adapter-base pins
are retained. Exact classic reruns and newly computed base covariance require
the original base revision map (`{"repository/id": "40-character-commit-sha"}`)
through `--base-revisions`. The runner refuses to substitute current Hub tips.
The read-only [kubapc provenance audit](docs/kubapc-provenance.json) records
surviving cache revisions and older covariance files. Their connection to the
October campaigns is unconfirmed, so they are not used as campaign pins.
Original own-covariance tensors were deleted after processing; receipts survive,
and reruns recompute them. A covariance corpus revision/package environment was
not pinned in the archived records, so bitwise numerical replay is not claimed.

## Reproduction commands

Inspect the frozen selections without loading a model:

```bash
bash scripts/csf.sh validate
bash scripts/csf.sh plan classic --family gpt2-large
bash scripts/csf.sh plan finetune-base --family gpt2-medium
bash scripts/csf.sh plan finetune-own --family gpt2-medium
```

Run the three campaigns with the original base snapshot revisions:

```bash
bash scripts/csf.sh run classic --base-revisions original-base-revisions.json --output results/classic
bash scripts/csf.sh run finetune-base --base-revisions original-base-revisions.json --output results/basecov
bash scripts/csf.sh run finetune-own --output results/owncov
bash scripts/csf.sh render-new classic --runs results/classic --output results/new-classic-graphs
bash scripts/csf.sh render-new finetune-base --runs results/basecov --output results/new-base-graphs
bash scripts/csf.sh render-new finetune-own --runs results/owncov --output results/new-own-graphs
```

If the original 100,000-sample base tensors survive, provide their directory
using `--base-covariance-dir`. Filenames must match the archived family, edit
layer and sample count. New base covariance is computed on the pinned base
checkpoint and reused with a checked receipt; own covariance is freshly computed
on each actual finetuned/merged checkpoint. The protocol concatenates Wikitext
103 raw train/test/validation, excludes short texts, accumulates uncentered MLP
input second moments over non-padding tokens in float32, regularizes with
`1e-5 I`, and stores the inverse. The 100,000 samples are eligible **texts**,
not tokens. Archived batching/length settings and configured target layers are
retained in the frozen configs.

Use `--family`, `--checkpoint`, `--limit-checkpoints` or `--limit-facts` for
explicit validation subsets or manually partitioned workers. These take fixed
entries/positions and mark outputs as subsets. Use distinct output directories
for concurrent SSH or scheduler workers. Full reruns execute the original batch
boundaries; Qwen's repair cohort remains separate. Failed downloads are retained
as recorded statuses, and efficacy failures remain evaluated outcomes.

For the fresh-cluster check, the following shared adapter is pinned in both
checkpoint campaigns. Smoke runs use one assigned fact and 32 covariance samples,
retain the original edit settings, and clearly mark the reduced protocol:

```bash
bash scripts/csf.sh smoke finetune-base --family gpt2-medium \
  --checkpoint AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT --output results/smoke-base
bash scripts/csf.sh smoke finetune-own --family gpt2-medium \
  --checkpoint AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT --output results/smoke-own
```

The base smoke uses the adapter's recorded base revision; it does not attest to
the unrecorded classic covariance-base revision. Smoke outputs are separate
from paper populations. Fresh-cluster validation passed on GPT-2 Medium, GPT-2
Large and Qwen3.5 4B: both covariance modes, full/merged adapter loading, ROME,
baseline/edited GRAM, independent weight restoration and rendering. These checks
use 32 covariance texts and do not replace the full paper campaigns.

## Saved-artifact validation and rendering

These commands read archived results and write exports to separate directories:

```bash
bash scripts/csf.sh download-archive --output artifacts
bash scripts/csf.sh validate --archive artifacts/classic-gram-merged
bash scripts/csf.sh validate --archive artifacts/finetune-no-cov
bash scripts/csf.sh validate --archive artifacts/finetune-covariance
bash scripts/csf.sh validate --archive artifacts/CSF-final-papers
bash scripts/csf.sh render classic --archive artifacts/classic-gram-merged --output results/render-classic
bash scripts/csf.sh render finetune-base --archive artifacts/finetune-no-cov --output results/render-base
bash scripts/csf.sh render finetune-own --archive artifacts/finetune-covariance --output results/render-own
bash scripts/csf.sh render-run results/smoke-own/runs/fleet_AdityaRamDas_SafeGenAI_GPT2_medium_DPO_LoRA_FT/assigned/run \
  --output results/render-one
```

The downloader resumes individually hashed files; folder ZIP streams can be
interrupted by Nextcloud locks. Validation checks byte hashes, artifact and
configuration hashes, manifest identities and input references. The validator
reports known post-index README changes and CSV line-ending changes separately;
experimental JSON/configuration changes fail validation. Rendering requires all
selected execution, baseline and edited GRAM artifacts and rejects missing or
duplicate observations. It never rewrites source manifests.

Exports contain per-case metrics and scores, per-family outcomes, per-layer
means/quartiles/minimum/maximum, source manifest hashes, causal tracing statuses,
and PNG/PDF/SVG profiles with configured layer markers. Profile bands show
observed variation, not confidence intervals. Unscored layers are not filled
with zero scores. Baselines count once per checkpoint; classic batch/repair
baseline replicas remain archived provenance, not independent controls.

ROME efficacy success means `efficacy_score >= 1.0`. Exact GRAM success means
the selected zero-based layer equals the **execution-configured** edit layer.
Baseline FP means that the **unedited checkpoint selects its future configured
ROME layer**. Joint success requires both ROME efficacy and exact GRAM matching.
Efficacy failures stay in evaluated denominators; download/runtime failures are
separate. This baseline-FP convention describes target coincidence by the
localizer. The exported JSON keeps all underlying scores and ES/PS/NS metrics.

## Causal tracing and checks

The archived campaigns use corrected fixed-legacy tracing. Preserve its raw
results, pre-edit phase and unavailable statuses. Classic tracing has 100
profiles for 17 families; the comparable **DeepSeek 7B base reference is missing**.
The checkpoint campaigns retain 766 and 727 accepted traces respectively.
Their tracing pool is separate from the assigned ROME facts. Do not infer a
replacement trace or change the frozen ROME layers based on a new trace.

`python -m src causal-kuba-fix model=gpt2-medium` remains available for a new
fixed-legacy trace. The windowed `causal-trace` implementation is retained with
its distinct protocol documented in [causal_tracing.md](causal_tracing.md).
Neither command overwrites recorded campaign results during artifact rendering.

See [docs/validation.md](docs/validation.md) for local and cluster results and
their limits, and [docs/archived-audit.md](docs/archived-audit.md) for the historical
source audit, family coverage, unavailable traces and download failures.

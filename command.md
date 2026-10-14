# CSF cluster validation commands — 14 October 2026

Validation host: `root@203.57.40.186`, SSH port `10173`. Ubuntu 22.04.5,
Python 3.11.11, NVIDIA RTX 6000 Ada (48 GB), driver 595.91.07. The virtual
environment and experiment files are isolated under `/workspace`; pip also
uses `/root/.cache/pip` for its download cache. No system packages are replaced.
Public frozen checkpoints require no Hugging Face token.

## Connect and clone

From the local machine:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 10173 root@203.57.40.186
```

On the cluster:

```bash
python3 --version
nvidia-smi
df -h / /workspace
git clone https://github.com/Security-FIT/Latium.git /workspace/Latium
cd /workspace/Latium
```

The branch was local at validation time. These commands transferred its exact
Git commits without publishing the branch. Run the first two locally from the
CSF checkout, then the remaining commands on the cluster:

```bash
mkdir -p .tmp
git bundle create .tmp/csf-submission.bundle origin/main..CSF-submission
scp -P 10173 .tmp/csf-submission.bundle root@203.57.40.186:/workspace/csf-submission.bundle
git fetch /workspace/csf-submission.bundle CSF-submission
git switch -c CSF-submission FETCH_HEAD
git rev-parse HEAD
git status --short --branch
```

After the branch is published, a new clone can use `git switch CSF-submission`
instead of transferring a bundle. The initial validation revision was
`071d2ffb9c81073219378d1cc3972bd7c7e38b46`.

## Install and check

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt > /workspace/csf-install.log 2>&1
```

Continue in the activated virtual environment. The tested package versions are
saved in [requirements-cluster.txt](requirements-cluster.txt); this includes a
CUDA 13 PyTorch build for this host. Other GPUs/drivers may need another PyTorch
build. For an identical environment on compatible hardware, use
`python -m pip install -r requirements-cluster.txt` in a clean Python 3.11 venv.

```bash
export HF_HOME=/workspace/hf-cache
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p results/cluster-validation
python -m pip check > results/cluster-validation/pip-check.log
python -m pip freeze > results/cluster-validation/requirements.lock.txt
python -m pytest -q > results/cluster-validation/pytest.log 2>&1
python -m ruff check src tests scripts > results/cluster-validation/ruff.log
python -m compileall -q src
bash -n scripts/csf.sh
bash scripts/csf.sh validate > results/cluster-validation/inputs.json
bash scripts/csf.sh plan classic --family gpt2-medium > results/cluster-validation/plan-classic.json
bash scripts/csf.sh plan finetune-base --family gpt2-medium \
  --checkpoint AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT > results/cluster-validation/plan-base.json
bash scripts/csf.sh plan finetune-own --family qwen3.5-4b \
  --checkpoint Sonpd/qwen3.5-4b-vietnamese-legal-kaggle-1K > results/cluster-validation/plan-own.json
```

The environment/CUDA check used:

```bash
python - <<'PY'
import json, platform, subprocess
from pathlib import Path
import importlib.metadata as m
import torch
assert torch.cuda.is_available()
x = torch.randn(32, 32, device='cuda')
y = x @ x.T + torch.eye(32, device='cuda')
assert torch.isfinite(torch.linalg.inv(y)).all()
torch.cuda.synchronize()
record = {
    'python': platform.python_version(), 'platform': platform.platform(),
    'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'packages': {name: m.version(name) for name in
        ['torch', 'transformers', 'datasets', 'peft', 'numpy', 'matplotlib', 'huggingface_hub', 'pytest', 'ruff']},
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    'bf16': torch.cuda.is_bf16_supported(), 'cuda_matrix_check': 'passed',
}
Path('results/cluster-validation/environment.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
```

## Pinned base versions for validation

Adapter smoke commands automatically use their recorded adapter-base pin.
The full GPT-2 Large checkpoint and classic smoke commands require an explicit
base map. This map comes from the frozen adapter records already in the
repository. It is used only for these validation subsets, and does not assert
that these were the unrecorded historical classic/base-covariance revisions.
No current Hub tip is queried for a replacement revision.

```bash
python - <<'PY'
import json
from pathlib import Path
entries = json.loads(Path('manifests/CSF-final/owncov-inputs.json').read_text())['entries']
chosen = ['AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT', 'gorni123/autotrain-vv7aw-d1ino']
pins, sources = {}, []
for name in chosen:
    e = next(e for e in entries if e['model_id'] == name)
    c = e['checkpoint']
    pins[e['family']] = c['adapter_base_revision']
    sources.append({'family': e['family'], 'adapter': name,
                    'base_repo': c['adapter_base_model'], 'revision': c['adapter_base_revision']})
Path('results/cluster-validation/smoke-base-revisions.json').write_text(json.dumps(pins, indent=2) + '\n')
Path('results/cluster-validation/smoke-base-revisions-provenance.json').write_text(json.dumps({
    'validation_only': True, 'sources': sources,
    'note': 'Recorded adapter-base revisions used for smoke validation; no claim about unrecorded historical classic/base-covariance pins.',
}, indent=2) + '\n')
print(json.dumps(pins, indent=2))
PY
```

## Smoke commands

These use frozen assigned facts, seeds, layers and original ROME settings.
Only the covariance sample count is reduced to 32 by `smoke`. Each checkpoint
command evaluates one assigned edit. Classic smoke evaluates the first fact of
each original 500-fact batch. Results are marked as validation subsets.

```bash
bash scripts/csf.sh smoke finetune-base --family gpt2-medium \
  --checkpoint AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT \
  --output results/smoke-base-medium > results/cluster-validation/smoke-base-medium.log 2>&1
bash scripts/csf.sh smoke finetune-own --family gpt2-medium \
  --checkpoint AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT \
  --output results/smoke-own-medium > results/cluster-validation/smoke-own-medium.log 2>&1
bash scripts/csf.sh smoke finetune-base --family gpt2-large \
  --checkpoint MiniLLM/MiniLLM-gpt2-760M \
  --base-revisions results/cluster-validation/smoke-base-revisions.json \
  --output results/smoke-base-large > results/cluster-validation/smoke-base-large.log 2>&1
bash scripts/csf.sh smoke finetune-own --family gpt2-large \
  --checkpoint MiniLLM/MiniLLM-gpt2-760M \
  --output results/smoke-own-large > results/cluster-validation/smoke-own-large.log 2>&1
bash scripts/csf.sh smoke finetune-base --family qwen3.5-4b \
  --checkpoint Sonpd/qwen3.5-4b-vietnamese-legal-kaggle-1K \
  --output results/smoke-base-qwen > results/cluster-validation/smoke-base-qwen.log 2>&1
bash scripts/csf.sh smoke finetune-own --family qwen3.5-4b \
  --checkpoint Sonpd/qwen3.5-4b-vietnamese-legal-kaggle-1K \
  --output results/smoke-own-qwen > results/cluster-validation/smoke-own-qwen.log 2>&1
bash scripts/csf.sh smoke classic --family gpt2-medium \
  --base-revisions results/cluster-validation/smoke-base-revisions.json \
  --output results/smoke-classic-medium > results/cluster-validation/smoke-classic-medium.log 2>&1
```

## Download failure and correction

The initial GPT-2 Large base smoke stopped at Hugging Face's Xet endpoint:
`gpt2-large/xet-read-token/<recorded revision>` returned HTTP 404. Metadata
resolved that exact revision to `openai-community/gpt2-large`. The downloader
now resolves the canonical repository at the requested immutable revision,
checks the returned revision, and saves `.latium-snapshot.json`. It retains the
original requested model name in the configuration and cache layout.

This read-only diagnostic confirmed the alias:

```bash
python - <<'PY'
from huggingface_hub import HfApi
pin = '32b71b12589c2f8d625668d2335a01cac3249519'
info = HfApi().model_info('gpt2-large', revision=pin)
print({'requested': 'gpt2-large', 'resolved': info.id, 'sha': info.sha})
PY
```

After the fix passed nine local submission tests and Ruff, it was committed as
`397d364` and transferred from the local CSF checkout:

```bash
git bundle create .tmp/csf-cluster-fix.bundle origin/main..CSF-submission
scp -P 10173 .tmp/csf-cluster-fix.bundle root@203.57.40.186:/workspace/csf-cluster-fix.bundle
```

On the cluster, the branch was updated, the initial failure log was retained,
checks were rerun, and the same GPT-2 Large smoke command above was retried:

```bash
git fetch /workspace/csf-cluster-fix.bundle CSF-submission
git merge --ff-only FETCH_HEAD
mv results/cluster-validation/smoke-base-large.log results/cluster-validation/smoke-base-large-initial.log
python -m pytest -q > results/cluster-validation/pytest-final.log 2>&1
python -m ruff check src tests scripts > results/cluster-validation/ruff-final.log
```

## Rendering

Smoke automatically renders each saved run. These commands also aggregate the
new saved runs with their original selection identities. Production runs place
each campaign under one output root, as shown in the README. The smoke commands
above deliberately use separate directories, so these read-only receipt indices
group them by campaign. Symlinks point at the original receipts and their
unchanged run roots:

```bash
python - <<'PY'
import json
from pathlib import Path
names = {'finetune-base': 'base', 'finetune-own': 'own', 'classic': 'classic'}
for p in sorted(Path('results').glob('smoke-*/runs/*/*/reproduction.json')):
    r = json.loads(p.read_text())
    directory = Path('/workspace/csf-render-inputs') / names[r['campaign']] / p.parent.parent.name / p.parent.name
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / 'reproduction.json'
    if not link.exists():
        link.symlink_to(p.resolve())
PY
bash scripts/csf.sh render-new finetune-base --runs /workspace/csf-render-inputs/base \
  --output /workspace/csf-rendered/base > results/cluster-validation/render-new-base.json
bash scripts/csf.sh render-new finetune-own --runs /workspace/csf-render-inputs/own \
  --output /workspace/csf-rendered/own > results/cluster-validation/render-new-own.json
bash scripts/csf.sh render-new classic --runs /workspace/csf-render-inputs/classic \
  --output /workspace/csf-rendered/classic > results/cluster-validation/render-new-classic.json
bash scripts/csf.sh render-run \
  results/smoke-own-medium/runs/fleet_AdityaRamDas_SafeGenAI_GPT2_medium_DPO_LoRA_FT/assigned/run \
  --output /workspace/csf-render-one > results/cluster-validation/render-one.json
```

Two initial render attempts were rejected by the existing guards: writing
inside `--runs results`, and reading mixed campaigns from that root. The commands
above use separate output directories and one campaign per receipt index. The
guards were retained.

Visual inspection found long names overlapping in per-run figure titles. Commit
`62a3db5` wraps the complete labels without shortening or replacing names.
The final export audit also found that split classic batches were reported as
separate checkpoints and their supplied version was absent in aggregate rows.
Commit `ef909fe` reports unique checkpoints separately from `source_runs` and
keeps resolved versions. Both changes passed the local submission tests and
were transferred like the download fix:

```bash
# Locally, after the figure fix:
git bundle create .tmp/csf-cluster-render.bundle origin/main..CSF-submission
scp -P 10173 .tmp/csf-cluster-render.bundle root@203.57.40.186:/workspace/csf-cluster-render.bundle
# On the cluster:
git fetch /workspace/csf-cluster-render.bundle CSF-submission
git merge --ff-only FETCH_HEAD
# Locally, after the export fix:
git bundle create .tmp/csf-cluster-export.bundle origin/main..CSF-submission
scp -P 10173 .tmp/csf-cluster-export.bundle root@203.57.40.186:/workspace/csf-cluster-export.bundle
# On the cluster:
git fetch /workspace/csf-cluster-export.bundle CSF-submission
git merge --ff-only FETCH_HEAD
```

Per-run figures were regenerated with the saved inputs and a byte-hash guard:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
from src.submission.render import render_saved_run
for p in sorted(Path('results').glob('smoke-*/runs/*/*/reproduction.json')):
    r = json.loads(p.read_text())
    root = Path(r['run_root'])
    before = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.json')}
    render_saved_run(root, root.parent / 'rendered')
    assert before == {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.json')}
print('Per-run figures regenerated; source hashes unchanged.')
PY
```

## GPU restoration and final artifact checks

The exact Python check programs used over SSH stdin are retained as
[cluster-restoration-check.py](docs/cluster-restoration-check.py) and
[cluster-output-check.py](docs/cluster-output-check.py). The first performs two
independent edits on one loaded model using the first two frozen classic facts
and its existing 32-text covariance. It verifies original bytes, parameter
identity and frozen flags after each edit, then checks rendering source hashes.
The second checks every smoke artifact, assigned fact, covariance identity/hash,
both-mode differences, aggregate versions/baseline counts and figure files.
The following file invocations replay the same checks from the repository root:

```bash
PYTHONPATH=. python docs/cluster-restoration-check.py > results/cluster-validation/gpu-restoration.log 2>&1
python -m pip install pypdf > results/cluster-validation/pdf-reader-install.log 2>&1
python -m pip freeze > results/cluster-validation/requirements.lock.txt
python -m pip check > results/cluster-validation/pip-check-final.log
PYTHONPATH=. python docs/cluster-output-check.py > results/cluster-validation/artifact-check.log 2>&1
python -m pytest -q > results/cluster-validation/pytest-final.log 2>&1
python -m ruff check src tests scripts > results/cluster-validation/ruff-final.log
python -m compileall -q src
bash -n scripts/csf.sh
bash scripts/csf.sh validate > results/cluster-validation/inputs-final.json
git diff --check
git status --short --branch
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

During this Windows-controlled session, multiline Bash commands were sent with
this PowerShell wrapper to remove transport-added CR characters:

```powershell
@'
set -euo pipefail
cd /workspace/Latium
# Linux commands from above
'@ | ssh -p 10173 root@203.57.40.186 'tr -d ''\r'' | bash -se'
```

Python check source was piped with
`Get-Content -Raw <check.py> | ssh ... 'cd /workspace/Latium; .venv/bin/python - > <log> 2>&1'`.
The Linux commands above can be entered directly in an SSH Bash session.

## Observed results

The initial suite passed 178 tests; the alias regression adds one test. The
post-fix suite passes 179 tests. Dependency checks, Ruff, compilation, Bash syntax,
frozen-input validation, CUDA matrix operations and BF16 availability pass.
The installed environment is recorded in [cluster-environment.json](docs/cluster-environment.json).

| Frozen checkpoint | Loading | Assigned fact | Target / GRAM layer, both modes | Efficacy, both modes | Neighborhood score, both modes |
|---|---|---:|---:|---:|---:|
| `AdityaRamDas/SafeGenAI_GPT2_medium_DPO_LoRA_FT` | Pinned base + merged LoRA | 18850 | 8 / 8 | 1.0 | 1.0 |
| `MiniLLM/MiniLLM-gpt2-760M` | Full checkpoint, BF16 | 173 | 12 / 12 | 1.0 | 0.1 |
| `Sonpd/qwen3.5-4b-vietnamese-legal-kaggle-1K` | Pinned base + merged LoRA, BF16 | 7780 | 5 / 5 | 1.0 | 1.0 |

Classic GPT-2 Medium smoke also passed on facts 4043 and 13954, with exact layer
8 and efficacy 1.0. It has one conceptual checkpoint/control and two source
batches. The restoration probe used facts 4043 and 886; both restorations matched
SHA256 `7ef9cd43263201ef5ede0c9ca689c939226b9e7b8343912ee82b923cc073fe15`.
All smoke baseline FP indicators are false (future target-layer convention).
All paraphrase scores are 1.0. The lower neighborhood scores are retained as
observed outcomes; efficacy success does not require neighborhood success.

All 48 primary smoke artifacts passed content/configuration hashes, identity and
dependency checks. All 102 PNG/PDF/SVG files passed decoding/parsing, including
strict PDF page/text reads. Representative per-run and aggregate linear/log
profiles were visually inspected after the label fix. Aggregate classic output
reports one checkpoint and two source runs, and all aggregate rows retain their
resolved immutable versions.

The durable [smoke-results report](docs/cluster-smoke-results.json) records
the metrics, pins, seeds, covariance identities/hashes, source revisions,
restoration evidence and figure hashes. Raw logs, full reproduction receipts,
configs, matrices and artifacts remain under `/workspace/Latium/results`.
Logs and validation receipts were also copied to the local ignored
`results/cluster-validation` directory. To fetch them from the local CSF checkout:

```bash
mkdir -p results/cluster-validation
scp -r -P 10173 root@203.57.40.186:/workspace/Latium/results/cluster-validation/. results/cluster-validation/
scp -P 10173 root@203.57.40.186:/workspace/Latium/results/cluster-validation/smoke-results.json docs/cluster-smoke-results.json
scp -P 10173 root@203.57.40.186:/workspace/Latium/results/cluster-validation/environment.json docs/cluster-environment.json
scp -P 10173 root@203.57.40.186:/workspace/Latium/results/cluster-validation/requirements.lock.txt requirements-cluster.txt
```

The finished documentation, check programs and package lock were committed and
the cluster checkout synchronized from the local CSF checkout:

```bash
# Locally:
git bundle create .tmp/csf-cluster-final.bundle origin/main..CSF-submission
scp -P 10173 .tmp/csf-cluster-final.bundle root@203.57.40.186:/workspace/csf-cluster-final.bundle
# On the cluster:
git fetch /workspace/csf-cluster-final.bundle CSF-submission
git merge --ff-only FETCH_HEAD
```

These are operational validation subsets with **32 eligible covariance texts**,
not new paper runs with 100,000 texts. Only three model families were tested on
this host. Qwen3.5 used the working PyTorch reference convolution/linear-attention
kernels; optional `causal_conv1d` and `flash-linear-attention` packages were not
installed, so optimized-kernel performance was not validated. No historical
inputs, paper results or causal tracing records were overwritten.

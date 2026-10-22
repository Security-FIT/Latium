# Workshop version

This branch is an interactive teaching build of Latium. It intentionally changes
the original ROME implementation with the user's authorization. It is not a
merge into `main` and does not contain the unrelated MEMIT working-tree changes.

## Minimal notebook

`Latium-workshop/clean-notebook.ipynb` has two cells: prepare this branch and its
dependencies, then:

```python
from src.workshop import launch_workshop
demo = launch_workshop()
```

The notebook imports implementation from this branch. It does not contain a copy
of ROME, the UI, causal tracing, or GRAM. `src.workshop` renders before downloading
the model; Drive/covariance upload and model preparation are UI actions.
An existing local checkout can be selected without switching or resetting it.
Private Drive authentication is an optional first-cell action
(`AUTHENTICATE_DRIVE`), not an interactive OAuth request from a Gradio worker.

## Patch provenance and changes

The source is the provided `latium-rome-t4.patch` (two patches, starting at
`27798d2c8c76656414a74c20643cc5565077bd7f`). It was applied to `main` at
`82b21ab`, excluding its stale hunk against `qwen3.5-4b.yaml`. The normal model
config still matches main. The new `qwen3.5-4b-t4.yaml` selects the workshop.

Retained from that patch:

- Key/value prompt microbatching and exact gradient accumulation.
- Key-capture early exit, logits selection before softmax.
- CPU inverse covariance, handler cache and in-place rank-one insertion.
- Frozen insertion/restoration and the high-level `src.rome.api` entrypoint.

Corrected during integration:

- Delta/Adam, selected-logit losses, norms and covariance direction use fp32.
- KL input is the subject-only control, with no relation or target prefix.
- Chat edit, generation, scoring and conversation share `src.rome.prompts`.
- Registry evaluation also selects the chat scorer when chat mode is enabled;
  it reuses the original ES/PS/NS aggregation with mean token NLLs.
- Exact offsets locate the subject inside user content. Target IDs are appended
  separately, avoiding BPE changes at the prompt/answer boundary.
- Chat generation explicitly disables thinking and fails instead of silently
  using a completion or thinking template. Cleanup precedes special-token removal.
- Hooks are removed on failure; leading padding positions stay stable.
- CPU-mapped covariance loading and chunked insertion avoid transient GPU copies.
- Rectangular and square `nn.Linear` weights have an explicit orientation.
- Repeated API edits seed their context sampling consistently.

`src.workshop.rome` only adapts arguments to `src.rome.common` and `src.rome.api`.
`src.workshop.gram` imports the main structural detector. Causal interventions
come from `src.causal_trace.model_adapter`. There is no second workshop-only
implementation of the editing equations or detector.

## Qwen main comparison

| Setting | main | workshop-version |
|---|---|---|
| Checkpoint | Qwen/Qwen3.5-4B | Same, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Edit layer | 5 | 5 |
| Precision | bf16 | bf16, including `auto`; fp16 requires explicit selection |
| Epochs | 25 | 25 |
| Prefix mode | self | static, 20 prompts |
| k/v batch size | full prompt set | 4/1 by default in UI |
| Covariance placement | edit device | CPU |
| Prompt format | completion | no-thinking chat, opt-in in T4 preset |
| Optimizer variable | model dtype | fp32 |

The previous standalone notebook selected fp16 automatically on T4. This branch
removes that unverified numerical substitution. T4 lacks native BF16 tensor
cores; emulation may be slow and actual supported kernels must be checked on the
target runtime. A GPU end-to-end test is still required before claiming T4 fit or
4B edit efficacy. Upcasting a loss cannot repair already overflowed activations.

The provided covariance artifact is validated for finite values and projection
dimensions. Its filename identifies model/layer but does not establish checkpoint
revision/content provenance; an artifact without such metadata cannot prove that
its estimation weights match the pinned checkpoint.

## UI behavior

- Original white/red/cyan design and logos, five lessons, collapsed source panels.
- Prepared and custom editing have side-by-side forms/results and chat.
- GPU memory/utilization updates outside the model queue.
- One serialized model-operation queue, explicit loading/error state, no blanket
  overlays over every output component.
- First field is original answer; second is desired answer.
- Reopening the UI preserves the loaded model, active edit and history. Reset is
  explicit. Empty generations are not invented as assistant history.
- Both chats use one system instruction and token budget; instruction changes
  clear old conversation context.
- Direct/implicit multihop and observed-city decomposition remain separate.
  Expected answers are never supplied to generation. Scores are normalized per
  target token, and actual generated text remains visible.

## Validation

```bash
python -m pytest tests/test_rome_batching.py tests/workshop -q
python -m ruff check src tests scripts huggingface-scraper rome_benchmark.py
```

The CPU suite exercises a tiny real Qwen3.5 full/linear-attention model, exact
chat token positions, microbatch gradients, insertion/restoration, failure cleanup,
real tracing interventions and GRAM. Optional browser audit:

Validation on this integration: 201 targeted tests passed, three optional
dependency checks skipped. The separate workshop/notebook suite passed 128 tests.
The real Qwen3.5 tokenizer passed English/Czech round-trip, no-thinking-prefill and
subject-position checks. The Chromium audit passed desktop/mobile rendering,
actual source-code display, covariance upload and preparation-error/retry checks,
with no JavaScript errors or mobile horizontal overflow.

```bash
pip install playwright
playwright install chromium
python scripts/workshop_browser_smoke.py --artifacts /tmp/opencode
```

The clean notebook's source and export tests live in the separate workshop repo.

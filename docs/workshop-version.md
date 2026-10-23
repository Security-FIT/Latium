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
| Epochs | 25 | UI demo 15 / full experiment 25 |
| Prefix mode | self | static, UI demo 8 / full experiment 20 prompts |
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
  Expected answers are never supplied to generation; actual generated answers
  are visible. The presentation does not compute or display candidate log-scores.

### Presentation flow, caching and performance

- State/progress has no internal scroll area. Technical revision/invalidation
  banners have been removed; the current fact stays visible above the lessons.
- Long work runs in one serialized worker and streams a heartbeat with elapsed
  time. Phase estimates come from completed work units, not invented download
  percentages. Closing a stream cancels at a checkpoint and joins its worker.
- One process-lifetime session and load lock survive app/page recreation. The
  same model is not loaded twice even when another UI opens during preparation.
  A Colab kernel reset still destroys GPU memory and all in-memory caches.
- Edits cache up to two owned projection snapshots in CPU RAM. Baseline/edited
  generated answers are keyed by exact model state, question and generation
  options. No full model checkpoints are cached. Tab 3 reuses the installed edit
  from tab 2, rather than restoring base and optimizing again. Cached reuse does
  not erase tab 2's results or conversation.
- Edit-budget controls remain usable after model preparation. Changing epochs,
  prefix counts or microbatches does not reload weights. Cache keys include the
  meaningful edit configuration.
- The default demo uses 8 prefixes / 15 optimization iterations (135 main+KL
  forwards at v-batch 1), versus 525 for 20/25. The full profile remains available.
  This is an explicit budget/quality tradeoff, not an efficacy guarantee.
- Qwen's supported `logits_to_keep` interface selects required hidden positions
  before the 248320-token LM head. This avoids computing vocabulary logits for
  every prompt token. Tests compare deltas against the full-logit reference;
  unsupported models retain the full-logit path.
- Ordinary verification no longer performs candidate-scoring forwards. Answers
  use a 40-token budget with a smaller explicit default set of controls. Multihop
  uses the same budget and reuses direct answers; it does not generate 192 tokens
  per probe or inject the expected city into the country question.
- Multihop always keeps the currently installed edit even when the user has
  changed the settings for a future edit. A changed city target (for example
  Berlin instead of the preset Rome) is preserved; expected countries for known
  cities are display metadata only and are never inserted into generation.
- First-call tracing embedding statistics are cached. Noise draws can share a
  forward batch; the UI selects noise strength before measuring paired effects.
  Default aggregation uses three distinct facts (UI range 1–5). Results cache per
  fact/state/options, so repeating or expanding a profile reuses finished cases.
  Weak signal is reported rather than amplified or invented. GRAM result reuse
  also avoids repeating unchanged analysis.

CPU measurements verify reduced LM-head position counts, identical optimized
deltas, model identity, one editor call across tabs 2/3 and no extra generation
on repeated requests. Full 4B/T4 wall-clock speed and multihop efficacy still
require the user's GPU runtime; there is no claimed GPU speedup measurement.

### Readable results and rainbow tracing

ROME comparisons sit below the form/chat row and use the full tab width.
Read-only HTML tables preserve complete answers and line breaks; mobile layouts
use labelled blocks instead of forcing five columns into a narrow spreadsheet.
Multihop probes use individual cards containing both answers and expectations.
History and optional tracing/GRAM details use responsive metric cards.
All model text is escaped before insertion into HTML. Controller row formats and
numerical calculations are unchanged.

The default causal chart now matches the rainbow bar style of the supplied
`causal data/llama2-7b/legacy_rainbow.png`: violet-to-red layer colors, L0…Ln labels
and a peak annotation. Its heights are the workshop's actual mean paired IE per
MLP window. The supplied legacy report instead aggregates the maximum across
subject-position rows within each run before averaging. This UI change borrows
the chart presentation; it does not silently claim to run that different legacy
protocol. The axis and summary identify the MLP-window measurement.

- Notebook `COVARIANCE_DRIVE_URL` is passed to `launch_workshop(covariance_drive_url=...)`
  and prefills the preparation field. Nonempty input replaces the previous link;
  empty input preserves the latest UI link (or the default on first launch).
  Changed Drive IDs trigger a new atomic download during model preparation.
- Custom fact fields stay empty and show a coherent Hamlet → Tolkien example
  through placeholders. Gradio explains verification, editing, chat and multihop
  reproducibility directly beside the relevant forms;
  participants can use the defaults without changing the PR implementation or checkout.
- `launch_workshop(seed=42)` initializes the preparation field. The selected seed
  reaches root `cfg.seed`, seeds model loading and is reset by the shared ROME API
  on every edit. Reopening a loaded session retains its original seed and weights.
  Greedy answers and fixed hardware/configuration are required for comparison.
- Default multihop scenarios now adapt Table 1 / §5.1 of
  [arXiv:2601.04600](https://arxiv.org/abs/2601.04600): edit tallest-building identity
  (hop 1) or Burj Khalifa's country (hop 2). The prepared tab defaults to the latter.
  Exact recall, a distinct edited-hop paraphrase, unchanged-hop recall and two
  composed queries are measured separately. An unrelated Q/A prefix requests
  direct answers without providing any edited entity or target to composed queries.
- Verdicts distinguish paraphrase failure, loss of unchanged knowledge, stale
  composed answers and returning an intermediate building instead of a country.
  Explicit UAE aliases avoid treating a valid country abbreviation as failure.
  Cached edits and measurements retain the upstream transaction/rollback semantics.
  Qwen/layer-5 results are not presented as a replication of the GPT-J layer study;
  the demo neither implements redundant editing nor infers causal layer ordering.

## Validation

```bash
python -m pytest tests/test_rome_batching.py tests/workshop -q
python -m ruff check src tests scripts huggingface-scraper rome_benchmark.py
```

The CPU suite exercises a tiny real Qwen3.5 full/linear-attention model, exact
chat token positions, microbatch gradients, insertion/restoration, failure cleanup,
real tracing interventions and GRAM. Optional browser audit:

Validation including presentation/session/performance changes: 214 targeted tests passed, three optional
dependency checks skipped. The separate workshop/notebook suite passed 128 tests.
The real Qwen3.5 tokenizer passed English/Czech round-trip, no-thinking-prefill and
subject-position checks. The Chromium audit passed desktop/mobile rendering,
actual source-code display, covariance upload and preparation-error/retry checks,
with no JavaScript errors or mobile horizontal overflow.

```bash
pip install playwright
playwright install chromium
python scripts/workshop_browser_smoke.py --artifacts /tmp/opencode
python scripts/workshop_browser_smoke.py --populated --artifacts /tmp/opencode
```

The clean notebook's source and export tests live in the separate workshop repo.
The populated browser audit uses clearly labelled layout fixtures with long
multiline answers. It checks result-cell widths/heights, not only the outer page,
on both desktop and mobile, and never supplies fake measurements in production.

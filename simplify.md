# Latium simplification and correctness review

> **Validated and implemented on 2026-07-10.** The finding text below is kept
> as the pre-fix rationale; status labels and this table describe the current
> tree. Findings 1, 4, 5, and the three-engine portion of 10 had already been
> fixed by commit `807a168`. The remaining applicable changes were implemented
> and verified in the current worktree.

## Current status

| # | Status | Current evidence |
|---:|---|---|
| 1 | Resolved earlier | Obsolete `rome-layer-sweep`/prototype command was removed; the one trace command reads `command.causal_trace`. |
| 2 | Resolved | Covariance contributions are committed only after a successful full forward; a late-OOM regression verifies no ghost tokens. |
| 3 | Resolved | Config hashes are present in graph rows, CSVs, labels, grouping keys, indexes, and detector-signal paths. |
| 4 | Resolved earlier, test added | Fallback matching indexes the exact model input including BOS; a no-offset/BOS fixture verifies positions 1–2. |
| 5 | Resolved earlier | Discovery and confirmation are disjoint and selection requires at least two confirmation facts. |
| 6 | Resolved | Analysis/render errors are persisted and raise by default; explicit `continue_on_error` enables best-effort runs. |
| 7 | Resolved | `src.common.config.strict_bool` is the single strict parser used by runtime and structural adapters. |
| 8 | Resolved | `model_families` was removed from edit/capture/renderer specs; it remains only on analyses, where it is enforced. |
| 9 | Resolved | `StructuralBenchmarkConfig` owns normalization and bounds; the Hydra adapter extracts nested values and uses schema defaults. |
| 10 | Resolved | There is one tracing engine, with tokenization and statistical selection split into focused modules. |
| 11 | Resolved | The unused `style_preset` setting and cache-hash plumbing were removed. |
| 12 | Resolved | Every renderer declares inputs, and `RendererSpec` rejects missing contracts. |
| 13 | Resolved | In-tree renderers use only typed `RenderContext`; the dict bridge was deleted. |
| 14 | Resolved | Structural benchmark config is coerced once and the same object is passed through capture/analysis/render. |
| 15 | Resolved | Sweep inputs use native YAML lists and reject invalid/even/legacy string values instead of rewriting them. |
| 16 | Resolved | The ROME monolith is now a compatibility facade over prefixes, subjects, optimization, activations, and covariance modules. |
| 17 | Resolved with compatibility retained | Dead `section_value` was removed; `src.utils`, `src.model_config`, and `src.rome.common` are documented compatibility facades, excluded from production imports. |
| 18 | Retained by design | Lazy `module:attribute` loading is the documented extension contract and avoids eager model/plotting imports. |

## Scope and verification

This is a code-focused review of the current refactored branch, with attention
to the recent causal-trace, structural/artifact, graph, ROME, and command-layer
changes. Existing untracked notebooks and notes were left untouched.

Checks completed:

- Full test suite: 132 passed.
- Ruff: python3 -m ruff check src tests scripts huggingface-scraper rome_benchmark.py passed.
- Targeted in-memory reproductions were used for the Hydra configuration and
  graph-output findings below.

Passing tests are useful but do not exercise the GPU/OOM, no-offset-tokenizer,
multiple-analysis-variant, or command-config paths described here.

## Original priority summary

| Priority | Count | Meaning |
|---|---:|---|
| P1 | 4 | Can silently produce incorrect research output or ignore user configuration. |
| P2 | 5 | Incorrect or misleading behavior under valid but less common input/failure conditions. |
| P3 | 9 | Simplifications that remove duplicate, dead, or misleading implementation surface. |

## P1 — fix before relying on the affected output

### 1. rome-layer-sweep ignores its configured tracing settings — resolved

**Files:** src/causal_trace/prototype.py:127-129, 895-900, 1118-1176; src/config/command/rome_layer_sweep.yaml:14-36  
**Confidence:** High

The active rome-layer-sweep command stores its settings below
cfg.command.tracing, but the prototype reads only cfg.tracing:

~~~py
def _cfg_section(cfg: DictConfig, name: str) -> Any:
    value = getattr(cfg, name, None)
    return value if value is not None else OmegaConf.create({})
~~~

Hydra composition for command=rome_layer_sweep was verified to produce:

~~~text
root_tracing=False
command_tracing=True
configured_output=analysis_out/causal_trace
~~~

Consequently, _run_dir, run_trace_mode, and run_rome_layer_sweep work from an
empty config. The command silently discards its configured output directory,
noise settings, thresholds, fallback, and ROME-validation options, including
user overrides made under command.tracing.

**Simplest fix:** make the configuration helper first look under cfg.command
(or reuse the nested-section helper used by causal_trace.py), then fall back to
the root for backwards compatibility. Add a regression test that composes
command=rome_layer_sweep and verifies an override reaches run_trace_mode.

### 2. A CUDA OOM can contaminate second-moment covariance results — resolved

**Files:** src/rome/common.py:1178-1185, 1253-1292, 1341-1346  
**Confidence:** High

The forward hook updates the shared covariance and token count before the
forward has completed:

~~~py
def hook(_, inp, out):
    nonlocal C, total_tokens, current_attention_mask
    if current_attention_mask is None:
        raise RuntimeError("Missing attention mask while accumulating covariance")
    hidden_states = inp[0] if isinstance(inp, tuple) else inp
    total_tokens += _accumulate_second_moment_tokens(C, hidden_states, current_attention_mask)
    return out
~~~

If a later operation in that forward raises CUDA OOM, the handler requeues or
skips the chunk but does not roll back C or total_tokens:

~~~py
except torch.cuda.OutOfMemoryError:
    LOGGER.warning("OOM during covariance computation (chunk=%d)", len(chunk))
    if len(chunk) <= 1:
        LOGGER.warning("Skipping sample that causes OOM even at batch_size=1")
        continue
    midpoint = max(1, len(chunk) // 2)
    queue.insert(0, chunk[midpoint:])
    queue.insert(0, chunk[:midpoint])
~~~

The eventual result is normalized from this contaminated state with
cov = C / total_tokens. Failed tokens can therefore be counted once, or
multiple times after a retry; a single-item OOM can leave ghost data despite
being logged as skipped.

**Simplest fix:** accumulate each attempted chunk into temporary covariance and
count buffers, committing them only after model forward succeeds. Alternatively
snapshot and restore the affected state before retrying. Add a fake model whose
hooked layer runs and then raises OOM; the existing batching test raises too
early to cover this path.

### 3. Graphs conflate and overwrite distinct analysis variants — resolved

**Files:** src/results/ids.py:36-49; src/graphs/runtime.py:38-40, 194-227; src/graphs/renderers.py:25-45, 211-238, 311-432  
**Confidence:** High

Analysis artifacts intentionally use the configuration hash in their identity:

~~~py
def analysis_id(model, plan_id, edit_method, category, analysis, analysis_config_hash):
    return (
        f"{_plan_prefix(model, plan_id)}/method/{safe_slug(edit_method)}"
        f"/analysis/{safe_slug(category)}/{safe_slug(analysis)}"
        f"/{safe_slug(analysis_config_hash)}"
    )
~~~

The graph runtime then loads every analysis record, but the generic summary
rows discard config_hash and label a variant only by model, method, and
analysis. Detector-signal paths also omit the variant identity:

~~~py
graph_path = output_dir / model / plan_id / method / analysis / f"{case_slug}.png"
~~~

A targeted reproduction passed two otherwise identical analysis payloads to the
signal renderer. It returned the same PNG path twice, so the second variant
overwrote the first:

~~~text
plot_outputs=[.../m/p/rome/rank1-blind/case.png,
              .../m/p/rome/rank1-blind/case.png]
unique_plot_paths=1
~~~

Detector-window aggregation also groups variants by a label without config
identity, mixing their cases into one percentage. This is especially dangerous
because analysis variants are an explicit supported feature.

**Simplest fix:** carry a short config-hash or explicit variant name through
summary rows, labels, CSV columns, and output directories. Decide whether
graphs should show every variant or require a selected variant, and test two
hashes for the same model/plan/method/producer.

### 4. No-offset tokenizers can patch the BOS token instead of the subject — resolved

**Files:** src/causal_trace/causal_trace.py:165-186, 486-487, 520, 525-529; src/causal_trace/prototype.py:208-228  
**Confidence:** High

The fallback strips BOS before matching the subject:

~~~py
prompt_ids = _strip_bos(tokenizer, _token_ids(tokenizer, prompt))
...
start, end = matches[0]
positions = list(range(start, end))
return TokenSpan(start, end, positions, end - 1)
~~~

But those zero-based, BOS-stripped positions are later used directly as indexes
into the model input activation tensors. For a tokenizer without offset mapping
whose model input retains BOS, a subject beginning at model position 1 is
reported and modified at position 0 instead.

**Simplest fix:** preserve the count of stripped special tokens and add it
back to fallback positions, or match directly against the exact input_ids sent
to the model. Add a no-offset tokenizer fixture that prepends BOS; the present
toy tokenizer ignores add_special_tokens and only exercises the offset path.

## P2 — correctness and failure-semantics risks

### 5. One fact is incorrectly treated as held-out confirmation — resolved

**Files:** src/causal_trace/causal_trace.py:870-884, 414-427, 623-741  
**Confidence:** High

The trace only rejects the zero-fact case, then reuses the discovery fact as
its own confirmation when exactly one fact was accepted:

~~~py
split = max(1, len(fact_results) // 2)
discovery_facts = fact_results[:split]
confirmation_facts = fact_results[split:] or fact_results[:split]
~~~

For one fact, the bootstrap interval is also the same point estimate. A noisy
single example can therefore satisfy the stated confirmation flow and produce
a strict selected layer.

**Simplest fix:** require a nonempty, disjoint confirmation split (at least
two accepted facts, preferably a configurable minimum per split). Otherwise
write an insufficient_facts result and leave strict_layer unset.

### 6. Operational failures are converted to successful command exits — resolved

**Files:** src/causal_trace/causal_trace.py:850-880; src/structural/analysis/runtime.py:343-359; src/graphs/runtime.py:81-101; src/command_handlers/graphs.py:19-35; src/command_handlers/structural.py:39-67  
**Confidence:** High for the behavior; Medium for whether best-effort
continuation is intended.

Several orchestration loops catch every Exception, save an error/rejection
artifact, and return normally:

~~~py
except Exception as exc:
    outputs = []
    status = "error"
    error = str(exc)
...
return {"run_id": reader.manifest["run_id"], "written": written, "skipped": skipped}
~~~

The CLI handlers then return 0. A renderer, analysis, or causal-trace
programming/configuration error can thus be mistaken for a successful run by
shell automation. Error details are recorded, but process status does not
surface them.

**Simplest fix:** catch only expected per-case validation exceptions by
default; expose an explicit continue_on_error mode for batch research runs,
and return non-zero when any artifact has status=error unless that mode was
chosen. Retain the error artifacts for diagnosis.

### 7. Boolean parsing is inconsistent and treats the string "false" as true — resolved

**Files:** src/runtime.py:57-65; src/structural/config.py:243-246; src/structural/hydra_config.py:36-39  
**Confidence:** High

Two strict boolean parsers already exist, but runtime settings use Python
truthiness:

~~~py
prefix_log_all=bool(_get(runtime, "prefix_log_all", fallback.prefix_log_all)),
second_moment_allow_autocompute=bool(...),
log_skip_traceback=bool(_get(runtime, "log_skip_traceback", fallback.log_skip_traceback)),
~~~

For a dictionary/compatibility configuration, bool("false") is True. This can
unexpectedly enable logging, automatic second-moment computation, or traceback
suppression.

**Simplest fix:** centralize one strict converter in src/common/config.py and
use it in runtime, structural config, and Hydra extraction. Reject unknown
strings rather than silently interpreting them.

### 8. model_families promises restrictions that most registries never enforce — resolved

**Files:** src/editing/registry.py:20-48; src/structural/capture/registry.py:16-57; src/graphs/registry.py:16-73; src/registry.py:74-76; src/structural/analysis/runtime.py:286  
**Confidence:** High for non-enforcement; current impact Low until a restricted
entry is registered.

The generic support check is sound:

~~~py
def supports_model(entry: Any, model: str) -> bool:
    families = tuple(getattr(entry, "model_families", ("all",)))
    return "all" in families or model_family(model) in families
~~~

It is used by analyses, but edit-method loading, capture selection, and
renderer selection never apply it. An entry can declare model_families=("gpt",)
and still run on a non-GPT model.

**Simplest fix:** enforce the field consistently at each selection boundary,
or remove it from the edit/capture/renderer specs and documentation until it
has a real contract.

### 9. Structural configuration has duplicated normalization with divergent behavior — resolved

**Files:** src/structural/config.py:59-246; src/structural/hydra_config.py:30-143; src/structural/runner.py:17-34  
**Confidence:** High

The Hydra adapter and StructuralBenchmarkConfig both own defaults, conversion,
and validation. They have already drifted:

~~~py
# src/structural/config.py
"capture_profile": "paper",

# src/structural/hydra_config.py
"capture_profile": ("capture", "profile", "spectral", str),
~~~

They also normalize numeric bounds differently. The targeted in-memory check
showed that direct construction accepts an invalid negative value:

~~~text
StructuralBenchmarkConfig(n_tests=-3).n_tests == -3
~~~

whereas the Hydra adapter clamps n_tests with _int_at_least(0). The public
mapping API is therefore not equivalent to the CLI API, and
validate_structural_config does not validate these numeric bounds.

**Simplest fix:** define one typed dataclass/schema with defaults and
validators. Let the Hydra adapter only extract nested values into it. Reject
invalid input rather than clamping it differently by entrypoint.

## P3 — simplify the refactor surface

### 10. Causal tracing has three overlapping engines and duplicated primitives — resolved

**Files:** src/causal_trace/causal_trace.py (976 lines); src/causal_trace/prototype.py (1,176 lines); src/causal_trace/alt_trace.py (496 lines); src/command_handlers/operations.py:41-52  
**Confidence:** High

The public causal-trace command invokes causal_trace.py, while
rome-layer-sweep invokes prototype.py. A separate alt_trace.py remains
unwired by any public command. The two large engines share 12 function names;
five definitions are AST-identical:

~~~text
_hidden_from_output, _replace_hidden, _strip_bos,
make_noise_samples, temporary_hooks
~~~

This is not merely a size concern: the engines use different config layouts,
and that divergence caused finding 1. prototype.py also exposes mode wrappers
that have no in-repository caller, while alt_trace.py defines another
run_alt_trace implementation.

**Simplest fix:** retain one supported tracing engine, move shared
tokenization/hook/noise primitives into a small common module, and make modes
data/configuration rather than parallel entrypoint implementations. Archive
alt_trace.py and compatibility helpers only after deciding whether external
direct-module callers are supported.

### 11. graphs.style_preset is a no-op that invalidates caches — resolved

**Files:** src/config/graphs/default.yaml:10; src/command_handlers/graphs.py:32; src/graphs/runtime.py:53-58, 82-90; src/graphs/context.py:30, 54; src/graphs/structural/style.py:43-67  
**Confidence:** High

The setting is threaded into the render config and its config hash, then only
stored in RenderContext. No renderer consumes it. The actual structural style
code hard-codes:

~~~py
plt.style.use("default")
~~~

Changing graphs.style_preset triggers a render cache miss without changing an
output.

**Simplest fix:** either remove the field and plumbing now, or implement a
small, tested named-style registry that every renderer actually uses.

### 12. Most renderers have an implicit dependency on every artifact — resolved

**Files:** src/graphs/registry.py:32-73; src/graphs/runtime.py:59-71, 138-146; src/graphs/renderers.py:200-431  
**Confidence:** High

Only structural-artifact-grid declares its inputs. When a renderer has no
declarations, the runtime silently gives it every execution, capture, and
analysis artifact:

~~~py
if not has_declarations:
    return [*executions, *captures, *analyses], [], []
~~~

This causes unrelated changes to invalidate and reload legacy graphs. It also
obscures each renderer's actual contract: paper and detector renderers consume
analyses, whereas ROME-success consumes executions.

**Simplest fix:** require every renderer to declare input kinds or producers
(for example all_analyses or all_executions). Remove the catch-all fallback
after migrating the six legacy renderers.

### 13. Graph rendering maintains two incompatible input APIs — resolved

**Files:** src/graphs/context.py:43-56; src/graphs/renderers.py:19-22; src/graphs/structural/artifact_grid.py:80; src/graphs/README.md:32-37  
**Confidence:** Medium

Most renderers accept Any and immediately convert a typed RenderContext back
to a dictionary:

~~~py
def _context_mapping(context: Any) -> dict[str, Any]:
    if hasattr(context, "as_mapping"):
        return context.as_mapping()
    return context
~~~

Only structural-artifact-grid uses the typed context directly. The repository
therefore maintains a legacy dict protocol and a new typed protocol in
parallel.

**Simplest fix:** migrate the six in-tree legacy renderers to RenderContext,
give their dependencies typed declarations (finding 12), then delete
_context_mapping and RenderContext.as_mapping.

### 14. Structural benchmark re-coerces the same mapping after capture — resolved

**Files:** src/structural/runner.py:44-58, 80-85  
**Confidence:** High

run_structural_benchmark normalizes a mapping in run_structural_capture, then
constructs a second StructuralBenchmarkConfig from that same mapping:

~~~py
capture_result = run_structural_capture(config)
resolved = config if isinstance(config, StructuralBenchmarkConfig) else StructuralBenchmarkConfig(**dict(config))
~~~

Besides needless work, this separates the generated run-id held by the
capture-local configuration from the later benchmark configuration.

**Simplest fix:** use one private coerce_structural_config function at the
start of run_structural_benchmark and pass the resolved object to capture,
analysis, and rendering.

### 15. Structural sweep parsing is a legacy mini-language beside Hydra lists — resolved

**Files:** src/structural/planning.py:23-172, 256-314; src/config/structural/default.yaml:44-67, 123-127; README.md:103-105  
**Confidence:** Medium

Planning implements comma splitting, semicolon grouping, permissive parsing,
deduplication, clamping, and odd-window coercion even though current YAML
already uses typed lists and the documentation says Hydra overrides are the
supported interface. Some parser paths silently drop invalid values or clamp
them, making an invalid experiment configuration hard to notice.

**Simplest fix:** make typed Hydra lists and fail-fast validation authoritative.
Keep one narrow compatibility normalizer only if historical comma-string input
must remain supported, and warn or reject rather than silently changing values.

### 16. src/rome/common.py is a 1,459-line mixed-responsibility module — resolved

**Files:** src/rome/common.py:40-1459; callers in src/rome/rome.py, src/handlers/rome.py, src/editing/rome.py, src/experiments/prefix_variability/runner.py, and src/command_handlers/operations.py  
**Confidence:** High

The module combines prefix-source configuration and caching, prefix generation,
subject-token positioning, ROME optimization, key/value insertion,
second-moment batching, and covariance cache loading. Its largest functions
are second_moment_wikipedia (207 lines) and optimize_v (179 lines), while
unrelated consumers import from the same facade.

This makes the OOM issue in finding 2 harder to isolate and tests unrelated
logic through the same import surface.

**Simplest fix:** split by stable responsibility without changing public
behavior:

- src/rome/prefixes.py for PrefixMode, PrefixGenerationHandler, and template/cache work;
- src/rome/subjects.py for subject-position helpers;
- src/rome/optimization.py for gather_k, optimize_v, and insert_kv;
- src/rome/covariance.py for second-moment collection and cache loading.

Keep a temporary compatibility re-export only if external imports require it.

### 17. Dead or conditional compatibility surface should be made explicit — resolved

**Files:** src/common/config.py:39-50, 100-110; src/utils.py:15-96; src/model_config.py:12-32; tests/test_architecture_imports.py:25-47  
**Confidence:** High for internal non-use; external compatibility is unknown.

section_value is defined and exported but has no in-repository call site.
src/utils.py and src/model_config.py are broad compatibility re-export facades;
the architecture tests explicitly ensure production code does not import them.

**Simplest fix:** remove section_value now. For the two facades, do not delete
blindly: document whether they are public, add a deprecation window if they
are, then move them to a clearly named legacy package or remove them.

### 18. Dynamic registry loading is more flexible than the current surface needs — retained by design

**Files:** src/editing/registry.py:25-48; src/structural/capture/registry.py:27-57; src/structural/analysis/registry.py:64-185; src/graphs/registry.py:32-73; src/registry.py:51-62  
**Confidence:** Medium; this is an optional architectural simplification.

Editing methods are dynamically discovered from YAML object paths, but there is
currently one in-tree edit-method configuration. Capture, analysis, and graph
registries are otherwise static Python lists which still resolve object paths
dynamically. The project should choose one extension model rather than carry
both throughout the core.

**Simplest fix:** either standardize on a documented plugin/discovery contract,
or use direct callable factories for in-tree registries and reserve dynamic
loading for explicitly external plugins.

## Original recommended order

1. Fix findings 1 through 4 and add their focused regression tests.
2. Make one decision about failure semantics (finding 6) before running larger
   experiments in automation.
3. Resolve graph variant identity and renderer contracts together (findings 3,
   11, 12, and 13).
4. Consolidate the causal-trace implementations, then split ROME common.py
   behind compatibility imports if needed.
5. Remove dead configuration/facades only after deciding their public
   compatibility status.

# Causal Tracing Implementations

Two causal tracing workflows are available.

---

## Version 1: Standard Causal Tracing

**User-facing entry point**

- Hydra command: `causal-trace`
- Companion notebook: standard trace notebook

**What it does**

For each prompt, the workflow corrupts the embeddings of **all subject
tokens**, then restores the clean hidden state of **each subject token
individually** at every layer. This produces a 2D restoration matrix per
prompt whose axes are subject-token position and layer depth.

**Steps per prompt**

1. Run a clean forward pass and sample the next token.
2. If the clean token does not match the expected target, skip the prompt.
3. Run one corrupted forward pass with noise injected at all subject-token
   positions; record the corrupted token and its probability.
4. For each subject token position:
   - Inject noise at all subject-token positions.
   - For every layer, restore the clean hidden state at that token position
     and run a restored forward pass.
   - Record the target-token probability at each layer.

**Technical characteristics**

- Uses one noise sample per prompt.
- Reads the clean hidden state on demand inside the layer loop.
- The corrupted baseline is measured before the restoration sweeps begin.
- Noise scale is controlled by a pre-computed noise multiplier derived from
  embedding standard deviation.
- No built-in layer ranking: the raw per-token, per-layer probabilities are
  written to CSV for downstream analysis.
- The notebook variant wraps tracing in a no-gradient context and uses
  defensive hook cleanup.

**Output**

A single timestamped CSV per run. Each row records:

- run and prompt identifiers
- clean token and probability
- corrupted token and probability
- the subject token position being restored
- a list of `(restored token, probability)` pairs, one per layer

---

## Version 2: Alternative Causal Tracing

**User-facing entry point**

- Hydra command: `alt-trace`
- Companion notebook: alternative trace notebook

**What it does**

For each prompt, the workflow corrupts all subject-token embeddings, but
restores **only the last subject token** at every layer. It averages the
per-layer restoration probabilities over multiple independent noise samples
and then ranks the layers, falling back to the middle third of the network
when the signal is unreliable.

**Steps per prompt**

1. Run a clean forward pass and sample the next token.
2. If the clean token does not match the expected target, skip the prompt.
3. Select the last subject token as the single restoration target.
4. Pre-cache the clean hidden state for that token at every layer.
5. Repeat `N` times (default `N = 10`, configurable):
   - Inject fresh noise at all subject-token positions.
   - For every layer, restore the pre-cached clean hidden state at the last
     subject token and run a restored forward pass.
   - Record the target-token probability at each layer.
6. Average the recorded probabilities across the `N` runs.
7. Measure the final corrupted baseline (no restoration) after the sweeps.

**Technical characteristics**

- Uses multiple independent noise samples per prompt and averages results.
- Pre-caches clean hidden states before the repeated run loop.
- The corrupted baseline is measured after the restoration sweeps.
- Noise scale is resolved automatically: a configured multiplier is respected,
  otherwise the scale is `3 × embedding standard deviation`, with a hard
  fallback of `0.1` when no data is available.
- Built-in layer ranking with signal-quality detection:
  - Coefficient-of-variation threshold: `0.15`
  - Peak-to-mean ratio threshold: `0.3`
  - Middle-third band: `[num_layers / 3, 2 × num_layers / 3]`
  - If any check fails, ranking is restricted to the middle third.
- Returns a structured recommendation containing the best layer, ranked
  candidates, signal quality, and whether the middle-third fallback was used.

**Outputs**

Two artifacts are written:

1. A wide CSV with one row per prompt and one column per layer, plus clean
   and corrupt probabilities.
2. A JSON recommendation containing:
   - the selected best layer
   - signal quality (`clean` or `noisy`)
   - whether the middle-third fallback was applied
   - the full ranked candidate list
   - the averaged per-layer probabilities

---

## Quick Comparison

| Aspect | Version 1 (Standard) | Version 2 (Alternative) |
|---|---|---|
| Restoration target | All subject token positions | Only the last subject token |
| Noise samples per prompt | 1 | 10 by default, configurable |
| Clean hidden state | Read inside layer loop | Pre-cached before run loop |
| Corrupted baseline timing | Before restoration sweeps | After restoration sweeps |
| Noise scale | Pre-computed multiplier from embedding std | `3 × embedding std`, fallback `0.1` |
| Built-in layer ranking | No | Yes, with middle-third fallback |
| Signal quality checks | None | CV < 0.15, peak ratio < 0.3, peak outside middle third |
| Output | One nested CSV | Wide CSV + JSON recommendation |
| Top-level result | Raw CSV rows only | Structured layer recommendation |

---

## CLI Usage

```bash
# Version 1
python3 -m src causal-trace model=gpt2-large generation.num_of_runs=5

# Version 2
python3 -m src alt-trace model=gpt2-large generation.num_of_runs=5 generation.num_trace_runs=10
```

# ROME reversal: recovering an edit from the weights

GRAM localizes a ROME edit because the edited down projection has a
residual-stream direction that its neighbours do not have. This branch turns
that detector into an estimator: find the direction, subtract the rank-one
piece that explains it, check that the GRAM profile becomes normal again, and
decode the removed piece into the edited subject and the new target. No
access to the edit request or to the unedited model is needed for the
recovery itself; the unedited weights are only used to score it.

| Path | Content |
|---|---|
| `src/structural/detectors/gram_peel.py` | direction search, closed-form peel, iterative peeling, peelability |
| `scripts/gram_peel_experiment.py` | real ROME edits → peel → comparison with the true update → blind decoding |
| `tests/test_gram_peel.py` | synthetic rank-one edits (recovery, layout invariance, unedited layers) |

## Method

### Where the edit direction is

ROME writes `W' = W + v kᵀ` (`v` in the residual stream, `k` on the MLP key
side). The raw hidden Gram changes by

    W k vᵀ + v kᵀ Wᵀ + |k|² v vᵀ,

an indefinite rank-two term whose span is `{v, W k}`. The written value `v`
therefore lies in the plane of the most positive and most negative
eigenvectors of `G_ℓ - R`, where `R = (G_(ℓ-1) + G_(ℓ+1)) / 2` is the GRAM
neighbour reference. The top eigenvector alone is not enough: on synthetic
edits it reaches cos ≈ 0.96 with `v` and leaves a residual anomaly.

### Peeling one direction

For a candidate unit `v`:

1. Keep every row of `W'` that is orthogonal to `v` (`P = I - v vᵀ`).
2. Replace the row along `v` by the minimum-norm row `a` that makes the
   peeled Gram agree with the reference on the `v` column: `P W' a = c P R v`.
3. Solve this in Gram space, as a `d × d` system. The removed key is
   `k̂ = W'ᵀ (v - P x)` and the peeled Gram follows without touching `W'`.

`v` is chosen by a one-dimensional search (grid plus golden section) in the
eigen-plane that minimizes the peeled layer's GRAM score. The removed
rank-one estimate of the edit is `v k̂ᵀ`.

### Decision

`iterative_peel` uses the following loop:

1. Peel each of the five highest-scoring layers on a copy of the weights.
2. Keep the peel that removes the largest fraction of its layer's score.
3. Repeat while that collapse is at least `DEFAULT_MIN_COLLAPSE = 0.7`.

The number of accepted peels is the number of rank-one anomalies. An edited
ROME layer is explained by one peel. Unedited layers spread their anomaly
over many directions and do not collapse.

### Blind decoding

| Question | How |
|---|---|
| Sign of `(v, k̂)` | The sign with the strongest positive key response over all subjects. |
| Edited subject | Rank the MLP keys of all 20,391 CounterFact subjects by `key · k̂`. Each key is the mean of the bare subject and of `"Here is a fact about <subject>"`. |
| Subject token | Rank every vocabulary token by the same response, using `"Here is a fact about <token>"`. |
| New target | Log-probability gain of every token, edited model minus peeled model, averaged over generic prompts with the blindly decoded subject (`"{} is"`, `"{} is located in"`, …). |
| Logit lens | Shown for comparison: `v` through the final norm and the unembedding. |

## Results

Facts come from `manifests/counterfact_seed42_n1000.json`. The runs used an
A100 80 GB and Transformers 5.20.

### GPT-2 (124M)

ROME layer 5, covariance from 2,000 wikitext texts, 8 facts.

| Measure | Result |
|---|---|
| GRAM peak on the edited layer | 6/8 (2 weak edits leave the baseline peak, layer 1) |
| One-peel collapse, edited layer | median **0.93** |
| One-peel collapse, same layer unedited | median 0.49; maximum over all unedited layers 0.65 |
| Edits explained by exactly one peel | 6/6 detected |
| cos(`v`, true value direction) | 0.956 |
| cos(`k̂`, true key direction) | 0.938 |
| ‖`v k̂ᵀ` − ΔW‖ / ‖ΔW‖ | 0.51 (estimate norm 1.05–1.16 × true) |
| Subject among 20,391 CounterFact subjects | **top-1 6/6** |
| Subject last token from the vocabulary | top-1 4/6, top-10 5/6 |
| New target, blind behavioural gain | **top-1 4/6, top-10 6/6** |
| New target, logit lens on `v` | top-1 2/6, median rank 28 |

### GPT-2 XL

ROME layer 16, 100k-text covariance. Partial run: 4 facts, run with the
earlier 0.85 acceptance threshold.

| Measure | Result |
|---|---|
| GRAM peak on the edited layer | 4/4 |
| One-peel collapse, edited layer | 0.80, 0.83, 0.85, 0.86 |
| One-peel collapse, same layer unedited | 0.12; maximum over all unedited layers 0.48 |
| Decoded case (collapse ≥ 0.85) | cos `v` 0.975, cos `k̂` 0.912, subject rank 1, target rank 2 (behaviour) |

Three of the four edits had collapses of 0.80–0.85, so the 0.85 threshold
rejected them. That is why the default is now 0.7, which separates edited
from unedited layers on both models. The GPT-2 XL subject-token and
logit-lens ranks are poor (908 and 2239). Subject and target decoding go
through the key responses and the model behaviour, not through the
vocabulary geometry of `v`.

### Limitations

- Small samples (8 and 4 edits). The threshold is set from two models and
  needs a fleet run.
- Recovery is only checked for ROME. Edits that spread over several layers
  (for example MEMIT) are not reversed by this procedure.
- The subject search covers the CounterFact subject list; an open-vocabulary
  subject search is only approximated by the single-token scan.

## Running

```bash
# ROME inverse covariance for model.layer (if missing)
python -m src second-moment model=gpt2-xl

# edit, peel, compare with the true update, decode
python scripts/gram_peel_experiment.py --model gpt2-xl --facts 20 \
  --case-index-file manifests/counterfact_seed42_n1000.json \
  --out analysis_out/gram-peel/gpt2-xl.json

# summary of a saved run
python scripts/gram_peel_experiment.py --model gpt2-xl \
  --out analysis_out/gram-peel/gpt2-xl.json --summary-only
```

Useful options:

| Option | Effect |
|---|---|
| `--min-collapse` | acceptance threshold for one peel |
| `--candidates` | layers tried per round |
| `--grid`, `--refine` | direction search resolution |
| `--no-vocab-scan` | skip the vocabulary key scan, which needs one forward pass per token |
| `--max-subjects` | limit the subject list for smoke tests |

`src/common/loading.py` also accepts GPT-2 causal-mask buffers that
Transformers 5 reports as `h.N.attn.bias`. Without this, GPT-2 checkpoints
fail to load on Transformers 5.

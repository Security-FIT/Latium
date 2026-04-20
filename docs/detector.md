# ROME Layer Detector — Technical Documentation

---

## Overview

Two post-hoc detectors identify which transformer layer was edited by ROME,
using only the weight matrices of the edited model (no access to the original
weights or the edit prompt).

| Detector | Scope | Accuracy | Algorithm |
|----------|-------|----------|-----------|
| **Composite v5c** | All non-GPT models | 161/169 (95%) | 4-signal spectral chain + structural fallback |
| **GPT v6** | GPT-2 family + GPT-J | 39/39 (100%) | 3-way norm_cv majority vote |

Both detectors read `blind_detection.layer_features` from structural
benchmark JSON files produced by `structural_benchmark.py`.

---

## Signals

Seven per-layer features are computed from each weight matrix via full SVD:

| Signal | Key | Definition |
|--------|-----|------------|
| Spectral gap | `spectral_gap` | $\sigma_1 / \sigma_2$ |
| Top-1 energy | `top1_energy` | $\sigma_1^2 / \lVert W \rVert_F^2$ |
| Effective rank | `effective_rank` | $e^{H(\mathbf{p})}$ where $p_i = \sigma_i^2 / \sum \sigma_j^2$ |
| Norm CV | `norm_cv` | $\text{std}(\lVert \mathbf{w}_i \rVert) / \text{mean}(\lVert \mathbf{w}_i \rVert)$ over row norms |
| Row alignment | `row_alignment` | $\max\lvert U_{:,0} \rvert / \text{mean}\lvert U_{:,0} \rvert$ |
| Spectral entropy | `spectral_entropy` | $H(\mathbf{p}) = -\sum p_i \log p_i$ |
| PCS | `pcs` | Mean pairwise cosine similarity of rows |

---

## Transforms

Each signal is passed through up to four transforms before peak detection:

| Transform | Tag | Formula |
|-----------|-----|---------|
| Raw | `raw` | Identity — use the signal directly |
| Local z-score (w=5) | `lz5` | $\lvert z_i \rvert$ where $z_i = (x_i - \bar{x}_{N(i)}) / s_{N(i)}$, neighborhood $w=5$ |
| Local z-score (w=7) | `lz7` | Same, $w=7$ |
| Curvature | `curv` | $\lvert x_{i-1} - 2x_i + x_{i+1} \rvert$ (absolute second-order finite difference) |

Local z-score measures how anomalous a layer is relative to its neighbors,
making it robust to global trends.  Curvature detects sharp local changes.

---

## Composite Detector v5c (`composite_detector_v2.py`)

**Trim**: 2 layers from each end (removes embedding/head artifacts).

### Algorithm

The detector uses a cascading confirmation chain of spectral signals,
falling back to structural signals when spectral evidence is weak.

#### Step 1 — Spectral chain (4 signals)

Primary signals:
- **SG**: spectral_gap raw peak
- **TE**: top1_energy lz5 peak
- **SL5**: spectral_gap lz5 peak
- **SL7**: spectral_gap lz7 peak

Decision logic:
1. **SG == TE** (same peak layer) → `agree` — highest confidence
2. SL5 near SG but not TE (±1 index) → `sg(lz5)` — SG confirmed
3. SL5 near TE but not SG → `te(lz5)` — TE confirmed
4. SL7 near SG but not TE → `sg(lz7)` — weaker SG confirmation
5. SL7 near TE but not SG → `te(lz7)` — weaker TE confirmation
6. Neither confirms exclusively:
   - If SG has monotone trend ($|\rho| > 0.3$):
     - SL5 ≈ SL7 (±1): local-z consensus → `lz_cons`
     - SL7 ≈ TE (±1): use TE → `te(trend)`
     - Else: use SL7 → `s7(trend)`
   - Else: SG reliable → `sg(fb)`

#### Step 2 — Structural override (v5b/c)

Only fires for the two lowest-confidence paths: `s7(trend)` and `sg(lz7)`.

Secondary signals:
- **NC**: norm_cv lz5 peak
- **ER**: effective_rank curvature peak
- **RA**: row_alignment raw peak

Override rules:
1. **ER + RA agree** (±1 index) → use their consensus (catches GPT-2-XL)
2. **NC + ER agree** (±1 index) → use their consensus (catches GPT-2-medium)
3. `s7(trend)` + edge artifact (peak in top 15% of layers) → use RA alone
4. `sg(lz7)` + NC disagrees → use NC alone (catches GPT-2-large)

### Binary edit detection

Compares the peak local z-score of SG/TE between a ROME-edited model
and a baseline.  When the baseline has at least two valid samples and
non-zero variance, separation $z > 2.0$ → model was edited.  A one-sample
or zero-variance baseline is reported as inconclusive instead of fabricating
an arbitrarily large separation score.

---

## GPT Detector v6 (`gpt_detector.py`)

Spectral signals (spectral_gap, top1_energy) produce no usable peak for
GPT-2/GPT-J architectures after ROME editing.  The GPT detector uses
only `norm_cv` with three transforms.

**Trim**: 5 layers from each end (GPT edge artifacts are stronger).

### Algorithm

Three signals vote:
- `norm_cv` raw → peak layer gets 1 vote
- `norm_cv` lz5 → peak layer gets 1 vote
- `norm_cv` curvature → peak layer gets 1 vote

**Primary**: Layer with ≥ 2 votes (majority) at trim=5 wins.

**Fallback**: If no majority at trim=5, pool votes across trims 4, 5, 6
(9 total votes). Layer with most votes wins.

### Why three transforms?

No single transform works for all GPT models:

| Model | nc_raw | nc_lz5 | nc_curv | 3-way vote |
|-------|--------|--------|---------|------------|
| GPT-J | ✓ t=4+ | ✗ | ✓ | ✓ |
| GPT-2-XL | ✓ | ✗ | ✓ | ✓ |
| GPT-2-large | ✗ | ✓ | ✓ | ✓ |
| GPT-2-medium | ✓ | ✓ | ✓ | ✓ |

---

## File Layout

```
detector/
├── composite_detector_v2.py   # v5c composite (all non-GPT models)
├── gpt_detector.py            # v6 GPT-specific detector
├── visualize_detector.py      # Jupytext notebook for visualization
└── graphs/
    ├── summary_accuracy.png
    ├── method_breakdown.png
    ├── signals_*.png          # Per-model spectral signal profiles
    └── gpt/
        └── gpt_signals_*.png  # Per-model NC signal profiles
```

---

## Running

```bash
# Composite detector on all models
python detector/composite_detector_v2.py ultrasupertest/ results_n5/

# GPT detector on GPT models only
python detector/gpt_detector.py ultrasupertest/ results_n5/

# Generate graphs
python detector/composite_detector_v2.py ultrasupertest/ results_n5/ --graphs --graph-dir detector/graphs
python detector/gpt_detector.py ultrasupertest/ results_n5/ --graphs --graph-dir detector/graphs/gpt

# Binary edit detection (ROME vs baseline)
python detector/composite_detector_v2.py --binary ultrasupertest/ --baseline-dir ultrasupertest/
```

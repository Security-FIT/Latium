# Structural Analysis 

---

## Metrics (`metrics.py`)

All metrics in metrics.py operate on a **neuron group** (subset of row indices) within a layer.

### l2_discrepancy

- Mean L2 norm of the delta rows for the group 
- Measures how much the group's weights changed on average

```
delta = modified_weights[layer_idx] - original_weights[layer_idx]

def l2_discrepancy(delta: torch.Tensor, indices: List[int]) -> float:
    return delta[indices].norm(dim=1).mean().item()
```

### relative_discrepancy

- Ratio of delta magnitude to original weight magnitude
- Normalizes change by original scale (0.01 change on 1.0 weight is less significant than 0.01 change on 0.1 weight)

### directional_coherence
How coherently did neurons in the group change direction? [0,1]

```
1. normalize each delta row `delta[i] / ||delta[i]||`
2. compute pairwise cosine similarity S = (normalized_delta @ normalized_delta.T)
3. coherence = (sum(S) - n) / (n*(n-1)) 
```
0 = random change directions (noise), 1 = all neurons changed in the same direction (coherent change)

### pcs_change

Absolute change in PCS for neuron group 

---

## Groupers (`groupers.py`)

Partition neurons (rows of `W`) into groups based on structural properties of the weight matrix. 
Groupers take a **single** weight matrix - in normal mode this is `W_orig`, in blind mode this is the (potentially modified) weights. They just partition the matrix they receive.

### MagnitudeGrouper(n_groups=4)

Groups by L2 row norm into quantiles (e.g. 4 groups = quartiles)

```
row_norms = ||W[i]|| for each row i
split into n_groups quantile bins: q1 (lowest norms) ... q4 (highest norms)
```
- magnitude_q1: lowest magnitude neurons
- magnitude_q4: highest magnitude neurons

### SpectralGrouper(top_k=10)

Groups by contribution to top singular vectors
```
U, S, V = SVD(W) # https://en.wikipedia.org/wiki/Singular_value_decomposition
contribution[i] = sum(|U[i, :top_k]|) # how much row i contributes to top_k singular vectors
split at median to low spectral / high spectral groups
```

### SparsityGrouper(threshold=0.01)

Groups by weight sparsity pattern

```
sparsity[i] = fraction |W[i,j]| < threshold
dense = sparsity < 0.3
medium = 0.3 <= sparsity <= 0.7
sparse = sparsity >= 0.7
```

### RandomGrouper(n_groups=2, seed=67)

Random partition, baseline

---

## Normal Detector (`detector.py` - `WeightMSDDetector`)

**Requires:** original weights + modified weights. Mostly used to validate the blind detector and understand how the metrics change.
This was just a initial test, the idea was to use this insight while developing the blind detector. Can be fully ignored.

### Layer-wise MSD

```
For each layer: score[layer] = ||W_mod[layer] - W_orig[layer]||_F (Frobenius norm of the delta)
anomolous_layer = argmax(score) 
z_score = (max_score - mean) / std
```
Find which layer has highest total change


### Neuron group MSD

```
On anomalous layer:
    For each group g (from groupers):
        group_scores[g] = l2_discrepancy(delta, g.indices)
    group_msd = max(group_scores) - min(group_scores)
    most_affected = argmax(group_scores)
```

Measures if edit affected some neuron groups more than others 

### ROME Signature Check

```
U, S, V = SVD(delta)
rank_one_score = S[0]^2 / sum(S^2)       # fraction of energy in first SV
effective_rank = exp(entropy(S/sum(S))) # Shannon entropy-based effective rank
```

- `rank_one_score ≈ 1.0` → delta is (near) rank-1 → classic ROME signature
- `effective_rank ≈ 1.0` → same idea via entropy

### Bootstrap Significance

Tests whether the observed group difference is significant compared to random grouping

### Outputs:

- `anomalous_layer` - highest delta norm
- `layer_z_score` - how many std deviations above mean is the anomalous layer's delta norm
- `all_layer_scores` - per layer frobenius norms of delta
- `most_affected_group` - neuron group with highest l2 discrepancy
- `group_msd` - `max - min` of group scores
- `group_scores` - per group l2 discrepancy scores
- `rank_one_score` - energy fraction in first singular value of delta
- `effective_rank` - entropy based effective rank of delta
- `top_singular_values` - top 5 SVs of delta matrix
- `p_value` - bootstrap significance of group MSD

---

## Blind Detector (`blind_detector.py` - `BlindMSDDetector`)

**Requires:** only the (potentially modified) weights. No original model needed.

Operates in two phases:
- **Phase 1 - Layer Detection:** find which layer was edited (Blind Layer MSD, Grouper-Based Detection)
- **Phase 2 - Layer Analysis:** characterize the edit in the suspicious layer (Blind Neuron Group MSD)

---

### Phase 1 - Layer Detection

### Blind Layer MSD (`blind_layer_msd`) - *primary*

Computes 7-feature vector per layer, uses Isolation Forest to find the outlier layer

#### Per-layer features

*NOTE*:
`U, S, V = SVD(W)` where:
- `S` = vector of singular values (sorted descending)
- `S1` = largest singular value, `S2` = second largest, etc.
- `U` = left singular vectors (neurons * features)
- `V` = right singular vectors

This is the CORE concept for many of the features below. 

`row_norms = ||W[i]||_2` for each row $i$.

- `effective_rank` - `exp(entropy(S/sum(S)))` - diversity of singular values, ROME reduces it, because it concentrates energy
- `spectral_gap` - `S1 / S2` - ratio of top singular values, ROME increases it - dominant SV grows
- `top1_energy` - `S1^2 / sum(S^2)` - fraction of energy in top singular value, ROME increases it 
- `pcs` - average pairwise cosine similarity of rows
- `norm_cv` - `std(row_norms) / mean(row_norms)` - coefficient of variation of row norms
- `row_alignment` - `max(|U[:,0]|) / mean(|U[:,0]|)` - how much the top singular vector is dominated by a few rows, spike = one neuron dominates
- `spectral_entropy` - `H(S^2 / sum(S^2)) / log(n)` -  normalized entropy of squared SVs. Lower = more concentrated spectrum

**Anomaly detection:**

```
feature_matrix = [x features * n_layers]
IsolationForest(contamination=0.1).fit(feature_matrix)
anomalous_layer = layer with highest anomaly score
```

**Outputs (core):**
- `anomalous_layer` - layer index with highest anomaly score
- `layer_anomaly_score` - z-score of the anomaly score
- `layer_features` - per-layer 7-feature dict
- `isolation_scores` - per-layer IsolationForest scores
- `feature_z_scores` - per-layer z-scores for each feature

---

### Phase 2 - Layer Analysis

### Blind Neuron Group MSD (`blind_neuron_group_msd`)

#### Intra-layer analysis on suspicious layer

Takes in single weight matrix (1 layer) that was identified as suspicious by Phase 1.

Determines whether the suspicious layer shows internal structural inconsistencies.
e.g. some subgroups look different than others? If ROME edited this layer, the rank-1 update will have affected some rows more than others.

Grouping: combines the groupers below (magnitude, spectral, sparsity) plus a median split by row L2 norm, then averages discrepancies across strategies.

#### Step by step:

1. Extract per-row features:
    - row_norms[i] = L2 norm of row i
    - row_spectral_contrib[i] = sum of absolute values in top-k columns of U (from SVD)
    - row_sparsity[i] = fraction of weights below threshold (mean * 0.1)

2. For each grouper (magnitude/spectral/sparsity):
    - partition rows into groups
    - compute mean spectral contribution, sparsity, and norm per group
    - add group spreads (max - min) to the discrepancy lists

3. Median split fallback:
    - high_mag = rows above median norm
    - low_mag = rows below median norm
    - add spectral/sparsity discrepancies for this split

4. Aggregate discrepancies:
    - mean across all grouping strategies

5. Outlier detection:
    - build feature matrix from (row_norms, spectral_contrib, sparsity)
    - use IsolationForest to flag anomalous rows (contamination = 0.05)


#### Outputs:

- `spectral_discrepancy` - Mean spectral discrepancy across all grouping strategies
- `sparsity_discrepancy` - Mean sparsity discrepancy across all grouping strategies
- `norm_spread` - Mean norm spread across grouping strategies
- `outlier_fraction` - Fraction of rows flagged as outlier by iforest
- `n_outlier_rows` - count of outlier rows
- `outlier_indices` - row indices flagged as anomalous

### Grouper-Based Detection (`blind_grouper_detection`)

Runs on every layer. Uses different classes (MagnitudeGrouper 4 quantiles, SpectralGrouper, SparsityGrouper)

#### Core idea

ROME applies rank-1 update to single layer's weight matrix, so it does not affect all neurons equally.
By splitting neurons into groups (by different metrics) and measuring how different those groups are from each other (within same layer) can spot edited layer.

In practice: 
for each layer, partition rows into groups (e.g. magnitude quartiles) -> compute stats per group -> measure the spread (difference) between groups.
If one layer has much bigger spread than all others, it was likely edited.

#### Step by step

```
for each layer:
    for each grouper (magnitude/spectral/sparsity):
        1. partition rows into groups
        2. per group: compute mean_norm, std_norm, cv_norm on row L2 norms
        3. per group: compute spectral features on the submatrix (SVD of group rows only)
           -> effective_rank, spectral_gap, top1_energy
        4. spread metrics = compare groups within this layer:
            norm_spread  = max(group_mean_norms) - min(group_mean_norms)
           cv_spread    = max(group_cvs) - min(group_cvs)
           norm_ratio   = max(group_mean_norms) / min(group_mean_norms)
           + effective_rank_spread, spectral_gap_spread, top1_energy_spread
```

## Interlayer Analysis

This is in progress; there is a lot of potential, many things that can be checked but its a lot of work.

Looks at relationships between layers (rather than within a single layer).
The idea: in an unmodified model, layers form smooth progressions, if we do an edit, then the edited layer may "stand out" as being different from the others.


### Compute per layer features

computes feature vector for a single weight matrix W. 

1. `S = stdvals(W)` gives singular values 

gives us features from S and from row norms:

```
top1_energy, top5_energy   — energy concentration in top SVs
spectral_gap               — S1/S2 ratio
effective_rank             — entropy-based diversity of SVs
sv_kurtosis                — kurtosis of SV distribution (heavy tails = one dominant component)
norm_cv                    — coefficient of variation of row norms
norm                       — overall Frobenius norm
```

### layer_block_analysis - block-wise z-scores

Splits layers into positional blocks (early/mid/late). Computes z-score within each block.

```
1. split layers into n_blocks positional groups
2. per block: compute mean and std for each feature
3. z-score each layer against its own block's mean/std
4. composite |z| = L2 norm of all feature z-scores per layer
```

### neighbor_transition_analysis - jump detection

Measures feature changes between consecutive layers. 

```
1. per consecutive pair: compute raw delta per feature
2. z-normalize each feature's deltas across all transitions
3. transition magnitude = l2 norm of z-normalized deltas
4. z-score the composite magnitudes
```

### leave_one_out_variance

only for top1_energy - remove one layer at a time, measure how much cross-layer variance drops. The edited layer has high top1_energy (we saw it spikes in ROME edits).

```
full_var = var(top1_energy across all layers)
For each layer i:
    reduced_var = var(top1_energy without layer i)
    influence[i] = full_var − reduced_var
Z-score the influence values
```

### cross_layer_fingerprint 

Separate SVD pass, then a simple similarity check.

```
1. for each layer, keep only the top-20 singular values and normalize them to sum to 1.
2. compare each layer's vector to every other layer's vector (cosine distance).
3. average those distances per layer and z-score them.
```

If a layer's average distance is high, its spectrum looks unlike the rest.
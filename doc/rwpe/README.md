# Random-Walk Positional Encoding (RWPE)

RWPE attaches to each node a vector of return probabilities under a random walk
on the molecular graph. Dimension `k` of node `v`'s encoding is:

```
RWPE[v, k] = (T^(k+1))[v, v]
```

where `T` is the row-normalised transition matrix of the adjacency matrix `A`.
The full vector fingerprints the local topology at each walk length.

## Design decisions

### Self-loop (`A + I`)

`T` is built from `A + I` rather than bare `A`.

- **Without self-loops**, a degree-`d` node has zero return probability at odd
  walk lengths (bipartite-like oscillation), making odd and even dimensions
  carry fundamentally different statistics.
- **With self-loops**, each step becomes a *lazy* random walk (the walker may
  stay in place), guaranteeing nonzero return probability at every dimension and
  producing smoother, better-spread `[0, 1]` histograms.
- The diagnostic signal `RWPE[:, k] − RWPE[:, k−2]` also has a tighter,
  more interpretable spread with self-loops.

```python
adjacency += np.eye(num_nodes, dtype=np.float64)
degree = adjacency.sum(axis=1)
transition[nonzero_degree] = adjacency[nonzero_degree] / degree[nonzero_degree, None]
```

### Dimension `dim = 12` (`RWPE_DIM`)

- Walk lengths beyond ~12 on typical drug-like molecules (median heavy-atom
  count ≈ 20–25, diameter ≈ 10–14) carry negligible additional structural
  signal once the transition matrix has largely mixed.
- 12 dimensions cover short-range (bonds, angles) and medium-range (rings,
  fused ring systems) topology without redundancy.
- Memory: at `float16`, 12 dimensions per node is ~half the cost of a
  16-dimension baseline for large datasets such as PCQM4Mv2 (≈ 3.8 M
  molecules).
- Distribution analysis on 10 000 random molecules confirms each of dimensions
  0–11 carries distinct, well-spread signal; dimensions beyond 12 show rapidly
  diminishing variance and increasing correlation with earlier ones.

## Files

| File | Role |
|---|---|
| `src/dataset.py` | Computes RWPE during graph preprocessing (`_compute_rwpe`). |
| `doc/rwpe/plot_rwpe_distribution.py` | Samples molecules from raw SMILES, computes RWPE on-the-fly, and plots per-dimension distributions and `dim k − dim k−2` difference histograms. |

## Usage

```bash
python doc/rwpe/plot_rwpe_distribution.py
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--raw-csv-path` | `dataset/pcqm4m-v2/raw/data.csv.gz` | Path to raw SMILES CSV |
| `--num-molecules` | `10000` | Molecules sampled |
| `--dim` | `64` | RWPE dimensions to plot |
| `--bins` | `80` | Histogram bins per panel |
| `--seed` | `0` | Random seed |
| `--out` | `doc/rwpe/rwpe_distribution.png` | Output image (relative paths resolve to script dir) |

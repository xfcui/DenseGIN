# k-hop Edge Feature Distribution Analysis

This document summarizes how k-hop edges are defined in the preprocessing graph
pipeline and how the analysis script in this folder builds joint statistics.

## Source behavior

The graph conversion in `src/dataset/graph.py` builds:

- Direct edges from the molecular graph
- `edge_index_2hop` and `edge_feat_2hop`
- `edge_index_3hop` and `edge_feat_3hop`
- `edge_index_4hop` and `edge_feat_4hop`

`_khop_edges` enumerates **acyclic simple paths** using neighbor sets:

- 2-hop edges: paths of length 2
- 3-hop edges: paths of length 3
- 4-hop edges: paths of length 4

Edges are stored as directed pairs, so each unordered pair contributes two directed
entries in the arrays.

Each k-hop edge stores `k` path-count features:

- 2-hop: `[path_count_1hop, path_count_2hop]`
- 3-hop: `[path_count_1hop, path_count_2hop, path_count_3hop]`
- 4-hop: `[path_count_1hop, path_count_2hop, path_count_3hop, path_count_4hop]`

Buckets use `possible_path_count_list = [0, 1, 2, 'misc']` from
`src/dataset/features.py`. All counts `>=3` are mapped to `'misc'`.

## What is plotted

1. Three joint bubble plots: number of k-hop directed edges (collapsed into
   undirected counts) versus number of heavy atoms, for k = 2, 3, 4.
2. A combined bar-plot figure of k-hop feature bucket distributions for each
   feature dimension:
   - 2-hop edges: 2 dimensions
   - 3-hop edges: 3 dimensions
   - 4-hop edges: 4 dimensions

Both plots are computed from random SMILES sampled from the raw PCQM4Mv2 CSV.

## Files generated

From the default run (10000 molecules), the script writes files under `doc/k_hop/`:

- `k_hop_2hop_vs_heavy_10k.svg`
- `k_hop_3hop_vs_heavy_10k.svg`
- `k_hop_4hop_vs_heavy_10k.svg`
- `k_hop_feat_distribution_10k.png`

## Run

```bash
python doc/k_hop/plot_k_hop_distribution.py
```

Optional flags:

- `--num-molecules`
- `--seed`
- `--raw-csv-path`
- `--out` (base output stem; default writes to `doc/k_hop` with names above)

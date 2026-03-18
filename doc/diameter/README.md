# Graph Diameter in the Default Molecular Graph

Each molecule can be seen as an undirected graph for the model:

1. parse SMILES,
2. expand to explicit H atoms with `Chem.AddHs`,
3. keep all heavy atoms plus polar hydrogens (`_keep_atom_mask`),
4. add an undirected edge for every retained bond.

This yields the same node set used by the graph construction code in
`src/dataset/graph.py` (`heavy atoms + polar H`) and therefore aligns the
exploration with the model’s default topology.

---

## Definition

For each sampled molecule, graph diameter is defined as:

`diameter = max_{u, v in V, u != v} shortest_path_length(u, v)`

That is, the largest shortest-path distance between any two nodes.

For a graph with a single node, the diameter is defined as `0`.

---

## Why diameter matters

Graph diameter summarizes coarse molecular span:

- larger diameters indicate more elongated/chain-like molecules,
- smaller diameters indicate compact, highly cyclic, or highly branched structures.

As an input topology descriptor it complements local motif-based features by
capturing global layout scale.

---

## Plot

`plot_diameter_distribution.py` samples random SMILES from
`dataset/pcqm4m-v2/raw/data.csv.gz` and renders a joint bubble plot of:

- x-axis: number of nodes in the default graph (heavy atoms + polar H),
- y-axis: graph diameter.

The output is:

`diameter_vs_heavy_10k.svg`

Run:

```bash
python doc/diameter/plot_diameter_distribution.py
```

You can change sample size and seed via `--num-molecules` and `--seed`, or choose
a different output name with `--out`.

---

## Files

| File | Role |
|---|---|
| `doc/diameter/plot_diameter_distribution.py` | Computes diameter on the default graph and writes the SVG output in `doc/diameter/`. |
| `doc/diameter/diameter_vs_heavy_10k.svg` | Generated joint bubble plot (diameter vs node count). |
| `doc/diameter/README.md` | Explains the graph diameter definition and the plotting setup. |

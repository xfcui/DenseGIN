# Feature Design: Molecular Graph Representation for HOMO–LUMO Gap Prediction

This document consolidates the feature engineering decisions for the PCQM4Mv2
dataset (~3.8 M molecules, target: HOMO–LUMO gap in eV, metric: MAE).

---

## Graph Construction

Each molecule is converted from a SMILES string into a directed graph by
`smiles2graph` in `src/dataset.py`. The graph carries two kinds of node
features (discrete and continuous), one set of direct-bond edge features, and
three additional hop-aware edge sets (2-hop, 3-hop, and 4-hop).

### Hydrogen Handling

Not all atoms become graph nodes. After `Chem.AddHs`, hydrogens are
partitioned:

| Category | Kept as node? | How info is preserved |
|---|---|---|
| Heavy atoms (Z ≥ 2) | Yes | Full feature vector |
| **Active H** — bonded to N, O, or S | Yes | Full feature vector |
| Non-active H — all other hydrogens | No | Count stored as `numH` feature on parent |

Active hydrogens participate in H-bonding, which influences frontier orbital
geometry. Retaining them adds ~1.3 nodes/molecule on average. Keeping all H
would add ~15, roughly doubling graph size for no benefit.

---

## Node Features

### Discrete features (`node_feat`) — 10 indices per atom

Each feature is an integer index into a vocabulary, intended for embedding
table lookup in the model.

| # | Feature | Vocabulary | Source |
|---|---------|-----------|--------|
| 0 | Atomic number | 1–118, misc | `GetAtomicNum` |
| 1 | Chirality | 4 tags + misc | `GetChiralTag` |
| 2 | Degree | 0–10, misc | `GetTotalDegree` |
| 3 | Formal charge | −5 to +5, misc | `GetFormalCharge` |
| 4 | Non-active H count | 0–8, misc | `_non_active_hydrogen_count` |
| 5 | Radical electrons | 0–4, misc | `GetNumRadicalElectrons` |
| 6 | Hybridisation | SP, SP2, SP3, SP3D, SP3D2, misc | `GetHybridization` |
| 7 | Aromatic | False, True | `GetIsAromatic` |
| 8 | In ring | False, True | `IsInRing` |
| 9 | Min ring size | 3–8, misc | `MinAtomRingSize` |

Feature 9 (min ring size) captures ring strain and fused-ring topology that
no other single feature encodes. See `doc/ring_size/README.md`.

### Continuous features (`node_rwpe`) — 17 floats per atom

Stored as a single `float16` array, concatenated in the order below.

| Slice | Dim | Feature | Value range | Centering |
|-------|-----|---------|-------------|-----------|
| `[:12]` | 12 | RWPE | [0, 1] | None (return probabilities) |
| `[12]` | 1 | Electronegativity | ≈ −1.7 to +1.4 | Carbon-centered (`EN − EN_C`) |
| `[13]` | 1 | Gasteiger charge | ≈ −0.8 to +0.8 | Uncentered (neutral-mol sum ≈ 0) |
| `[14:17]` | 3 | 3D coordinates | unbounded | Centroid-subtracted per molecule |

These four sub-features serve distinct roles:

**RWPE** (`doc/rwpe/README.md`) — encodes graph topology via random-walk
return probabilities at walk lengths 1–12. Uses a lazy random walk (A + I) to
avoid bipartite oscillation. Dimensions beyond 12 show diminishing variance on
drug-like molecules (median diameter ~10–14).

**Electronegativity** (`doc/electronegativity.md`) — the intrinsic tendency of
an atom to attract electrons. Carbon-centering maps the most common element to
0.0. A non-monotonic function of Z that would otherwise need to be memorised
from the periodic table.

**Gasteiger charge** (`doc/gasteiger_charge.md`) — context-dependent partial
charge from iterative electronegativity equilibration. In principle derivable
from EN via multi-hop message passing, but providing it explicitly gives
shallow models access to pre-computed electronic information. NaN/outlier
guarded (`|gc| > 4` → 0).

**3D coordinates** — centroid-subtracted Cartesian positions from the training
SDF. Zero-filled when no conformer is available (test set).

---

## Edge Features

### Discrete features (`edge_feat`) — 5 indices per bond

| # | Feature | Vocabulary | Source |
|---|---------|-----------|--------|
| 0 | Bond type | SINGLE, DOUBLE, TRIPLE, AROMATIC, misc | `GetBondType` |
| 1 | Stereo | 6 stereo labels | `GetStereo` |
| 2 | Conjugated | False, True | `GetIsConjugated` |
| 3 | Rotatable | False, True | `_is_rotable_bond` |
| 4 | Min ring size | 3–8, misc | `MinBondRingSize` |

**Rotatable bonds** (`doc/rot_bond/README.md`) — uses RDKit's strict SMARTS
definition, which excludes partial-double-bond single bonds (amide C–N, ester
C–O). ~12% of bonds are rotatable on average.

**Min ring size** on bonds — the same vocabulary as atoms, applied to
`MinBondRingSize`. Distinguishes bridgehead and fused bonds by the tighter
ring constraint.

All edges are stored in both directions (i→j and j→i) with identical features.

### K-hop edge features (`edge_feat_2hop`, `edge_feat_3hop`, `edge_feat_4hop`)

In addition to the direct-bond edges, `smiles2graph` now adds directed 2-hop, 3-hop, and
4-hop edge streams:

| Stream | Edge index | Edge feature shape | Feature meaning |
|---|---|---|---|
| `2-hop` | `edge_index_2hop` | `(num_2hop_edges, 2)` | Counts of acyclic paths at step lengths 1 and 2 |
| `3-hop` | `edge_index_3hop` | `(num_3hop_edges, 3)` | Counts of acyclic paths at step lengths 1, 2, and 3 |
| `4-hop` | `edge_index_4hop` | `(num_4hop_edges, 4)` | Counts of acyclic paths at step lengths 1, 2, 3, and 4 |

For each pair `(u, v)`, a k-hop edge exists whenever at least one acyclic path of exactly
that length exists. Pairs can therefore appear in multiple hop sets (for example, both 2-hop
and 3-hop) and may also be direct bonds (count_1=1).

Path-count vocabulary:

`possible_path_count_list = [0, 1, 2, 'misc']`  
Counts above `2` map to `misc`.

The feature order is:

- 2-hop: `[count_1, count_2]`
- 3-hop: `[count_1, count_2, count_3]`
- 4-hop: `[count_1, count_2, count_3, count_4]`
---

## Storage

Graphs are serialised into a single HDF5 file (`data_processed.h5`) using
flat concatenated arrays with boundary pointers:

| Dataset | dtype | Shape |
|---------|-------|-------|
| `node_feat` | int8 | (total_nodes, 10) |
| `node_embd` | float16 | (total_nodes, 17) |
| `edge_feat` | int8 | (total_edges, 5) |
| `edge_index` | int8 | (2, total_edges) |
| `edge_index_2hop` | int8 | (2, total_2hop_edges) |
| `edge_feat_2hop` | int8 | (total_2hop_edges, 2) |
| `edge_ptr_2hop` | int32 | (num_molecules + 1,) |
| `edge_index_3hop` | int8 | (2, total_3hop_edges) |
| `edge_feat_3hop` | int8 | (total_3hop_edges, 3) |
| `edge_ptr_3hop` | int32 | (num_molecules + 1,) |
| `edge_index_4hop` | int8 | (2, total_4hop_edges) |
| `edge_feat_4hop` | int8 | (total_4hop_edges, 4) |
| `edge_ptr_4hop` | int32 | (num_molecules + 1,) |
| `node_ptr` | int32 | (num_molecules + 1,) |
| `edge_ptr` | int32 | (num_molecules + 1,) |
| `labels` | float32 | (num_molecules,) |

Molecule `i` occupies `node_ptr[i]:node_ptr[i+1]` in node arrays and
`edge_ptr[i]:edge_ptr[i+1]` in direct-bond edge arrays, with analogous 2-hop/3-hop/4-hop
bounds via `edge_ptr_2hop`, `edge_ptr_3hop`, and `edge_ptr_4hop`.

---

## Model Integration (Recommended)

The continuous features in `node_rwpe` serve three semantically distinct
roles — **structural** (RWPE), **electronic** (EN, Gasteiger), and **spatial**
(3D coords). Separating their projection pathways avoids forcing shared
capacity on unrelated signals:

```
PE_embed   = MLP(node_rwpe[:, :12])          # structural topology
chem_embed = MLP(node_rwpe[:, 12:14])        # electronic properties
coord_embed = MLP(node_rwpe[:, 14:17])       # spatial geometry

atom_embed = Σ_i Embed_i(node_feat[:, i])    # discrete chemistry

node_embed = atom_embed + PE_embed + chem_embed + coord_embed
```

### Why MLP, not Linear, for RWPE?

A single linear layer treats the 12 walk lengths as independent signals.
An MLP (2–3 layers, e.g. 12 → 64 → d with GELU) can learn nonlinear
combinations — the ratio of short-range to long-range return probabilities
distinguishes ring nodes from chain nodes, fused junctions from terminal
atoms. This is the single largest upgrade over a naive `Linear(rwpe)` and is
the standard approach on PCQM4Mv2 (GraphGPS, Rampášek et al. 2022).

### Why sum, not concatenation?

Addition is cheaper (no extra projection) and performs comparably when the
hidden dimension is large enough. Concatenation + projection is a valid
alternative if the model struggles to disentangle the streams.

---

## Activation Layer

MoTanh (`layer/`) provides a learnable, bounded, monotone activation for
embedding continuous scalars into (−1, 1). It is a weighted sum of `tanh(s_i · x)`
with softplus-normalised mixing weights and softplus-positive scales. See
`layer/README.md` for properties and expressivity analysis.

Potential use: apply MoTanh as the nonlinearity inside the continuous-feature
MLPs above, replacing GELU/ReLU where a bounded, order-preserving embedding is
desirable (e.g. for the RWPE or coordinate pathways).

---

## Design Principles

1. **Explicit over implicit.** Features like EN and Gasteiger charge are
   theoretically recoverable by a deep GNN, but providing them directly
   reduces learning burden and helps shallow architectures.

2. **Discrete for categorical, continuous for scalar.** Atomic number and
   ring membership are vocabulary indices; EN and RWPE are floats. No
   unnecessary binning or sin/cos encoding of bounded scalar values.

3. **Minimal graph size.** Only chemically relevant hydrogens (N–H, O–H,
   S–H) are kept as nodes. Non-active H count is preserved as a feature.

4. **Motivated vocabulary sizes.** Ring sizes capped at 8 (covers >99% of
   drug-like molecules), H count at 8, rotatable bonds as binary. Each
   boundary is justified by distribution analysis on 10k PCQM4Mv2 samples.

5. **Robustness.** Gasteiger charge NaN-guarded, EN double-fallback
   (RDKit API → curated table → carbon default), coordinates zero-filled
   when conformer absent.

6. **Absolute per-atom over pairwise edge encodings.** EN and Gasteiger
   charge are stored as per-atom (node) values, not as per-bond (edge)
   differences or sums. Three alternatives were compared:

   | Encoding | What it captures | What it loses |
   |---|---|---|
   | Absolute per atom: \(x_i\) | Full information | Nothing (difference and sum are derivable in one message-passing step) |
   | Difference per edge: \(x_i - x_j\) | Bond polarization | Absolute energy level (mean of the pair) |
   | Sum per edge: \(x_i + x_j\) | Bond energy level | Polarization direction (spread of the pair) |

   Difference and sum are complementary projections — together they form a
   lossless transform \((x_i, x_j) \leftrightarrow \bigl(\tfrac{x_i+x_j}{2},\, x_i - x_j\bigr)\) —
   but each alone discards exactly the information the other preserves.
   Absolute per-atom values are strictly dominant: they are already lossless,
   and any pairwise function (difference, sum, product) is trivially
   derivable in a single GNN layer via \(m_{ij} = f(h_i, h_j)\).

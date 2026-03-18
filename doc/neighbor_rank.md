# Neighbor Rank — Ordered Bond Feature for Chirality

## Problem: chirality without order is meaningless

Tetrahedral chirality (`CHI_TETRAHEDRAL_CW` / `CHI_TETRAHEDRAL_CCW`) is a
relative concept: it describes whether the substituents around a chiral center
appear in clockwise or counter-clockwise order when viewed from a specific
direction. The RDKit tag returned by `GetChiralTag()` is defined with respect
to the **sorted neighbor list** of each atom (by atom index), following the
CIP priority rules projected onto that ordering.

A standard GNN accumulates neighbor messages with a **permutation-invariant**
aggregation (sum, mean, or max). This means the model receives the same
aggregated message regardless of which neighbor sits at rank 0 and which sits
at rank 3. Given only the chirality tag and unordered neighbors, there is no
way to recover the spatial arrangement — CW vs CCW is structurally undefined.

## Solution: neighbor rank as a directed edge feature

For each **directed** edge `(u -> v)`, store the **rank of v in u's sorted
neighbor list**, where neighbors are sorted by their new (post-H-removal) atom
index.

```
Atom u has kept-atom neighbors [2, 5, 8, 11] (sorted by new index)
Edge (u -> 2):  neighbor_rank = 0
Edge (u -> 5):  neighbor_rank = 1
Edge (u -> 8):  neighbor_rank = 2
Edge (u -> 11): neighbor_rank = 3
```

The receiving atom `u` can now, in a single message-passing step, reconstruct
the full ordered sequence of its neighbors: message from `v` carries the
information "I am neighbor #k of yours." Combined with the CW/CCW chirality
tag on `u`, the model can determine the 3D arrangement.

## Why this satisfies the stated constraints

| Constraint | Satisfied? | How |
|---|---|---|
| Not absolute position | Yes | Rank is local: it indexes neighbors of a single atom, not a global node ordering |
| Not relative distance | Yes | No metric, hop count, or spatial quantity — purely ordinal within a neighborhood |
| Encodes order | Yes | Two atoms with swapped neighbors produce swapped ranks on their respective edges |

## Consistency with the chirality tag after H-removal

`mol_to_graph` in `src/dataset/graph.py` removes non-active hydrogens and
remaps indices via `old_to_new`:

```python
old_to_new = -np.ones(mol.GetNumAtoms(), dtype=np.int32)
old_to_new[keep_idx] = np.arange(len(keep_idx))
```

Because `keep_idx = np.flatnonzero(keep_atom)` is already sorted, the mapping
is **order-preserving**: if original atom `i < j` are both kept, their new
indices satisfy `new_i < new_j`. The relative ordering of kept neighbors is
therefore identical before and after the remapping. Since RDKit assigns the
chirality tag based on the relative ordering of neighbor indices, the tag
remains valid when neighbor rank is computed on new indices.

## First asymmetric direct-bond edge feature

All five previous edge features (`bond_type`, `stereo`, `is_conjugated`,
`is_rotatable`, `ring_size`) are symmetric: both directions `(u -> v)` and
`(v -> u)` receive the same feature vector.

`neighbor_rank` is inherently **directional** — the rank of `v` in `u`'s
neighborhood is generally different from the rank of `u` in `v`'s
neighborhood. The two directed copies of a bond now carry different feature
vectors in slot #5 only. This asymmetry is a prerequisite for encoding
chirality: symmetric features by definition cannot distinguish CW from CCW.

## Implementation

### `src/dataset/features.py`

```python
'possible_neighbor_rank_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
```

Same range as `possible_degree_list` (degree ≤ 10 covers all atoms in the
dataset). Rank is always strictly less than degree, so no extra headroom is
needed.

### `src/dataset/graph.py` — inside `mol_to_graph`

After collecting bonds into `direct_bonds`, the neighbor lists are sorted once
and converted to O(1) lookup dictionaries:

```python
for neighbors in neighbor_lists:
    neighbors.sort()

rank_lookup = [
    {neighbor: rank for rank, neighbor in enumerate(neighbors)}
    for neighbors in neighbor_lists
]
```

Then each bond produces two directed edge features with different rank slots:

```python
for ni, nj, bond in direct_bonds:
    edge_feature = bond_features(bond, rotatable_bond_indices)

    rank_fwd = rank_lookup[ni].get(nj, ...)   # rank of nj in ni's neighbors
    rank_rev = rank_lookup[nj].get(ni, ...)   # rank of ni in nj's neighbors

    edge_feature_fwd = edge_feature + [vocab_index(..., rank_fwd)]
    edge_feature_rev = edge_feature + [vocab_index(..., rank_rev)]
```

### `src/dataset/dataset.py`

`EDGE_FEAT_VOCAB_SIZES[""]` now has 6 entries. The offset and total-vocab
calculations in `_compute_offsets` derive from this list automatically; no
other dataset code requires changes.

## Vocabulary

`possible_neighbor_rank_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']`

| Index | Meaning |
|---|---|
| 0–10 | Exact rank of the destination in the source's sorted neighbor list |
| misc (11) | Fallback; should never occur for valid bonds |

## Interaction with existing features

| Feature | Role | Interaction with neighbor rank |
|---|---|---|
| `chirality` (node #1) | CW or CCW tag at the chiral center | Requires rank to be interpretable |
| `bond_stereo` (edge #1) | E/Z tag on double bonds | Independent; double bond stereo uses bond geometry, not neighbor rank |
| `degree` (node #2) | Number of bonds | Upper bound on rank: rank < degree |

## Why not use 3D coordinates instead?

3D coordinates (`node_embd[:, 12:15]`) are available only for the training SDF
and are zero-filled at inference. Neighbor rank is derived purely from graph
topology and is always available without a conformer.

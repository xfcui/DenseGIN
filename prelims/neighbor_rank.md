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

For each **directed** edge `(u -> v)`, store the **rank of u in v's sorted
neighbor list**, where neighbors are sorted by their new (post-H-removal) atom
index — **but only when v carries `CHI_TETRAHEDRAL_CW` or
`CHI_TETRAHEDRAL_CCW`**. For all other chirality values the rank is set to
`misc`, signalling that no ordering information is defined or needed.

```
Atom v: CHI_TETRAHEDRAL_CW, neighbors [2, 5, 8, 11] (sorted by new index)
Edge (2  -> v): neighbor_rank = 0   # 2  is rank-0 neighbor of v
Edge (5  -> v): neighbor_rank = 1   # 5  is rank-1 neighbor of v
Edge (8  -> v): neighbor_rank = 2
Edge (11 -> v): neighbor_rank = 3

Atom w: CHI_UNSPECIFIED, neighbors [1, 3]
Edge (1 -> w): neighbor_rank = misc
Edge (3 -> w): neighbor_rank = misc
```

Each incoming message to a chiral centre now carries the sender's position in
the centre's ordered neighbor list: "I am neighbor #k of yours." The receiving
atom can reconstruct the full ordered sequence from its incoming messages.
Combined with the CW/CCW chirality tag, it can determine the 3D arrangement.

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
'possible_neighbor_rank_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
```

Same range as `possible_degree_list` (degree ≤ 6 covers all atoms in the
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

Then each bond produces two directed edge features with different rank slots.
The rank is set to `misc` whenever the **target (destination)** atom is not a
tetrahedral chiral centre (`CHI_TETRAHEDRAL_CW` / `CHI_TETRAHEDRAL_CCW`):

```python
_TETRAHEDRAL_CHIRAL_TAGS = {'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'}
tetrahedral_chiral = {
    int(old_to_new[atom.GetIdx()])
    for atom in mol.GetAtoms()
    if keep_atom[atom.GetIdx()] and str(atom.GetChiralTag()) in _TETRAHEDRAL_CHIRAL_TAGS
}

_RANK_MISC = len(FEATURE_VOCAB['possible_neighbor_rank_list']) - 1

for ni, nj, bond in direct_bonds:
    edge_feature = bond_features(bond, rotatable_bond_indices)

    # edge (ni -> nj): rank of ni in nj's neighbor list, gated on nj being chiral
    if nj in tetrahedral_chiral:
        rank_fwd = vocab_index(..., rank_lookup[nj].get(ni, _RANK_MISC))
    else:
        rank_fwd = _RANK_MISC

    # edge (nj -> ni): rank of nj in ni's neighbor list, gated on ni being chiral
    if ni in tetrahedral_chiral:
        rank_rev = vocab_index(..., rank_lookup[ni].get(nj, _RANK_MISC))
    else:
        rank_rev = _RANK_MISC

    edge_feature_fwd = edge_feature + [rank_fwd]
    edge_feature_rev = edge_feature + [rank_rev]
```

### `src/dataset/dataset.py`

`EDGE_FEAT_VOCAB_SIZES[""]` now has 6 entries. The offset and total-vocab
calculations in `_compute_offsets` derive from this list automatically; no
other dataset code requires changes.

## Vocabulary

`possible_neighbor_rank_list = [0, 1, 2, 3, 4, 5, 6, 'misc']`

| Index | Meaning |
|---|---|
| 0–6 | Rank of the **source** atom in the **destination**'s sorted neighbor list (destination is `CHI_TETRAHEDRAL_CW` or `CHI_TETRAHEDRAL_CCW`) |
| misc (7) | Destination is not a tetrahedral chiral centre, or rank exceeds 6 |

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

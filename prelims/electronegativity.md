# Electronegativity Atom Feature

## Motivation

The HOMO-LUMO gap is a purely electronic property: the energy difference between the highest occupied and lowest unoccupied molecular orbital. Electronegativity (EN) directly determines how strongly an atom attracts bonding electrons, thereby shaping the energy levels of frontier orbitals:

- Electron-withdrawing atoms (high EN: O, F, N) lower both HOMO and LUMO energies and tend to narrow the gap in donor-acceptor conjugated systems.
- Electron-donating atoms (low EN: metals, Si) raise HOMO energy and typically widen the gap.
- EN differences between bonded atoms create bond polarisation, breaking orbital symmetry and perturbing the gap.

While atomic number is already present as a categorical feature, EN is a non-monotonic, non-linear function of Z that a GNN must otherwise implicitly recover by memorising periodic-table trends — encoding it explicitly removes that burden.

## Encoding

EN is stored as a single continuous float appended to the `node_rwpe` array:

```
node_rwpe shape: (num_atoms, RWPE_DIM + EN_DIM)  # 12 + 1 = 13
```

The value is **carbon-centered** (i.e. `EN_atom - EN_C`) so that the most common organic atom maps to `0.0`. This keeps the feature near zero on average and makes the model's initialisation easier.

Centering on carbon also means the feature naturally represents relative electron-withdrawing/donating character with respect to a carbon baseline, which is the quantity most directly relevant to frontier orbital perturbation in organic molecules.

## Source

EN values are sourced via priority:

1. **RDKit `GetPaulingElectronegativity`** — used if available in the installed RDKit build.
2. **Curated fallback table** (`_PAULING_EN` in `src/dataset.py`) — used when RDKit does not expose the API. Covers all elements likely to appear in PCQM4Mv2: H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, transition metals Sc–Zn, Ga, Ge, As, Se, Br, I.
3. **Carbon value** — any element absent from both sources falls back to `_CARBON_EN`, yielding a centered value of `0.0`.

## Why not discretise?

The existing atom features (atomic number, hybridisation, etc.) are categorical indices fed through embedding tables. EN could be binned the same way, but:

- Binning loses resolution at no benefit; the model processes it as a float either way after the first linear layer.
- A single float concatenated to the RWPE vector costs nothing extra in parameters and avoids the need to pick bin boundaries.

## Why not sin/cos encoding?

Sin/cos positional encoding is useful for **unbounded integer sequences** (e.g. transformer position indices from 0 to thousands) where raw values would dominate learned embeddings in scale. EN is a small, bounded continuous value (~0.7–4.0). A GNN linear layer can learn any affine rescaling trivially, so sin/cos adds no information over the raw centered value.

## Integration

`node_rwpe[:, :RWPE_DIM]` — 12-dimensional random-walk positional encoding.  
`node_rwpe[:, RWPE_DIM]`  — centered Pauling electronegativity (this feature).

The model reads both as a single array and is responsible for projecting the combined continuous features into its hidden dimension, typically via a shared linear layer.

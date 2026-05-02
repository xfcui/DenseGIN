# Gasteiger Partial Charge Atom Feature

## Motivation

The HOMO-LUMO gap is determined by the energy of the frontier orbitals, which in turn depends on how electron density is distributed across the molecule. While electronegativity (EN) encodes an atom's *intrinsic* tendency to attract electrons, it is a per-element constant — every nitrogen atom gets the same EN value regardless of its chemical context.

Gasteiger partial charge is the context-dependent counterpart: it encodes the *actual* partial charge an atom carries within the specific molecular environment, shaped by inductive and resonance effects from its neighbors. This makes it directly relevant to:

- Identifying electron-rich atoms that contribute strongly to the HOMO.
- Identifying electron-poor atoms that lower the LUMO.
- Encoding bond polarization in heterocyclic and conjugated systems, where small charge differences produce large gap shifts.

## Algorithm

The Gasteiger-Marsili method is a purely empirical, topology-based iterative scheme — no quantum mechanics or 3D geometry is required:

1. Each element has empirically fitted coefficients `a`, `b`, `c` describing how EN varies with formal charge: `EN(q) = a + b·q + c·q²`.
2. For each bond, charge flows from the atom with lower EN to the atom with higher EN, proportional to the EN difference.
3. Transferred charge is damped by a factor of `0.5^k` at iteration `k`, ensuring convergence within ~6–8 iterations.
4. The process terminates when charges change by less than a threshold.

Charges sum to the net formal charge of the molecule (zero for neutral molecules).

## Relationship to Electronegativity

Gasteiger charges are initialized from Pauling EN values and iteratively propagated along the molecular graph. This means they are, in principle, derivable from the EN feature already present in `node_rwpe` via multi-hop message passing. A sufficiently deep GNN could recover them implicitly.

However, providing them explicitly:
- Gives shallow GNNs access to pre-computed multi-hop electronic information at zero extra depth.
- Reduces the learning burden on early layers, freeing capacity for higher-level pattern recognition.
- Provides a stable, interpretable initialization for the model's understanding of charge distribution.

## Encoding

Gasteiger charge is stored as a single continuous float appended to the `node_rwpe` array after the EN feature:

```
node_rwpe shape: (num_atoms, RWPE_DIM + EN_DIM + GC_DIM)  # 12 + 1 + 1 = 14
```

Layout:
- `node_rwpe[:, :RWPE_DIM]` — 12-dimensional random-walk positional encoding.
- `node_rwpe[:, RWPE_DIM]` — carbon-centered Pauling electronegativity.
- `node_rwpe[:, RWPE_DIM + EN_DIM]` — Gasteiger partial charge (this feature).

The value is used **uncentered**; since neutral molecules sum to zero and carbon (the most common element) typically sits near zero, the dataset-level mean is already close to `0.0`.

## Value Range

Typical range in organic drug-like molecules:

| Atom | Typical Range |
|------|--------------|
| H | +0.05 to +0.15 |
| C | -0.15 to +0.15 |
| N | -0.35 to -0.05 |
| O | -0.55 to -0.25 |
| F | -0.30 to -0.15 |
| S | -0.15 to +0.10 |
| Cl/Br | -0.15 to -0.05 |

The overall range is approximately **-0.8 to +0.8**, with a mean near **0.0**.

## Robustness

RDKit's `ComputeGasteigerCharges` can produce `NaN` or numerically extreme values for atoms with unusual valence, radicals, or certain metal coordination environments. The helper `_get_gasteiger_charge` in `src/dataset.py` guards against this:

- `NaN` values are replaced with `0.0`.
- Values with `|gc| > 4.0` are treated as numerical failures and replaced with `0.0`.
- If `ComputeGasteigerCharges` raises an exception (e.g. unsupported element), the entire molecule falls back to `0.0` for all atoms.

## Implementation

Computed in `smiles2graph` via RDKit after `AddHs`:

```python
rdPartialCharges.ComputeGasteigerCharges(mol)
```

Collected per atom via `_get_gasteiger_charge(atom)`, which reads the `_GasteigerCharge` atom property set by RDKit. Concatenated into `node_rwpe` alongside RWPE and EN.

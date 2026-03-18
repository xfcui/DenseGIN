# PCQM4Mv2 HOMO–LUMO Gap Prediction

JAX/Equinox pipeline for predicting HOMO–LUMO gaps on the [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) dataset (~3.8 M molecules, target: eV, metric: MAE).

## Repository Layout

```
src/
  dataset/          # Graph preprocessing, HDF5 I/O, feature extraction
  dataset.py        # Compatibility re-export of PCQMDataset / PCQMDataloader
tst/
  test_dataset_and_dataloader.py
doc/
  design.md         # Full feature engineering rationale
  rwpe/             # Random-walk positional encoding analysis
  k_hop/            # K-hop edge construction
  ring_size/        # Ring-size feature analysis
  rot_bond/         # Rotatable-bond feature analysis
  diameter/         # Molecular diameter analysis
  act_h_atom/       # Active-hydrogen retention analysis
  mix_tanh/         # MoTanh activation analysis
results/            # Prediction artefacts (.npz)
```

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `jax`, `equinox`, `optax`, `rdkit`, `ogb`, `h5py`, `numpy`, `pandas`, `tqdm`.

## Data Preparation

Pre-process raw SMILES into HDF5 (expects `dataset/pcqm4m-v2/raw/data.csv.gz` and the 3D SDF):

```bash
python -m src.dataset   # or whichever preprocessing entry point applies
```

Output: `dataset/pcqm4m-v2/processed/data_processed.h5` and `split_dict.h5`.

## Public API

### `PCQMDataset`
- **Signature**: `PCQMDataset(dataset_root=None, split="train", split_file=None, load_in_memory=True)`
- **Purpose**: Loads preprocessed molecular graphs from HDF5; provides `batch_collapse` to merge a list of graph IDs into one batched graph block.
- **Usage**:
  ```python
  from src.dataset import PCQMDataset
  ds = PCQMDataset(split="train")
  batch = ds.batch_collapse([0, 1, 2, 3])
  ds.close()
  ```

### `PCQMDataloader`
- **Signature**: `PCQMDataloader(dataset, *, indices=None, batch_size=1, shuffle=False, drop_last=False, pad_to_multiple=1024, seed=None)`
- **Purpose**: Iterator that yields batched graph dicts from a `PCQMDataset`.
- **Usage**:
  ```python
  from src.dataset import PCQMDataset, PCQMDataloader
  ds = PCQMDataloader(PCQMDataset(split="train"), batch_size=256, shuffle=True)
  for batch in ds:
      node_feat = batch["node_feat"]   # int32 (total_nodes, 10)
      node_embd = batch["node_embd"]   # float32 (total_nodes, 17)
      edge_index = batch["edge_index"] # int32 (2, total_edges)
      labels = batch["labels"]         # float32 (batch_size,)
  ```

### `batch_collapse`
- **Signature**: `batch_collapse(dataset: PCQMDataset, graph_ids: Sequence[int], *, pad_to_multiple: int = 1024) -> Dict[str, np.ndarray]`
- **Purpose**: Functional alias for `dataset.batch_collapse`.

### Feature constants
| Name | Description |
|---|---|
| `NODE_FEAT_VOCAB_SIZES` | Per-feature vocabulary sizes (list of 10 ints) |
| `NODE_FEAT_TOTAL_VOCAB` | Total embedding table size for node features |
| `EDGE_FEAT_VOCAB_SIZES` | Per-edge-kind vocab sizes (`""`, `"_2hop"`, `"_3hop"`, `"_4hop"`) |
| `EDGE_FEAT_TOTAL_VOCAB` | Total embedding table size per edge kind |

## Batch Format

Each batch dict contains:

| Key | dtype | Shape | Description |
|---|---|---|---|
| `node_feat` | int32 | `(N, 10)` | Discrete node features with vocab offsets applied |
| `node_embd` | float32 | `(N, 17)` | Continuous: RWPE×12, EN×1, Gasteiger×1, 3D coords×3 |
| `node_ptr` | int32 | `(G+2,)` | Graph boundaries in node array |
| `node_batch` | int32 | `(N,)` | Graph index per node |
| `edge_index` | int32 | `(2, E)` | Direct-bond edge indices |
| `edge_feat` | int32 | `(E, 5)` | Direct-bond edge features with offsets |
| `edge_ptr` | int32 | `(G+2,)` | Graph boundaries in edge array |
| `edge_batch` | int32 | `(E,)` | Graph index per edge |
| `edge_2hop_index` | int32 | `(2, E₂)` | 2-hop edges |
| `edge_3hop_index` | int32 | `(2, E₃)` | 3-hop edges |
| `edge_4hop_index` | int32 | `(2, E₄)` | 4-hop edges |
| `labels` | float32 | `(B,)` | HOMO–LUMO gap (eV) |
| `molecule_ids` | int64 | `(B,)` | Original dataset indices |
| `batch_n_graphs` | int32 | scalar | Number of real graphs (`B`) |

`N`, `E`, `E₂`… are padded to multiples of `PAD_TO_MULTIPLE` (1024). The first graph in every batch is a zero-filled null graph that absorbs padding.

## Invariants

- Datasets sharing the same `data_file` path re-use a shared in-process cache; initialising `PCQMDataset(split="train")` and `PCQMDataset(split="valid")` loads HDF5 arrays only once.
- Molecules with `label < 0` are filtered out at dataset construction time.
- Node/edge feature indices have per-feature offsets pre-applied; use the full `*_TOTAL_VOCAB` size for embedding tables.
- All edge indices are absolute (graph node offsets already applied) and in `int32`.
- Null graph is always batch index `0`; real graph indices start at `1`.

## Quick Smoke Test

```bash
python -m pytest tst/ -q
```

## Feature Engineering

See [`doc/design.md`](doc/design.md) for the complete feature design rationale, including hydrogen handling, RWPE, electronegativity, Gasteiger charges, k-hop edges, and model integration recommendations.

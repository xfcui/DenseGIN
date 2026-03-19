# PCQM4Mv2 HOMO–LUMO Gap Prediction

JAX/Equinox pipeline for predicting HOMO–LUMO gaps on the [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) dataset (~3.8 M molecules, target: eV, metric: MAE).

## Repository Layout

```
src/
  dataset/          # Graph preprocessing, HDF5 I/O, feature extraction
    dataset.py      # PCQMDataset, batch_collapse, feature constants
    dataloader.py   # PCQMDataloader
    dataprocess.py  # SMILES → HDF5 preprocessing pipeline
    features.py     # FEATURE_VOCAB, atom/bond feature encoders
    graph.py        # RWPE, k-hop edges, Gasteiger, electronegativity
    hdf5.py         # HDF5 save/load helpers
  dataset.py        # Compatibility re-export of PCQMDataset / PCQMDataloader
  model.py          # DenseGIN architecture (Equinox)
  train.py          # Training loop (Adan optimizer + geometric LR schedule)
tst/
  test_dataset_and_dataloader.py
  test_model_compat.py
  test_train.py
doc/
  design.md         # Full feature engineering rationale
  rwpe/             # Random-walk positional encoding analysis
  k_hop/            # K-hop edge construction
  ring_size/        # Ring-size feature analysis
  rot_bond/         # Rotatable-bond feature analysis
  diameter/         # Molecular diameter analysis
  act_h_atom/       # Active-hydrogen retention analysis
  mix_tanh/         # MoTanh activation analysis
results/            # Prediction artefacts and saved model weights (.eqx)
```

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `jax`, `equinox`, `optax`, `rdkit`, `ogb`, `h5py`, `numpy`, `pandas`, `tqdm`.

## Data Preparation

Pre-process raw SMILES into HDF5 (expects `dataset/pcqm4m-v2/raw/data.csv.gz` and the 3D SDF):

```bash
python -m src.dataset.dataprocess
```

Output: `dataset/pcqm4m-v2/processed/data_processed.h5` and `split_dict.h5`.

## Training

```bash
python -m src.train \
  --batch_size 256 \
  --learning_rate 3e-3 \
  --weight_decay 2e-2 \
  --scheduler_period 8 \
  --model_save_path results/best_model.eqx
```

This trains for `scheduler_period²` epochs using an **Adan** optimizer with a geometric cosine-annealing LR schedule. The best checkpoint (by validation MAE) is saved to `--model_save_path`.

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 256 | Graphs per batch (actual batch is `batch_size - 1` real graphs) |
| `--learning_rate` | `3e-3` | Peak learning rate |
| `--weight_decay` | `2e-2` | L2 regularisation on `kernel` arrays |
| `--scheduler_period` | 8 | Period *k* for geometric LR schedule; trains for *k²* epochs |
| `--model_save_path` | `results/best_model.eqx` | Path for best model checkpoint |

## Model

`DenseGIN` (`src/model.py`) is the core architecture:

- **Atom encoder**: multi-feature embedding (`EmbedLayer`) + positional projection via `GatedLinearBlock` on RWPE-12 features.
- **Message passing** (depth=5 layers): each layer runs a `MixerKernel` (1-hop + 2-hop + 3-hop + 4-hop `ConvKernel`s with a `VirtKernel` virtual node) followed by a `MetaFormerBlock`.
- **Readout**: `HeadKernel` sums node features, normalises, fuses the virtual node, and projects to a scalar HOMO–LUMO gap.

Default hyperparameters: `depth=5, width=256, num_head=16, dim_head=16`.

### Key design choices

- `ConvKernel`: bond-aware difference messages with degree-normalised aggregation (inspired by PNA/Graphormer).
- `MixerKernel`: learnable softmax-normalised mixing of multi-hop and virtual-node messages.
- `MetaFormerBlock`: two-step MetaFormer update (ReZero/LayerScale style) with `GatedLinearBlock` (GLU variant).
- Loss: smooth MAE — below a threshold `δ=0.06 eV` residuals are quadratically penalised; above, linearly.

## Public API

### `PCQMDataset`

```python
PCQMDataset(dataset_root=None, split="train", split_file=None, load_in_memory=True)
```

Loads preprocessed molecular graphs from HDF5; provides `batch_collapse` to merge graph IDs into one batched block.

```python
from src.dataset import PCQMDataset
ds = PCQMDataset(split="train")
batch = ds.batch_collapse([0, 1, 2, 3])
ds.close()
```

### `PCQMDataloader`

```python
PCQMDataloader(dataset, *, indices=None, batch_size=1, shuffle=False, drop_last=False, pad_to_multiple=1024, seed=None)
```

Iterator that yields batched graph dicts from a `PCQMDataset`.

```python
from src.dataset import PCQMDataset, PCQMDataloader
ds = PCQMDataloader(PCQMDataset(split="train"), batch_size=256, shuffle=True)
for batch in ds:
    node_feat  = batch["node_feat"]    # int32  (N, 10)
    node_embd  = batch["node_embd"]    # float32 (N, 17)
    edge_index = batch["edge_index"]   # int32  (2, E)
    labels     = batch["labels"]       # float32 (B,)
```

### `batch_collapse`

```python
batch_collapse(dataset: PCQMDataset, graph_ids: Sequence[int], *, pad_to_multiple: int = 1024) -> Dict[str, np.ndarray]
```

Functional alias for `dataset.batch_collapse`.

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
| `edge_feat` | int32 | `(E, 6)` | Direct-bond edge features with offsets |
| `edge_ptr` | int32 | `(G+2,)` | Graph boundaries in edge array |
| `edge_batch` | int32 | `(E,)` | Graph index per edge |
| `edge_2hop_index` | int32 | `(2, E₂)` | 2-hop edge indices |
| `edge_2hop_feat` | int32 | `(E₂, 2)` | 2-hop edge features |
| `edge_2hop_batch` | int32 | `(E₂,)` | Graph index per 2-hop edge |
| `edge_3hop_index` | int32 | `(2, E₃)` | 3-hop edge indices |
| `edge_3hop_feat` | int32 | `(E₃, 3)` | 3-hop edge features |
| `edge_3hop_batch` | int32 | `(E₃,)` | Graph index per 3-hop edge |
| `edge_4hop_index` | int32 | `(2, E₄)` | 4-hop edge indices |
| `edge_4hop_feat` | int32 | `(E₄, 4)` | 4-hop edge features |
| `edge_4hop_batch` | int32 | `(E₄,)` | Graph index per 4-hop edge |
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
- Direct-bond (`edge_feat`) has 6 features; 2-/3-/4-hop edges carry 2/3/4 path-count features respectively.

## Quick Smoke Test

```bash
python -m pytest tst/ -q
```

## Feature Engineering

See [`doc/design.md`](doc/design.md) for the complete feature design rationale, including hydrogen handling, RWPE, electronegativity, Gasteiger charges, k-hop edges, and model integration recommendations.

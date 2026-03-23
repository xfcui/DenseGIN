# PCQM4Mv2 HOMO–LUMO Gap Prediction

JAX/Equinox pipeline for predicting HOMO–LUMO gaps on the [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) dataset (~3.8 M molecules, target: eV, metric: MAE).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess raw SMILES → HDF5
#    Expects: dataset/pcqm4m-v2/raw/data.csv.gz  and  the 3D SDF archive
python -m src.dataset.dataprocess

# 3. Train
python -m src.train --model_save_path results/best_model.eqx

# 4. Smoke test
python -m pytest tst/ -q
```

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
  model.py          # DuAxMPNN architecture (Equinox)
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

## Training

```bash
python -m src.train \
  --batch_size      256             \
  --learning_rate   3e-3            \
  --weight_decay    2e-2            \
  --scheduler_period 8             \
  --model_save_path results/best_model.eqx
```

Trains for `scheduler_period²` epochs using an **Adan** optimizer with a geometric cosine-annealing LR schedule. The best checkpoint (by validation MAE) is written to `--model_save_path`.

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 256 | Graphs per batch (first graph is a zero-filled null padding graph) |
| `--learning_rate` | `3e-3` | Peak learning rate |
| `--weight_decay` | `2e-2` | L2 regularisation on `kernel` arrays |
| `--scheduler_period` | `8` | Period *k*; total epochs = *k²* |
| `--model_save_path` | `results/best_model.eqx` | Checkpoint path |

## Model

`DuAxMPNN` (`src/model.py`) with default hyperparameters `depth=5, width=256, num_head=16, dim_head=16`.

| Stage | Component | Notes |
|---|---|---|
| **Atom encoder** | `EmbedLayer` + `GatedLinearBlock` | Multi-feature embedding; RWPE-12 positional projection |
| **Message passing** × 5 | `MixerKernel` + `MetaFormerBlock` | 1/2/3/4-hop `ConvKernel`s + `VirtKernel` virtual node |
| **Readout** | `HeadKernel` | Sum → normalise → fuse virtual node → scalar gap |

**Key design choices:**

- `ConvKernel` — bond-aware difference messages with degree-normalised aggregation (inspired by PNA/Graphormer).
- `MixerKernel` — learnable softmax-normalised mixing of multi-hop and virtual-node messages.
- `MetaFormerBlock` — two-step MetaFormer update (ReZero/LayerScale style) with `GatedLinearBlock` (GLU variant).
- Loss — smooth MAE: residuals are quadratically penalised below `δ=0.06 eV`, linearly above.

## Data Preparation

```bash
python -m src.dataset.dataprocess
```

Reads `dataset/pcqm4m-v2/raw/data.csv.gz` (and the 3D SDF archive) and writes:

- `dataset/pcqm4m-v2/processed/data_processed.h5`
- `dataset/pcqm4m-v2/processed/split_dict.h5`

## Public API

### `PCQMDataset`

```python
PCQMDataset(dataset_root=None, split="train", split_file=None, load_in_memory=True)
```

Loads preprocessed molecular graphs from HDF5. Datasets sharing the same `data_file` path share an in-process cache, so `PCQMDataset(split="train")` and `PCQMDataset(split="valid")` load HDF5 arrays only once.

```python
from src.dataset import PCQMDataset
ds = PCQMDataset(split="train")
batch = ds.batch_collapse([0, 1, 2, 3])
ds.close()
```

### `PCQMDataloader`

```python
PCQMDataloader(dataset, *, indices=None, batch_size=1, shuffle=False,
               drop_last=False, pad_to_multiple=1024, seed=None)
```

Iterator that yields batched graph dicts from a `PCQMDataset`.

```python
from src.dataset import PCQMDataset, PCQMDataloader
dl = PCQMDataloader(PCQMDataset(split="train"), batch_size=256, shuffle=True)
for batch in dl:
    node_feat  = batch["node_feat"]    # int32   (N, 10)
    node_embd  = batch["node_embd"]    # float32 (N, 17)
    edge_index = batch["edge_index"]   # int32   (2, E)
    labels     = batch["labels"]       # float32 (B,)
```

### `batch_collapse`

```python
batch_collapse(dataset: PCQMDataset, graph_ids: Sequence[int],
               *, pad_to_multiple: int = 1024) -> Dict[str, np.ndarray]
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

`N`, `E`, `E₂`… are padded to multiples of `PAD_TO_MULTIPLE` (1024). The first graph in every batch is a zero-filled null graph that absorbs padding; real graph indices start at `1`.

| Key | dtype | Shape | Description |
|---|---|---|---|
| `node_feat` | int32 | `(N, 10)` | Discrete node features (vocab offsets pre-applied) |
| `node_embd` | float32 | `(N, 17)` | Continuous: RWPE×12, EN×1, Gasteiger×1, 3D coords×3 |
| `node_ptr` | int32 | `(G+2,)` | Graph boundaries in node array |
| `node_batch` | int32 | `(N,)` | Graph index per node |
| `edge_index` | int32 | `(2, E)` | Direct-bond edge indices (absolute, int32) |
| `edge_feat` | int32 | `(E, 6)` | Direct-bond edge features (6 features, offsets applied) |
| `edge_ptr` | int32 | `(G+2,)` | Graph boundaries in edge array |
| `edge_batch` | int32 | `(E,)` | Graph index per edge |
| `edge_2hop_index` | int32 | `(2, E₂)` | 2-hop edge indices |
| `edge_2hop_feat` | int32 | `(E₂, 2)` | 2-hop path-count features |
| `edge_2hop_batch` | int32 | `(E₂,)` | Graph index per 2-hop edge |
| `edge_3hop_index` | int32 | `(2, E₃)` | 3-hop edge indices |
| `edge_3hop_feat` | int32 | `(E₃, 3)` | 3-hop path-count features |
| `edge_3hop_batch` | int32 | `(E₃,)` | Graph index per 3-hop edge |
| `edge_4hop_index` | int32 | `(2, E₄)` | 4-hop edge indices |
| `edge_4hop_feat` | int32 | `(E₄, 4)` | 4-hop path-count features |
| `edge_4hop_batch` | int32 | `(E₄,)` | Graph index per 4-hop edge |
| `labels` | float32 | `(B,)` | HOMO–LUMO gap (eV) |
| `molecule_ids` | int64 | `(B,)` | Original dataset indices |
| `batch_n_graphs` | int32 | scalar | Number of real graphs (`B`) |

## Invariants

- Molecules with `label < 0` are filtered out at dataset construction time.
- Node/edge feature indices have per-feature offsets pre-applied; use `*_TOTAL_VOCAB` sizes for embedding tables.
- All edge indices are absolute (graph node offsets already applied) and in `int32`.

## Feature Engineering

See [`doc/design.md`](doc/design.md) for the complete feature design rationale, including hydrogen handling, RWPE, electronegativity, Gasteiger charges, k-hop edges, and model integration.

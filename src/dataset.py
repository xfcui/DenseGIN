"""Utilities for loading PCQM4Mv2 graph data and building padded batches.

The module reads the preprocessed HDF5 graph store in
`dataset/pcqm4m-v2/processed/data_processed.h5` and the split indices in
`dataset/pcqm4m-v2/split_dict.h5`, then collapses a set of molecules into a
single padded batch suitable for batched graph execution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Sequence

import h5py
import numpy as np

PAD_TO_MULTIPLE = 1024


def _load_feature_vocab() -> Dict[str, list]:
    feature_file = Path(__file__).resolve().parent / "dataset" / "features.py"
    spec = importlib.util.spec_from_file_location("_pcqm_dataset_features", feature_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load feature vocabulary module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feature_vocab = getattr(module, "FEATURE_VOCAB", None)
    if feature_vocab is None:
        raise RuntimeError("FEATURE_VOCAB not found in features module.")
    return feature_vocab


_FEATURE_VOCAB = _load_feature_vocab()

NODE_FEAT_VOCAB_SIZES = [
    len(_FEATURE_VOCAB["possible_atomic_num_list"]),
    len(_FEATURE_VOCAB["possible_chirality_list"]),
    len(_FEATURE_VOCAB["possible_degree_list"]),
    len(_FEATURE_VOCAB["possible_formal_charge_list"]),
    len(_FEATURE_VOCAB["possible_numH_list"]),
    len(_FEATURE_VOCAB["possible_number_radical_e_list"]),
    len(_FEATURE_VOCAB["possible_hybridization_list"]),
    len(_FEATURE_VOCAB["possible_is_aromatic_list"]),
    len(_FEATURE_VOCAB["possible_is_in_ring_list"]),
    len(_FEATURE_VOCAB["possible_ring_size_list"]),
]

EDGE_FEAT_VOCAB_SIZES = {
    "": [
        len(_FEATURE_VOCAB["possible_bond_type_list"]),
        len(_FEATURE_VOCAB["possible_bond_stereo_list"]),
        len(_FEATURE_VOCAB["possible_is_conjugated_list"]),
        len(_FEATURE_VOCAB["possible_is_rotable_list"]),
        len(_FEATURE_VOCAB["possible_ring_size_list"]),
    ],
    "_2hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 2,
    "_3hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 3,
    "_4hop": [len(_FEATURE_VOCAB["possible_path_count_list"])] * 4,
}

NODE_FEAT_TOTAL_VOCAB = int(1 + np.sum(NODE_FEAT_VOCAB_SIZES))
EDGE_FEAT_TOTAL_VOCAB = {
    suffix: int(1 + np.sum(size_list))
    for suffix, size_list in EDGE_FEAT_VOCAB_SIZES.items()
}


def _compute_offsets(sizes: list[int]) -> np.ndarray:
    if not sizes:
        return np.zeros(0, dtype=np.int32)
    return np.asarray([1] + list(np.cumsum(sizes[:-1], dtype=np.int32)), dtype=np.int32)


def _apply_offsets(features: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros_like(features, dtype=np.int32)
    return np.asarray(features, dtype=np.int32) + offsets[np.newaxis, :]


def _default_dataset_root() -> Path:
    return Path.cwd() / "dataset" / "pcqm4m-v2"


def _load_split_indices(split_path: Path, split_name: str, num_graphs: int) -> np.ndarray:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with h5py.File(split_path, "r") as f:
        if split_name not in f:
            available = ", ".join(sorted(f.keys()))
            raise ValueError(f"Unknown split '{split_name}'. Available splits: {available}")
        indices = np.asarray(f[split_name][()], dtype=np.int64)

    if np.any(indices < 0) or np.any(indices >= num_graphs):
        raise ValueError("Split indices contain values outside [0, num_graphs).")
    return indices


def _multiple_of(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("`multiple` must be a positive integer.")
    return ((x + multiple - 1) // multiple) * multiple


class PCQMDataset:
    """Load preprocessed PCQM4Mv2 arrays and provide batch collation helpers."""

    def __init__(
        self,
        dataset_root: Path | str | None = None,
        split: str | None = "train",
        split_file: Path | str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root or _default_dataset_root())
        self.data_file = self.dataset_root / "processed" / "data_processed.h5"
        if not self.data_file.exists():
            raise FileNotFoundError(f"Processed HDF5 not found: {self.data_file}")

        self.h5_file = h5py.File(self.data_file, "r")

        self.node_ptr = np.asarray(self.h5_file["node_ptr"][()], dtype=np.int32)
        self.node_feat = self.h5_file["node_feat"]
        self.node_embd = self.h5_file["node_embd"]
        self.labels = np.asarray(self.h5_file["labels"][()], dtype=np.float32)

        self.num_graphs = int(self.node_ptr.shape[0] - 1)
        self.edge_kinds = self._discover_edge_kinds()
        self.node_feat_vocab_sizes = list(NODE_FEAT_VOCAB_SIZES)
        self.node_feat_total_vocab = NODE_FEAT_TOTAL_VOCAB
        self.edge_ptrs = {
            suffix: np.asarray(self.h5_file[f"edge_ptr{suffix}"][()], dtype=np.int32)
            for suffix in self.edge_kinds
        }
        self.edge_feat_dtypes = {
            suffix: self.h5_file[f"edge_feat{suffix}"].dtype
            for suffix in self.edge_kinds
        }
        self.edge_feat_dims = {
            suffix: int(self.h5_file[f"edge_feat{suffix}"].shape[1])
            for suffix in self.edge_kinds
        }
        self.edge_feat_vocab_sizes = {
            suffix: EDGE_FEAT_VOCAB_SIZES.get(
                suffix, [len(_FEATURE_VOCAB["possible_path_count_list"])] * self.edge_feat_dims[suffix]
            )
            for suffix in self.edge_kinds
        }
        self.edge_feat_total_vocab = {
            suffix: int(1 + int(np.sum(size_list)))
            for suffix, size_list in self.edge_feat_vocab_sizes.items()
        }
        self._node_feat_offsets = _compute_offsets(self.node_feat_vocab_sizes)
        self._edge_feat_offsets = {
            suffix: _compute_offsets(size_list) for suffix, size_list in self.edge_feat_vocab_sizes.items()
        }

        if split is None:
            self.split_indices = np.arange(self.num_graphs, dtype=np.int64)
        else:
            split_file = Path(split_file or self.dataset_root / "split_dict.h5")
            self.split_indices = _load_split_indices(split_file, split, self.num_graphs)

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()

    def __del__(self) -> None:
        self.close()

    def _discover_edge_kinds(self) -> list[str]:
        keys = set(self.h5_file.keys())
        suffixes = []
        for key in sorted(keys):
            if key == "edge_index" or key.startswith("edge_index_"):
                suffix = key[len("edge_index"):]
                ptr = f"edge_ptr{suffix}"
                feat = f"edge_feat{suffix}"
                if ptr in keys and feat in keys:
                    suffixes.append(suffix)
        if "" not in suffixes and "edge_index" in keys:
            # Ensure direct edge stream is always included when present.
            suffixes.append("")
        return suffixes

    def get_graph_count(self) -> int:
        return self.num_graphs

    def get_split_indices(self) -> np.ndarray:
        return self.split_indices.copy()

    def _get_node_block(self, graph_id: int) -> np.ndarray:
        start, end = self.node_ptr[graph_id], self.node_ptr[graph_id + 1]
        return np.asarray(self.node_feat[start:end], dtype=self.node_feat.dtype)

    def _get_node_emb_block(self, graph_id: int) -> np.ndarray:
        start, end = self.node_ptr[graph_id], self.node_ptr[graph_id + 1]
        return np.asarray(self.node_embd[start:end], dtype=self.node_embd.dtype)

    def _get_edge_blocks(
        self,
        graph_id: int,
        suffix: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_ptr = self.edge_ptrs[suffix]
        start, end = edge_ptr[graph_id], edge_ptr[graph_id + 1]
        edge_idx_key = f"edge_index{suffix}"
        edge_feat_key = f"edge_feat{suffix}"
        edge_index = np.asarray(self.h5_file[edge_idx_key][:, start:end], dtype=np.int32)
        edge_feat = np.asarray(self.h5_file[edge_feat_key][start:end], dtype=self.edge_feat_dtypes[suffix])
        return edge_index, edge_feat

    def batch_collapse(
        self,
        graph_ids: Sequence[int],
        *,
        pad_to_multiple: int = PAD_TO_MULTIPLE,
    ) -> Dict[str, np.ndarray]:
        """
        Collapse a molecule list into one batched graph block.

        Padding rules:
          - prepend one dynamic null graph whose size is chosen so that total nodes/edges
            become multiples of `pad_to_multiple`.
          - no trailing padded rows are appended after valid graphs.
        """
        ids = np.asarray(graph_ids, dtype=np.int64).ravel()
        if np.any(ids < 0) or np.any(ids >= self.num_graphs):
            raise ValueError("All graph IDs must be in [0, num_graphs).")

        n_graphs = ids.size
        total_graphs = n_graphs + 1  # +1 for null graph at the front

        node_counts: list[int] = []
        node_feat_blocks: list[np.ndarray] = []
        node_embd_blocks: list[np.ndarray] = []
        edge_index_blocks: Dict[str, list[np.ndarray]] = {
            suffix: [] for suffix in self.edge_kinds
        }
        edge_feat_blocks: Dict[str, list[np.ndarray]] = {
            suffix: [] for suffix in self.edge_kinds
        }
        edge_counts: Dict[str, list[int]] = {suffix: [] for suffix in self.edge_kinds}

        node_offset = 0
        for gid in ids:
            start, end = self.node_ptr[gid], self.node_ptr[gid + 1]
            n_nodes = int(end - start)
            node_counts.append(n_nodes)
            node_feat_blocks.append(np.asarray(self.h5_file["node_feat"][start:end], dtype=self.node_feat.dtype))
            node_embd_blocks.append(np.asarray(self.h5_file["node_embd"][start:end], dtype=self.node_embd.dtype))

            for suffix in self.edge_kinds:
                edge_counts_list = edge_counts[suffix]
                edge_idx_raw, edge_feat_raw = self._get_edge_blocks(gid, suffix)
                edge_counts_list.append(int(edge_idx_raw.shape[1]))
                if edge_idx_raw.size:
                    edge_idx_raw = edge_idx_raw + node_offset
                edge_index_blocks[suffix].append(edge_idx_raw)
                edge_feat_blocks[suffix].append(edge_feat_raw)

            node_offset += n_nodes

        valid_node_count = int(np.sum(node_counts))
        valid_node_counts = np.asarray(node_counts, dtype=np.int32)
        node_target = _multiple_of(valid_node_count + 1, pad_to_multiple)
        null_node_count = node_target - valid_node_count

        node_ptr = np.zeros(total_graphs + 1, dtype=np.int32)
        node_ptr[0] = 0
        node_ptr[1] = null_node_count
        if n_graphs > 0:
            np.cumsum(valid_node_counts, out=node_ptr[2:])
            node_ptr[2:] += null_node_count

        node_feat_padded = np.zeros((node_target, int(self.node_feat.shape[1])), dtype=np.int32)
        node_embd_padded = np.zeros((node_target, int(self.node_embd.shape[1])), dtype=self.node_embd.dtype)
        if valid_node_count > 0:
            node_feat = np.concatenate(node_feat_blocks, axis=0)
            node_feat = _apply_offsets(node_feat, self._node_feat_offsets)
            node_feat_padded[null_node_count : null_node_count + valid_node_count] = node_feat

            node_embd = np.concatenate(node_embd_blocks, axis=0)
            node_embd_padded[null_node_count : null_node_count + valid_node_count] = node_embd

        node_batch = np.zeros(node_target, dtype=np.int32)
        for batch_gid in range(total_graphs):
            s, e = node_ptr[batch_gid], node_ptr[batch_gid + 1]
            node_batch[s:e] = batch_gid

        out: Dict[str, np.ndarray] = {
            "node_feat": node_feat_padded,
            "node_embd": node_embd_padded,
            "node_ptr": node_ptr,
            "node_batch": node_batch,
            "labels": np.asarray(self.labels[ids], dtype=np.float32),
            "molecule_ids": ids,
            "batch_n_graphs": np.int32(n_graphs),
        }

        for suffix in self.edge_kinds:
            valid_edge_counts = np.asarray(edge_counts[suffix], dtype=np.int32)
            valid_edge_count = int(np.sum(valid_edge_counts))
            edge_target = _multiple_of(valid_edge_count + 1, pad_to_multiple)
            null_edge_count = edge_target - valid_edge_count

            edge_ptr = np.zeros(total_graphs + 1, dtype=np.int32)
            edge_ptr[0] = 0
            edge_ptr[1] = null_edge_count
            if n_graphs > 0:
                np.cumsum(valid_edge_counts, out=edge_ptr[2:])
                edge_ptr[2:] += null_edge_count

            edge_index_padded = np.zeros((2, edge_target), dtype=np.int32)
            edge_feat_padded = np.zeros((edge_target, self.edge_feat_dims[suffix]), dtype=np.int32)
            edge_batch = np.zeros(edge_target, dtype=np.int32)

            if valid_edge_count > 0:
                edge_index = np.concatenate(edge_index_blocks[suffix], axis=1)
                edge_index = edge_index + null_node_count
                edge_index_padded[:, null_edge_count : null_edge_count + valid_edge_count] = edge_index

                edge_feat = np.concatenate(edge_feat_blocks[suffix], axis=0)
                edge_feat = _apply_offsets(edge_feat, self._edge_feat_offsets[suffix])
                edge_feat_padded[null_edge_count : null_edge_count + valid_edge_count] = edge_feat

            for batch_gid in range(total_graphs):
                s, e = edge_ptr[batch_gid], edge_ptr[batch_gid + 1]
                edge_batch[s:e] = batch_gid

            edge_suffix_name = f"edge{suffix}"
            out[f"{edge_suffix_name}_index"] = edge_index_padded
            out[f"{edge_suffix_name}_feat"] = edge_feat_padded
            out[f"{edge_suffix_name}_ptr"] = edge_ptr
            out[f"{edge_suffix_name}_batch"] = edge_batch

        return out


class PCQMDataloader:
    """Simple batch iterator around `PCQMDataset.batch_collapse`."""

    def __init__(
        self,
        dataset: PCQMDataset,
        *,
        indices: Sequence[int] | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        pad_to_multiple: int = PAD_TO_MULTIPLE,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_to_multiple = pad_to_multiple
        self.seed = seed

        if indices is None:
            self.indices = np.arange(dataset.get_graph_count(), dtype=np.int64)
        else:
            candidate = np.asarray(indices, dtype=np.int64)
            if np.any(candidate < 0) or np.any(candidate >= dataset.get_graph_count()):
                raise ValueError("All dataset indices must be in [0, num_graphs).")
            self.indices = candidate

    def __iter__(self) -> Iterable[Dict[str, np.ndarray]]:
        n = self.indices.size
        if n == 0:
            return iter(())

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            order = self.indices.copy()
            rng.shuffle(order)
        else:
            order = self.indices

        def _iter() -> Iterable[Dict[str, np.ndarray]]:
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_ids = order[start:end]
                if batch_ids.size == 0:
                    continue
                if self.drop_last and batch_ids.size < self.batch_size:
                    break
                yield self.dataset.batch_collapse(
                    batch_ids,
                    pad_to_multiple=self.pad_to_multiple,
                )

        return _iter()

    def __len__(self) -> int:
        if self.drop_last:
            return self.indices.size // self.batch_size
        return (self.indices.size + self.batch_size - 1) // self.batch_size

    def get_split(self, split_name: str) -> "PCQMDataloader":
        return PCQMDataloader(
            self.dataset,
            indices=_load_split_indices(self.dataset.dataset_root / "split_dict.h5", split_name, self.dataset.get_graph_count()),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pad_to_multiple=self.pad_to_multiple,
            seed=self.seed,
        )


def batch_collapse(
    dataset: PCQMDataset,
    graph_ids: Sequence[int],
    *,
    pad_to_multiple: int = PAD_TO_MULTIPLE,
) -> Dict[str, np.ndarray]:
    return dataset.batch_collapse(graph_ids, pad_to_multiple=pad_to_multiple)


__all__ = [
    "batch_collapse",
    "PCQMDataset",
    "PCQMDataloader",
    "PAD_TO_MULTIPLE",
    "NODE_FEAT_VOCAB_SIZES",
    "NODE_FEAT_TOTAL_VOCAB",
    "EDGE_FEAT_VOCAB_SIZES",
    "EDGE_FEAT_TOTAL_VOCAB",
]

"""Tests for PCQM dataset batching and dataloader behavior."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import h5py
import numpy as np
import unittest

import importlib.util

# Resolve module import from local `src/dataset.py` without loading the `dataset`
# package (which has unrelated import-time issues in this environment).
_DATASET_SRC = Path(__file__).resolve().parents[1] / "src" / "dataset.py"
_SPEC = importlib.util.spec_from_file_location("pcqm_dataset", _DATASET_SRC)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load src/dataset.py module.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]

PCQMDataset = _MODULE.PCQMDataset
PCQMDataloader = _MODULE.PCQMDataloader
batch_collapse = _MODULE.batch_collapse


def _build_toy_dataset(root: Path) -> Dict[str, np.ndarray]:
    """Write a small synthetic PCQM-style HDF5 dataset and return raw source blocks."""
    root.mkdir(parents=True, exist_ok=True)
    processed = root / "processed"
    processed.mkdir(parents=True)

    node_ptr = np.array([0, 2, 3, 6], dtype=np.int32)
    labels = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    node_feat = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        dtype=np.uint8,
    )
    node_embd = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
            [1.0, 1.1],
        ],
        dtype=np.float32,
    )

    edge_ptr = np.array([0, 1, 2, 3], dtype=np.int32)
    edge_index = np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int32).T
    edge_index = edge_index.T
    edge_feat = np.array(
        [
            [0, 1, 2, 3, 4, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=np.uint8,
    )

    edge_ptr_2hop = np.array([0, 1, 1, 2], dtype=np.int32)
    edge_index_2hop = np.array([[0, 1], [1, 0]], dtype=np.int32)
    edge_feat_2hop = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    with h5py.File(processed / "data_processed.h5", "w") as f:
        f.create_dataset("node_ptr", data=node_ptr)
        f.create_dataset("node_feat", data=node_feat)
        f.create_dataset("node_embd", data=node_embd)
        f.create_dataset("labels", data=labels)

        f.create_dataset("edge_ptr", data=edge_ptr)
        f.create_dataset("edge_index", data=edge_index)
        f.create_dataset("edge_feat", data=edge_feat)

        f.create_dataset("edge_ptr_2hop", data=edge_ptr_2hop)
        f.create_dataset("edge_index_2hop", data=edge_index_2hop)
        f.create_dataset("edge_feat_2hop", data=edge_feat_2hop)

    with h5py.File(root / "split_dict.h5", "w") as f:
        f.create_dataset("train", data=np.array([0, 2], dtype=np.int64))
        f.create_dataset("valid", data=np.array([1], dtype=np.int64))

    return {
        "node_ptr": node_ptr,
        "labels": labels,
        "node_feat": node_feat,
        "node_embd": node_embd,
        "edge_ptr": edge_ptr,
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "edge_ptr_2hop": edge_ptr_2hop,
        "edge_index_2hop": edge_index_2hop,
        "edge_feat_2hop": edge_feat_2hop,
    }


def _build_toy_dataset_with_4hop(root: Path) -> Dict[str, np.ndarray]:
    blocks = _build_toy_dataset(root)
    processed = root / "processed" / "data_processed.h5"
    edge_ptr_4hop = np.array([0, 1, 2, 3], dtype=np.int32)
    edge_index_4hop = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    edge_feat_4hop = np.array(
        [[1, 0, 1, 2], [3, 1, 0, 1], [2, 2, 1, 0]], dtype=np.uint8
    )

    with h5py.File(processed, "a") as f:
        f.create_dataset("edge_ptr_4hop", data=edge_ptr_4hop)
        f.create_dataset("edge_index_4hop", data=edge_index_4hop)
        f.create_dataset("edge_feat_4hop", data=edge_feat_4hop)

    blocks["edge_ptr_4hop"] = edge_ptr_4hop
    blocks["edge_index_4hop"] = edge_index_4hop
    blocks["edge_feat_4hop"] = edge_feat_4hop
    return blocks


def _apply_offsets(features: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.int32) + offsets[np.newaxis, :]


class PCQMDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dataset_root = Path(self._tmp.name) / "pcqm4m-v2"
        blocks = _build_toy_dataset(self.dataset_root)
        self.blocks = blocks
        self.dataset = PCQMDataset(dataset_root=self.dataset_root, split=None, split_file=self.dataset_root / "split_dict.h5")

    def tearDown(self) -> None:
        self.dataset.close()
        self._tmp.cleanup()

    def test_init_and_split_indices(self) -> None:
        self.assertEqual(self.dataset.get_graph_count(), 3)
        self.assertEqual(len(self.dataset.split_indices), 3)
        self.assertEqual(self.dataset.get_split_indices().shape, (3,))
        self.assertEqual(self.dataset.edge_kinds, ["", "_2hop"])

        with TemporaryDirectory() as bad_dir:
            bad_root = Path(bad_dir)
            with self.assertRaises(FileNotFoundError):
                PCQMDataset(dataset_root=bad_root / "missing", split=None, split_file=bad_root / "split_dict.h5")

    def test_processed_h5_alternate_basename(self) -> None:
        proc = self.dataset_root / "processed"
        alt = proc / "data_processed_copy.h5"
        shutil.copyfile(proc / "data_processed.h5", alt)
        ds2 = PCQMDataset(
            dataset_root=self.dataset_root,
            split=None,
            split_file=self.dataset_root / "split_dict.h5",
            processed_h5="data_processed_copy.h5",
        )
        try:
            self.assertEqual(ds2.get_graph_count(), self.dataset.get_graph_count())
        finally:
            ds2.close()
            alt.unlink(missing_ok=True)

    def test_batch_collapse_dynamic_null_and_offsets(self) -> None:
        batch = self.dataset.batch_collapse([0, 2], pad_to_multiple=4)

        valid_node_counts = np.array([2, 3], dtype=np.int32)
        valid_node_count = int(valid_node_counts.sum())
        node_target = ((valid_node_count + 1 + 4 - 1) // 4) * 4
        null_node_count = node_target - valid_node_count

        expected_node_ptr = np.array([0, null_node_count, *(null_node_count + np.cumsum(valid_node_counts))], dtype=np.int32)

        self.assertEqual(batch["node_feat"].shape, (node_target, self.blocks["node_feat"].shape[1]))
        self.assertEqual(batch["node_embd"].shape, (node_target, self.blocks["node_embd"].shape[1]))
        self.assertTrue(np.array_equal(batch["node_ptr"], expected_node_ptr))
        self.assertTrue(np.all(batch["node_ptr"][1:] >= batch["node_ptr"][:-1]))
        self.assertTrue(np.all((batch["node_batch"] >= 0)))
        self.assertTrue(np.all(batch["node_batch"] < 3))
        self.assertEqual(int(batch["batch_n_graphs"]), 2)
        self.assertTrue(np.all(batch["labels"] == np.array([10.0, 30.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(batch["molecule_ids"], np.array([0, 2], dtype=np.int64)))

        expected_node_feat = np.zeros_like(batch["node_feat"])
        valid_node = np.vstack([self.blocks["node_feat"][:2], self.blocks["node_feat"][3:]])
        expected_node_feat[null_node_count : null_node_count + valid_node_count] = _apply_offsets(
            valid_node, self.dataset._node_feat_offsets
        )
        np.testing.assert_array_equal(batch["node_feat"], expected_node_feat)

        expected_node_embd = np.zeros_like(batch["node_embd"])
        valid_embd = np.vstack([self.blocks["node_embd"][:2], self.blocks["node_embd"][3:]])
        expected_node_embd[null_node_count : null_node_count + valid_node_count] = valid_embd
        np.testing.assert_allclose(batch["node_embd"], expected_node_embd)

        for suffix in self.dataset.edge_kinds:
            edge_key = f"edge{suffix}"
            idx = batch[f"{edge_key}_index"]
            feat = batch[f"{edge_key}_feat"]
            ptr = batch[f"{edge_key}_ptr"]
            batch_ids = batch[f"{edge_key}_batch"]

            valid_edge_counts = np.array([1, 1], dtype=np.int32)
            valid_edge_count = int(valid_edge_counts.sum())
            edge_target = ((valid_edge_count + 1 + 4 - 1) // 4) * 4
            null_edge_count = edge_target - valid_edge_count
            expected_ptr = np.zeros(4, dtype=np.int32)
            expected_ptr[0] = 0
            expected_ptr[1] = null_edge_count
            np.cumsum(valid_edge_counts, out=expected_ptr[2:])
            expected_ptr[2:] += null_edge_count
            self.assertEqual(idx.shape, (2, edge_target))
            self.assertEqual(feat.shape[0], edge_target)
            self.assertTrue(np.all(ptr >= 0))
            self.assertTrue(np.array_equal(ptr, expected_ptr))
            self.assertTrue(np.all(batch_ids >= 0))
            self.assertTrue(np.all(batch_ids < 3))
            self.assertTrue(np.all(idx[:, :null_edge_count] == 0))
            self.assertTrue(np.all(feat[:null_edge_count] == 0))

            if suffix == "":
                valid_edge_feat = np.vstack([self.blocks["edge_feat"][:1], self.blocks["edge_feat"][2:]])
                valid_edge_idx = np.concatenate(
                    [
                        self.blocks["edge_index"][:, :1],
                        self.blocks["edge_index"][:, 2:] + 2,
                    ],
                    axis=1,
                ) + null_node_count
            else:
                valid_edge_feat = np.vstack([self.blocks["edge_feat_2hop"][:1], self.blocks["edge_feat_2hop"][1:]])
                valid_edge_idx = np.concatenate(
                    [
                        self.blocks["edge_index_2hop"][:, :1],
                        self.blocks["edge_index_2hop"][:, 1:] + 2,
                    ],
                    axis=1,
                ) + null_node_count

            expected_edge_feat = np.zeros_like(feat)
            expected_edge_idx = np.zeros_like(idx)
            expected_edge_feat[null_edge_count : null_edge_count + valid_edge_count] = _apply_offsets(
                valid_edge_feat, self.dataset._edge_feat_offsets[suffix]
            )
            expected_edge_idx[:, null_edge_count : null_edge_count + valid_edge_count] = valid_edge_idx
            np.testing.assert_array_equal(idx, expected_edge_idx)
            np.testing.assert_array_equal(feat, expected_edge_feat)

        top_level_batch = batch_collapse(self.dataset, [0, 2], pad_to_multiple=4)
        for key in top_level_batch:
            np.testing.assert_array_equal(top_level_batch[key], batch[key])

    def test_batch_collapse_empty_graph_list(self) -> None:
        batch = self.dataset.batch_collapse([], pad_to_multiple=4)
        self.assertEqual(batch["node_feat"].shape[0], 4)
        self.assertEqual(batch["edge_index"].shape[1], 4)
        self.assertEqual(batch["edge_2hop_index"].shape[1], 4)
        self.assertTrue(np.all(batch["node_batch"] == 0))
        self.assertTrue(np.all(batch["edge_batch"] == 0))
        self.assertTrue(np.all(batch["edge_2hop_batch"] == 0))
        self.assertTrue(np.array_equal(batch["node_ptr"], np.array([0, 4], dtype=np.int32)))
        self.assertTrue(np.array_equal(batch["edge_ptr"], np.array([0, 4], dtype=np.int32)))
        self.assertTrue(np.array_equal(batch["edge_2hop_ptr"], np.array([0, 4], dtype=np.int32)))
        self.assertEqual(batch["labels"].shape[0], 0)
        self.assertEqual(int(batch["batch_n_graphs"]), 0)

    def test_invalid_batch_ids(self) -> None:
        with self.assertRaises(ValueError):
            self.dataset.batch_collapse([99], pad_to_multiple=4)
        with self.assertRaises(ValueError):
            self.dataset.batch_collapse([0, -1], pad_to_multiple=4)
        with self.assertRaises(ValueError):
            self.dataset.batch_collapse([0], pad_to_multiple=0)

    def test_batch_collapse_includes_4hop_edge_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _build_toy_dataset_with_4hop(dataset_root)
            dataset = PCQMDataset(
                dataset_root=dataset_root,
                split=None,
                split_file=dataset_root / "split_dict.h5",
            )
            self.assertIn("_4hop", dataset.edge_kinds)
            batch = dataset.batch_collapse([0, 2], pad_to_multiple=4)
            self.assertIn("edge_4hop_index", batch)
            self.assertIn("edge_4hop_feat", batch)
            self.assertIn("edge_4hop_ptr", batch)
            self.assertIn("edge_4hop_batch", batch)



class PCQMDataloaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dataset_root = Path(self._tmp.name) / "pcqm4m-v2"
        _build_toy_dataset(self.dataset_root)
        self.dataset = PCQMDataset(dataset_root=self.dataset_root, split=None, split_file=self.dataset_root / "split_dict.h5")

    def tearDown(self) -> None:
        self.dataset.close()
        self._tmp.cleanup()

    def test_len_and_drop_last(self) -> None:
        loader = PCQMDataloader(self.dataset, batch_size=2)
        self.assertEqual(len(loader), 2)
        loader_drop_last = PCQMDataloader(self.dataset, batch_size=2, drop_last=True)
        self.assertEqual(len(loader_drop_last), 1)

    def test_default_pad_to_multiple_is_four_times_batch_size(self) -> None:
        for bs in (1, 2, 3, 5):
            loader = PCQMDataloader(self.dataset, batch_size=bs)
            self.assertEqual(loader.pad_to_multiple, bs * 4)

    def test_iter_and_batching(self) -> None:
        loader = PCQMDataloader(
            self.dataset,
            batch_size=2,
            indices=np.array([2, 0, 1], dtype=np.int64),
            shuffle=False,
            drop_last=False,
            pad_to_multiple=4,
            seed=42,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        # shuffle=False shuffles once at init; seed=42 permutes [2,0,1] -> [1,0,2]
        self.assertTrue(np.array_equal(np.concatenate([batch["molecule_ids"] for batch in batches]), np.array([1, 0, 2])))

        batch_sizes = [batch["batch_n_graphs"] for batch in batches]
        self.assertEqual(batch_sizes, [np.int32(2), np.int32(1)])
        self.assertEqual(batches[0]["node_ptr"].shape[0], 4)
        self.assertEqual(batches[1]["node_ptr"].shape[0], 3)

    def test_shuffle_determinism(self) -> None:
        loader_a = PCQMDataloader(self.dataset, batch_size=2, shuffle=True, seed=123, pad_to_multiple=4)
        loader_b = PCQMDataloader(self.dataset, batch_size=2, shuffle=True, seed=123, pad_to_multiple=4)

        order_a = np.concatenate([batch["molecule_ids"] for batch in loader_a])
        order_b = np.concatenate([batch["molecule_ids"] for batch in loader_b])
        expected = np.arange(self.dataset.get_graph_count(), dtype=np.int64)
        rng = np.random.default_rng(123)
        rng.shuffle(expected)
        self.assertTrue(np.array_equal(order_a, order_b))
        self.assertTrue(np.array_equal(order_a, expected))

    def test_shuffle_false_same_order_on_repeated_passes(self) -> None:
        loader = PCQMDataloader(self.dataset, batch_size=1, shuffle=False, seed=7, pad_to_multiple=4)
        order_1 = np.concatenate([b["molecule_ids"] for b in loader])
        order_2 = np.concatenate([b["molecule_ids"] for b in loader])
        self.assertTrue(np.array_equal(order_1, order_2))

    def test_shuffle_true_new_order_each_pass(self) -> None:
        loader = PCQMDataloader(self.dataset, batch_size=1, shuffle=True, seed=0, pad_to_multiple=4)
        order_1 = np.concatenate([b["molecule_ids"] for b in loader])
        order_2 = np.concatenate([b["molecule_ids"] for b in loader])
        self.assertFalse(np.array_equal(order_1, order_2))

    def test_get_split(self) -> None:
        loader = PCQMDataloader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            pad_to_multiple=4,
            seed=3,
        )
        split_loader = loader.get_split("train")
        self.assertTrue(np.array_equal(split_loader.indices, np.array([0, 2], dtype=np.int64)))
        self.assertEqual(len(split_loader), 1)
        split_batch = next(iter(split_loader))
        # One-time shuffle with seed=3 permutes [0, 2] -> [2, 0]
        self.assertTrue(np.array_equal(split_batch["molecule_ids"], np.array([2, 0], dtype=np.int64)))

        with self.assertRaises(ValueError):
            loader.get_split("missing")

    def test_invalid_dataloader_args(self) -> None:
        with self.assertRaises(ValueError):
            PCQMDataloader(self.dataset, batch_size=0)
        with self.assertRaises(ValueError):
            PCQMDataloader(self.dataset, indices=[-1, 1])

    def test_dataloader_batch_uses_new_batch_contract_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _build_toy_dataset_with_4hop(dataset_root)
            dataset = PCQMDataset(
                dataset_root=dataset_root,
                split=None,
                split_file=dataset_root / "split_dict.h5",
            )
            loader = PCQMDataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                drop_last=False,
                pad_to_multiple=4,
            )
            batch = next(iter(loader))

            self.assertIn("node_batch", batch)
            self.assertIn("node_embd", batch)
            self.assertIn("batch_n_graphs", batch)
            self.assertNotIn("node_mask", batch)
            self.assertNotIn("batch_size", batch)
            self.assertNotIn("node_graph_id", batch)

            for suffix in dataset.edge_kinds:
                self.assertIn(f"edge{suffix}_batch", batch)
                self.assertIn(f"edge{suffix}_index", batch)
                self.assertIn(f"edge{suffix}_feat", batch)
                self.assertIn(f"edge{suffix}_ptr", batch)
                self.assertNotIn(f"edge{suffix}_mask", batch)
                self.assertTrue(np.all(batch[f"edge{suffix}_batch"] >= 0))


if __name__ == "__main__":
    unittest.main()


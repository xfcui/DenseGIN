"""Tests for model compatibility with dataset batch structure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

_DATASET_SRC = REPO_ROOT / "src" / "dataset.py"
_DATASET_SPEC = importlib.util.spec_from_file_location("pcqm_dataset", _DATASET_SRC)
if _DATASET_SPEC is None or _DATASET_SPEC.loader is None:
    raise RuntimeError("Unable to load src/dataset.py module.")
_DATASET_MODULE = importlib.util.module_from_spec(_DATASET_SPEC)
_DATASET_SPEC.loader.exec_module(_DATASET_MODULE)  # type: ignore[arg-type]

sys.modules["dataset"] = _DATASET_MODULE

_MODEL_SRC = REPO_ROOT / "src" / "model.py"
_MODEL_SPEC = importlib.util.spec_from_file_location("pcqm_model", _MODEL_SRC)
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise RuntimeError("Unable to load src/model.py module.")
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)  # type: ignore[arg-type]

DenseGIN = _MODEL_MODULE.DenseGIN
EmbedLayer = _MODEL_MODULE.EmbedLayer
get_model = _MODEL_MODULE.get_model

PCQMDataset = _DATASET_MODULE.PCQMDataset
NODE_FEAT_VOCAB_SIZES = list(_DATASET_MODULE.NODE_FEAT_VOCAB_SIZES)
EDGE_FEAT_VOCAB_SIZES = _DATASET_MODULE.EDGE_FEAT_VOCAB_SIZES
EDGE_FEAT_TOTAL_VOCAB = _DATASET_MODULE.EDGE_FEAT_TOTAL_VOCAB
EDGE_SUFFIXES = list(EDGE_FEAT_VOCAB_SIZES.keys())
MODEL_EDGE_SUFFIXES = _MODEL_MODULE.EDGE_SUFFIXES
MODEL_EDGE_DIMS_PER_HOP = _MODEL_MODULE.EDGE_DIMS_PER_HOP


def _make_toy_dataset(path: Path) -> None:
    processed = path / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    node_ptr = np.array([0, 2, 3], dtype=np.int32)
    labels = np.array([1.25, 2.5], dtype=np.float32)
    node_feat = np.vstack(
        [
            np.asarray([i for i in range(len(NODE_FEAT_VOCAB_SIZES))], dtype=np.int32),
            np.asarray([(i + 1) % n for i, n in enumerate(NODE_FEAT_VOCAB_SIZES)], dtype=np.int32),
            np.asarray([(i + 2) % n for i, n in enumerate(NODE_FEAT_VOCAB_SIZES)], dtype=np.int32),
        ]
    ).astype(np.uint8)
    node_embd = np.arange(node_ptr[-1] * 17, dtype=np.float32).reshape(node_ptr[-1], 17) / 10.0

    edge_ptr = np.array([0, 1, 2], dtype=np.int32)
    edge_index = np.array([[0, 0], [0, 0]], dtype=np.int32)

    with h5py.File(processed / "data_processed.h5", "w") as f:
        f.create_dataset("node_ptr", data=node_ptr)
        f.create_dataset("node_feat", data=node_feat)
        f.create_dataset("node_embd", data=node_embd)
        f.create_dataset("labels", data=labels)
        f.create_dataset("edge_ptr", data=edge_ptr)
        f.create_dataset("edge_index", data=edge_index)
        f.create_dataset("edge_feat", data=np.array([[0, 1, 2, 3, 4, 0], [1, 0, 1, 0, 1, 2]], dtype=np.uint8))

        f.create_dataset("edge_ptr_2hop", data=edge_ptr)
        f.create_dataset("edge_index_2hop", data=edge_index)
        f.create_dataset("edge_feat_2hop", data=np.array([[0, 1], [1, 2]], dtype=np.uint8))

        f.create_dataset("edge_ptr_3hop", data=edge_ptr)
        f.create_dataset("edge_index_3hop", data=edge_index)
        f.create_dataset("edge_feat_3hop", data=np.array([[0, 1, 2], [1, 0, 1]], dtype=np.uint8))

        f.create_dataset("edge_ptr_4hop", data=edge_ptr)
        f.create_dataset("edge_index_4hop", data=edge_index)
        f.create_dataset("edge_feat_4hop", data=np.array([[0, 1, 2, 3], [1, 0, 1, 0]], dtype=np.uint8))

    with h5py.File(path / "split_dict.h5", "w") as f:
        f.create_dataset("train", data=np.array([0, 1], dtype=np.int64))


def _make_minimal_batch() -> dict[str, np.ndarray]:
    node_feat = np.zeros((3, 10), dtype=np.int32)
    node_embd = np.zeros((3, 17), dtype=np.float32)
    batch = {
        "node_feat": node_feat,
        "node_embd": node_embd,
        "node_batch": np.array([0, 1, 1], dtype=np.int32),
        "batch_n_graphs": np.int32(1),
    }

    for suffix in EDGE_SUFFIXES:
        dim = len(EDGE_FEAT_VOCAB_SIZES[suffix])
        batch[f"edge{suffix}_index"] = np.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int32)
        feat = np.zeros((4, dim), dtype=np.int32)
        feat[0] = np.arange(1, dim + 1) % max(1, len(EDGE_FEAT_VOCAB_SIZES[suffix]))
        batch[f"edge{suffix}_feat"] = feat
        batch[f"edge{suffix}_batch"] = np.array([1, 0, 0, 0], dtype=np.int32)

    return batch


def _dense_gin(
    *,
    depth: int,
    width: int,
    num_head: int,
    key: jax.Array,
) -> DenseGIN:
    if width % num_head != 0:
        raise ValueError("width must be divisible by num_head")
    return DenseGIN(
        depth=depth,
        width=width,
        num_head=num_head,
        dim_head=width // num_head,
        key=key,
    )


class ModelCompatibilityTest(unittest.TestCase):
    def test_embed_layer_uses_pre_offset_tokens(self) -> None:
        embeddings = np.zeros((32, 6), dtype=np.float32)
        for row in range(1, 32):
            embeddings[row] = row
        layer = EmbedLayer(total_vocab=32, num_features=3, width=6, key=jax.random.PRNGKey(0))
        layer = eqx.tree_at(lambda x: x.embeddings, layer, jnp.array(embeddings))

        x = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int32)
        out = layer(jnp.array(x))
        zero_row_out = np.asarray(out[0])
        self.assertTrue(np.allclose(zero_row_out, 0.0))

    def test_get_edge_matches_dataset_keying_and_masks(self) -> None:
        batch = _make_minimal_batch()
        model = _dense_gin(depth=1, width=16, num_head=4, key=jax.random.PRNGKey(0))
        edges = model._get_edge(batch)

        self.assertEqual(len(edges), len(EDGE_SUFFIXES))
        for (edge_index, edge_attr, _deg), suffix in zip(edges, EDGE_SUFFIXES):
            self.assertEqual(edge_index.shape[1], 4)
            self.assertEqual(edge_attr.shape[0], 4)
            edge_mask = batch[f"edge{suffix}_batch"] > 0
            self.assertTrue(np.array_equal(np.array(edge_mask), np.array(batch[f"edge{suffix}_batch"] > 0)))

    def test_model_forward_output_uses_new_batch_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(dataset_root=dataset_root, split="train", split_file=dataset_root / "split_dict.h5")
            batch = dataset.batch_collapse([0, 1], pad_to_multiple=4)
            model = _dense_gin(depth=1, width=32, num_head=2, key=jax.random.PRNGKey(0))

            out = model(batch, training=False, key=None)
            self.assertEqual(out.shape, (2, 1))
            self.assertTrue(np.all(np.isfinite(np.asarray(out))))

    def test_model_uses_all_dataset_suffixes_and_masked_edges(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(
                dataset_root=dataset_root,
                split="train",
                split_file=dataset_root / "split_dict.h5",
            )
            batch = dataset.batch_collapse([0, 1], pad_to_multiple=4)

            self.assertEqual(MODEL_EDGE_SUFFIXES, EDGE_SUFFIXES)
            for suffix in EDGE_SUFFIXES:
                self.assertIn(f"edge{suffix}_index", batch)
                self.assertIn(f"edge{suffix}_feat", batch)
                self.assertIn(f"edge{suffix}_batch", batch)
                self.assertIn(f"edge{suffix}_ptr", batch)

                self.assertEqual(batch[f"edge{suffix}_index"].shape[0], 2)
                self.assertEqual(batch[f"edge{suffix}_index"].shape[1] % 4, 0)
                self.assertEqual(batch[f"edge{suffix}_feat"].shape[1], MODEL_EDGE_DIMS_PER_HOP[EDGE_SUFFIXES.index(suffix)][1])
                self.assertEqual(batch[f"edge{suffix}_feat"].shape[0], batch[f"edge{suffix}_index"].shape[1])
                self.assertTrue(np.array_equal(batch[f"edge{suffix}_batch"][0], 0))

            # The null graph has index 0 and must be removed from model output.
            model = _dense_gin(depth=1, width=32, num_head=2, key=jax.random.PRNGKey(1))
            out = model(batch, training=False, key=None)
            self.assertEqual(out.shape[0], int(batch["batch_n_graphs"]))

    def test_model_forward_training_and_eval_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(dataset_root=dataset_root, split="train", split_file=dataset_root / "split_dict.h5")
            batch = dataset.batch_collapse([0, 1], pad_to_multiple=4)

            model = _dense_gin(depth=2, width=32, num_head=2, key=jax.random.PRNGKey(7))
            out_eval = model(batch, training=False, key=None)
            out_train = model(batch, training=True, key=jax.random.PRNGKey(0))

            self.assertEqual(out_eval.shape, (int(batch["batch_n_graphs"]), 1))
            self.assertEqual(out_train.shape, out_eval.shape)
            self.assertTrue(np.all(np.isfinite(np.asarray(out_eval))))
            self.assertTrue(np.all(np.isfinite(np.asarray(out_train))))

    def test_model_forward_accepts_node_mask_derived_graph_id(self) -> None:
        # This intentionally exercises the updated node_mask path (node_batch > 0).
        batch = _make_minimal_batch()
        model = _dense_gin(depth=1, width=16, num_head=2, key=jax.random.PRNGKey(42))
        batch["node_batch"] = np.array([0, 1, 1, 1], dtype=np.int32)
        batch["batch_n_graphs"] = np.int32(2)
        batch["node_feat"] = np.pad(
            batch["node_feat"],
            pad_width=((0, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        batch["node_embd"] = np.pad(
            batch["node_embd"],
            pad_width=((0, 1), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        for suffix in EDGE_SUFFIXES:
            batch[f"edge{suffix}_index"] = np.pad(
                batch[f"edge{suffix}_index"],
                ((0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        out = model(batch, training=False, key=None)
        self.assertEqual(out.shape[0], 2)

    def test_model_and_batch_suffix_config_match(self) -> None:
        self.assertEqual(MODEL_EDGE_SUFFIXES, EDGE_SUFFIXES)
        self.assertEqual(len(MODEL_EDGE_DIMS_PER_HOP), len(EDGE_SUFFIXES))
        for i, suffix in enumerate(EDGE_SUFFIXES):
            total_vocab, n_feat = MODEL_EDGE_DIMS_PER_HOP[i]
            self.assertEqual(total_vocab, EDGE_FEAT_TOTAL_VOCAB[suffix])
            self.assertEqual(n_feat, len(EDGE_FEAT_VOCAB_SIZES[suffix]))

    def test_model_forward_uses_null_graph_exclusion(self) -> None:
        batch = _make_minimal_batch()
        batch["batch_n_graphs"] = np.int32(1)
        batch["node_batch"] = np.array([0, 1, 0, 0], dtype=np.int32)
        batch["node_embd"] = np.zeros((4, 17), dtype=np.float32)
        batch["node_feat"] = np.zeros((4, 10), dtype=np.int32)
        for suffix in EDGE_SUFFIXES:
            idx = np.array([[0, 1, 0, 1], [2, 0, 0, 0]], dtype=np.int32)
            batch[f"edge{suffix}_index"] = idx

            model = _dense_gin(depth=1, width=16, num_head=2, key=jax.random.PRNGKey(2025))
        out = model(batch, training=False, key=None)
        self.assertEqual(out.shape, (1, 1))

        deg = model._get_edge(batch)[0][2]
        self.assertTrue(jnp.all(deg >= 1))
        self.assertEqual(int(deg[0]), 3)  # null-node receives all incident null edges

    def test_model_forward_with_empty_real_graph_batch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(dataset_root=dataset_root, split="train", split_file=dataset_root / "split_dict.h5")
            batch = dataset.batch_collapse([], pad_to_multiple=4)

            self.assertEqual(int(batch["batch_n_graphs"]), 0)
            model = _dense_gin(depth=1, width=16, num_head=2, key=jax.random.PRNGKey(11))
            out = model(batch, training=False, key=None)
            self.assertEqual(out.shape, (0, 1))

    def test_get_edge_uses_derived_masks_and_keeps_null_nodes_min_degree(self) -> None:
        batch = _make_minimal_batch()
        for suffix in EDGE_SUFFIXES:
            batch[f"edge{suffix}_batch"] = np.array([0, 0, 1, 1], dtype=np.int32)
            batch[f"edge{suffix}_index"] = np.array(
                [
                    [0, 0, 1, 2],
                    [0, 1, 1, 2],
                ],
                dtype=np.int32,
            )

        edges = _dense_gin(depth=1, width=8, num_head=2, key=jax.random.PRNGKey(3))._get_edge(batch)
        for suffix, (_edge_index, _edge_attr, deg) in zip(EDGE_SUFFIXES, edges):
            edge_mask = batch[f"edge{suffix}_batch"] > 0
            self.assertTrue(np.array_equal(edge_mask, np.array([False, False, True, True])))
            self.assertEqual(deg.shape[0], 3)
            self.assertTrue(int(deg[0]) >= 1)
            self.assertTrue(int(deg[1]) >= 1)

    def test_embed_layer_indexes_pre_offset_token_ids_directly(self) -> None:
        total_vocab = 32
        width = 8
        layer = EmbedLayer(total_vocab=total_vocab, num_features=4, width=width, key=jax.random.PRNGKey(123))
        custom_embeddings = np.arange(total_vocab * width, dtype=np.float32).reshape(total_vocab, width)
        layer = eqx.tree_at(lambda x: x.embeddings, layer, jnp.array(custom_embeddings))

        x = np.array([[0, 1, 2, 3], [5, 6, 7, 8]], dtype=np.int32)
        out = np.asarray(layer(jnp.array(x)))

        expected_first = custom_embeddings[[0, 1, 2, 3]].sum(axis=0)
        expected_second = custom_embeddings[[5, 6, 7, 8]].sum(axis=0)
        np.testing.assert_allclose(out[0], expected_first)
        np.testing.assert_allclose(out[1], expected_second)

    def test_model_forward_with_all_edges_in_null_graph_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(dataset_root=dataset_root, split="train", split_file=dataset_root / "split_dict.h5")
            batch = dataset.batch_collapse([0, 1], pad_to_multiple=4)

            for suffix in EDGE_SUFFIXES:
                batch[f"edge{suffix}_batch"] = np.zeros_like(batch[f"edge{suffix}_batch"])

            model = _dense_gin(depth=1, width=16, num_head=2, key=jax.random.PRNGKey(2026))
            out = model(batch, training=False, key=None)
            self.assertEqual(out.shape, (2, 1))
            self.assertTrue(np.all(np.isfinite(np.asarray(out))))

    def test_model_forward_accepts_dataset_batch_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "pcqm4m-v2"
            _make_toy_dataset(dataset_root)
            dataset = PCQMDataset(dataset_root=dataset_root, split="train", split_file=dataset_root / "split_dict.h5")
            batch = dataset.batch_collapse([0, 1], pad_to_multiple=4)

            self.assertIn("node_batch", batch)
            self.assertNotIn("node_graph_id", batch)
            self.assertIn("node_embd", batch)
            self.assertNotIn("rwpe", batch)
            self.assertIn("batch_n_graphs", batch)
            self.assertNotIn("batch_size", batch)

            model = _dense_gin(depth=1, width=16, num_head=4, key=jax.random.PRNGKey(2027))
            out = model(batch, training=False, key=None)
            self.assertEqual(out.shape[0], int(batch["batch_n_graphs"]))
            self.assertTrue(np.all(np.isfinite(np.asarray(out))))



    def test_get_model_matches_expected_default_config(self) -> None:
        model = get_model(None)
        self.assertIsInstance(model, DenseGIN)
        self.assertEqual(model.depth, 5)
        self.assertEqual(model.width, 256)
        self.assertEqual(model.num_head, 16)
        self.assertEqual(model.dim_head, 16)

    def test_get_model_uses_stable_seed_when_none(self) -> None:
        model_a = get_model(None)
        model_b = get_model(None)
        params_a = jax.tree_util.tree_leaves(eqx.filter(model_a, eqx.is_array))
        params_b = jax.tree_util.tree_leaves(eqx.filter(model_b, eqx.is_array))
        self.assertEqual(len(params_a), len(params_b))
        for pa, pb in zip(params_a, params_b):
            self.assertEqual(pa.shape, pb.shape)
            np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))

    def test_get_model_output_is_finite_and_matches_explicit_ctor(self) -> None:
        batch = _make_minimal_batch()
        model_default = get_model(None)
        model_explicit = DenseGIN(
            depth=5,
            width=256,
            num_head=16,
            dim_head=16,
            key=jax.random.PRNGKey(0),
        )

        out_default = model_default(batch, training=False, key=None)
        out_explicit = model_explicit(batch, training=False, key=None)

        self.assertEqual(out_default.shape, (int(batch["batch_n_graphs"]), 1))
        self.assertTrue(np.all(np.isfinite(np.asarray(out_default))))
        np.testing.assert_allclose(np.asarray(out_default), np.asarray(out_explicit))

if __name__ == "__main__":
    unittest.main()

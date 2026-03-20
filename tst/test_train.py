from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import patch
import sys

import math
import importlib.util

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
_TRAIN_SRC = REPO_ROOT / "src" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("pcqm_train", _TRAIN_SRC)
if _TRAIN_SPEC is None or _TRAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load src/train.py module.")
TRAIN = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(TRAIN)  # type: ignore[arg-type]

to_jax_batch = TRAIN.to_jax_batch
loss_fn = TRAIN.loss_fn
train_step = TRAIN.train_step
eval_step = TRAIN.eval_step
get_scheduled_hparams = TRAIN.get_scheduled_hparams
_resolve_dataset_root = TRAIN._resolve_dataset_root
get_jax_dataloader = TRAIN.get_jax_dataloader


class _AffineModel(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, weight: float = 1.0, bias: float = 0.0):
        self.weight = jnp.array(weight, dtype=jnp.float32)
        self.bias = jnp.array(bias, dtype=jnp.float32)

    def __call__(self, batch, training: bool = False, key=None):
        labels = batch["labels"]
        return labels.reshape(-1, 1) * self.weight + self.bias


class TrainUtilityTest(TestCase):
    def test_scheduled_lr_is_piecewise(self) -> None:
        base_lr = 1.0
        base_wd = 0.0
        k = 4
        lr, wd = get_scheduled_hparams(0.0, k, base_lr, base_wd)
        self.assertEqual(lr, 0.0)
        lr, wd = get_scheduled_hparams(2.0, k, base_lr, base_wd)
        self.assertAlmostEqual(lr, 0.5)
        lr, wd = get_scheduled_hparams(4.0, k, base_lr, base_wd)
        self.assertAlmostEqual(lr, base_lr)

        gr = (1 + math.sqrt(5)) / 2
        t = 0.5
        period_one = 1
        lr_start = base_lr / gr ** (period_one - 1)
        lr_end = lr_start / gr ** 2
        expected = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))
        lr, wd = get_scheduled_hparams(4.0 + 4.0 * t, k, base_lr, base_wd)
        self.assertAlmostEqual(
            lr,
            expected,
            places=12,
        )

    def test_scheduled_wd_is_piecewise(self) -> None:
        base_lr = 0.0
        base_wd = 0.2
        k = 8
        lr, wd = get_scheduled_hparams(0.0, k, base_lr, base_wd)
        self.assertEqual(wd, 0.0)
        lr, wd = get_scheduled_hparams(4.0, k, base_lr, base_wd)
        self.assertAlmostEqual(wd, base_wd * 0.5)
        lr, wd = get_scheduled_hparams(8.0, k, base_lr, base_wd)
        self.assertAlmostEqual(wd, base_wd)

    def test_resolve_dataset_root_handles_processed_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "pcqm4m-v2"
            processed = root / "processed"
            processed.mkdir(parents=True, exist_ok=True)
            (root / "other").mkdir(exist_ok=True)

            self.assertEqual(_resolve_dataset_root(processed), root)
            self.assertEqual(_resolve_dataset_root(root), root)
            self.assertEqual(_resolve_dataset_root(root / "other"), root)
            self.assertEqual(_resolve_dataset_root(Path(temp_dir) / "not-found"), Path(temp_dir))

    def test_to_jax_batch_converts_all_values(self) -> None:
        batch: dict[str, Any] = {
            "labels": np.array([1.5, 2.5], dtype=np.float32),
            "indices": np.array([10, 20], dtype=np.int64),
            "flag": np.array(True),
            "scalar": 7,
        }
        converted = to_jax_batch(batch)

        for key in ("labels", "indices", "flag", "scalar"):
            self.assertTrue(hasattr(converted[key], "shape"))
            self.assertTrue(hasattr(converted[key], "dtype"))

        np.testing.assert_array_equal(np.asarray(converted["labels"]), batch["labels"])
        np.testing.assert_array_equal(np.asarray(converted["indices"]), batch["indices"])
        self.assertEqual(int(np.asarray(converted["scalar"]),), batch["scalar"])

    def test_loss_fn_applies_mae_threshold(self) -> None:
        model = _AffineModel(weight=1.5, bias=0.0)
        high_residual_batch = {"labels": jnp.array([[1.0]], dtype=jnp.float32)}
        low_residual_batch = {"labels": jnp.array([[0.005]], dtype=jnp.float32)}

        # High residual: |1.0 - 1.5*1.0| = 0.5 > threshold (0.06), so uses MAE directly
        loss = loss_fn(model, high_residual_batch, key=None)
        np.testing.assert_allclose(np.asarray(loss), np.array(0.5, dtype=np.float32))
        
        # Low residual: |0.005 - 1.5*0.005| = 0.0025 < threshold (0.06), so uses quadratic penalty
        # loss = (0.0025^2) / 0.06 ≈ 0.000104
        loss_low = loss_fn(model, low_residual_batch, key=None)
        expected_low = (0.0025 ** 2) / 0.06
        np.testing.assert_allclose(np.asarray(loss_low), np.array(expected_low, dtype=np.float32), rtol=1e-5)


class TrainStepTest(TestCase):
    def test_train_step_changes_weights_and_returns_loss(self) -> None:
        model = _AffineModel(weight=-0.25, bias=0.1)
        optimizer = optax.sgd(learning_rate=0.1)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        batch = {"labels": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)}

        new_model, new_opt_state, loss = train_step(model, opt_state, batch, optimizer, jax.random.PRNGKey(0))

        self.assertFalse(np.isnan(float(loss)))
        self.assertFalse(np.isinf(float(loss)))
        self.assertNotEqual(float(new_model.weight), float(model.weight))
        self.assertNotEqual(float(new_model.bias), float(model.bias))
        self.assertIsNot(new_opt_state, opt_state)

    def test_eval_step_matches_loss_fn(self) -> None:
        model = _AffineModel(weight=0.75, bias=0.5)
        batch = {"labels": jnp.array([1.0, -2.0, 0.03], dtype=jnp.float32)}
        expected = loss_fn(model, batch, key=None)
        got = eval_step(model, batch)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


class TrainLoopTest(TestCase):
    def test_get_jax_dataloader_calls_components(self) -> None:
        captured: dict[str, Any] = {}
        expected_root = Path("/tmp") / "pcqm4m-v2"

        class FakeDataset:
            def __init__(self, dataset_root, split=None, split_file=None, **_kwargs):
                captured["dataset_root"] = dataset_root
                captured["split"] = split
                captured["split_file"] = split_file

        class FakeLoader:
            def __init__(self, dataset, batch_size, shuffle, drop_last, pad_to_multiple=16, seed=None):
                captured["loader_dataset"] = dataset
                captured["batch_size"] = batch_size
                captured["shuffle"] = shuffle
                captured["drop_last"] = drop_last
                captured["seed"] = seed
                captured["pad_to_multiple"] = pad_to_multiple

            def __iter__(self):
                return iter([])

            def __len__(self):
                return 0

        with (
            patch.object(TRAIN, "PCQMDataset", FakeDataset),
            patch.object(TRAIN, "PCQMDataloader", FakeLoader),
        ):
            loader = get_jax_dataloader(expected_root / "processed", split="train", batch_size=7, shuffle=True, drop_last=False)

        self.assertEqual(captured["dataset_root"], expected_root)
        self.assertEqual(captured["split"], "train")
        self.assertIsInstance(loader, FakeLoader)
        self.assertEqual(captured["batch_size"], 7)
        self.assertTrue(captured["shuffle"])
        self.assertFalse(captured["drop_last"])

    def test_train_uses_doubled_batch_size_for_valid_dataloader(self) -> None:
        recorded: list[tuple[str, int, bool, bool]] = []

        class FakeLoader:
            def __iter__(self):
                return iter([{"labels": np.array([1.0], dtype=np.float32)}])

            def __len__(self):
                return 1

        def capture_loaders(
            hdf5_path, split, batch_size, shuffle, drop_last=False, *args, **kwargs
        ):
            recorded.append((split, batch_size, shuffle, drop_last))
            return FakeLoader()

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(TRAIN, "get_jax_dataloader", capture_loaders),
                patch.object(TRAIN, "get_model", lambda _key: _AffineModel()),
                patch.object(TRAIN.eqx, "tree_serialise_leaves", lambda path, model: None),
            ):
                TRAIN.train(
                    num_epochs=1,
                    batch_size=9,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    model_save_path=str(Path(temp_dir) / "best.eqx"),
                    scheduler_period=None,
                )

        self.assertEqual(
            recorded,
            [
                ("train", 9, True, True),
                ("valid", 18, False, False),
            ],
        )

    def test_train_loop_uses_patched_step_and_saves_best(self) -> None:
        class FakeLoader:
            def __init__(self, batches: list[dict[str, np.ndarray]], expected_batch_size: int):
                self._batches = batches
                self.batch_size = expected_batch_size

            def __iter__(self):
                return iter(self._batches)

            def __len__(self):
                return len(self._batches)

        train_loader = FakeLoader(
            batches=[
                {"labels": np.array([1.0, 2.0], dtype=np.float32)},
                {"labels": np.array([3.0], dtype=np.float32)},
            ],
            expected_batch_size=2,
        )
        valid_loader = FakeLoader(
            batches=[{"labels": np.array([0.5, 0.25], dtype=np.float32)}],
            expected_batch_size=2,
        )

        calls = {"train_step": 0, "eval_step": 0, "saved": []}

        class TrainModel(eqx.Module):
            weight: jnp.ndarray

            def __init__(self, weight: float = 0.0):
                self.weight = jnp.array(weight, dtype=jnp.float32)

            def __call__(self, batch, training=False, key=None):
                return batch["labels"].reshape(-1, 1) * self.weight

        def fake_get_jax_dataloader(
            hdf5_path, split, batch_size, shuffle, drop_last=False, *args, **kwargs
        ):
            if split == "train":
                return train_loader
            if split == "valid":
                return valid_loader
            raise AssertionError(f"Unexpected split={split}")

        def fake_get_model(key):
            return TrainModel()

        def fake_train_step(model, opt_state, batch, optimizer, key):
            calls["train_step"] += 1
            updated_model = eqx.tree_at(lambda m: m.weight, model, model.weight + 1.0)
            return updated_model, opt_state, jnp.sum(batch["labels"]) / batch["labels"].shape[0]

        def fake_eval_step(model, batch):
            calls["eval_step"] += 1
            return jnp.sum(batch["labels"]) / batch["labels"].shape[0]

        def fake_serialize(path, model):
            calls["saved"].append((str(path), model))

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(TRAIN, "get_jax_dataloader", fake_get_jax_dataloader),
                patch.object(TRAIN, "get_model", fake_get_model),
                patch.object(TRAIN, "train_step", fake_train_step),
                patch.object(TRAIN, "eval_step", fake_eval_step),
                patch.object(TRAIN.eqx, "tree_serialise_leaves", fake_serialize),
            ):
                model = TRAIN.train(
                    num_epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    model_save_path=str(Path(temp_dir) / "best.eqx"),
                    scheduler_period=None,
                )

        self.assertEqual(calls["train_step"], len(train_loader))
        self.assertEqual(calls["eval_step"], len(valid_loader))
        self.assertEqual(len(calls["saved"]), 1)
        self.assertIsInstance(model, TrainModel)
        self.assertIsInstance(calls["saved"][0][1], TrainModel)

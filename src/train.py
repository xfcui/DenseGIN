"""Training loop, optimizer construction, and LR/WD schedules for PCQM4Mv2."""

import sys
from pathlib import Path

# Add project root and src directories to path so imports work both when
# running this file directly and via `python -m src.train`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings
# Suppress pkg_resources deprecation warning from outdated package (dependency of ogb)
warnings.filterwarnings('ignore', category=UserWarning, module='outdated')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from optax._src import base as optax_base
import equinox as eqx
from tqdm import tqdm
from dataset import PCQMDataset, PCQMDataloader
from model import get_model


def _attr_segment_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Attribute names along a JAX pytree path (ignore sequence/dict keys)."""
    out: list[str] = []
    for k in path:
        if isinstance(k, jtu.GetAttrKey):
            out.append(k.name)
    return tuple(out)


def lr_multiplier_for_param_path(path: tuple[Any, ...]) -> float:
    """Per-parameter LR scale relative to the global optimizer LR.

    - 0.5×: ``ScaleLayer.scale``; ``HeadKernel`` readout scalars; atom embedding table and
      atom position encoder (``atom_embed``, ``atom_pos``).
    - 4×: ConvKernel bond/degree embedding tensors, LoRA factors; HeadKernel ``act_out`` only.
    """
    names = _attr_segment_names(path)
    if not names:
        return 1.0
    leaf = names[-1]
    if leaf in ("scale", "readout_scale", "readout_bias"):
        return 0.5
    if names[0] in ("atom_embed", "atom_pos"):
        return 0.5
    if names[0] == "head":
        if "act_out" in names:
            return 4.0
    if "conv" in names:
        if any(name in ("lora_down", "lora_up") for name in names):
            return 4.0
        if leaf == "embeddings" and ("embed_edge" in names or "embed_deg" in names):
            return 4.0
    return 1.0


def wd_multiplier_for_param_path(path: tuple[Any, ...]) -> float:
    """Per-parameter weight-decay scale relative to the global WD (before scheduling).

    - 1.0×: ``kernel`` (linear / grouped linear weights).
    - 0.5×: embedding tables.
    - 0.0×: ``scale``, ``bias``, readout scalars, ``lora_down`` / ``lora_up`` (product decay
      is applied separately via :func:`_add_lora_product_decay`), and any unknown leaf names.
    """
    names = _attr_segment_names(path)
    if not names:
        return 0.0
    leaf = names[-1]
    if leaf in ("scale", "bias", "lora_down", "lora_up", "readout_scale", "readout_bias"):
        return 0.0
    if leaf == "embeddings":
        return 0.5
    if leaf == "kernel":
        return 1.0
    return 0.0


def per_param_lr_multiplier_tree(params: Any) -> Any:
    """PyTree matching ``params`` with a positive float LR multiplier per array leaf."""
    return jtu.tree_map_with_path(
        lambda path, leaf: lr_multiplier_for_param_path(path),
        params,
    )


def per_param_wd_multiplier_tree(params: Any) -> Any:
    """PyTree matching ``params`` with a non-negative float WD multiplier per array leaf."""
    return jtu.tree_map_with_path(
        lambda path, leaf: wd_multiplier_for_param_path(path),
        params,
    )


def _scale_updates_by_lr_multipliers(multipliers: Any) -> optax.GradientTransformation:
    """Multiply each update leaf by the corresponding scalar (fixed multipliers)."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        scaled = jtu.tree_map(lambda u, m: u * m, updates, multipliers)
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def _make_lr_schedule(steps_per_epoch: int, k: int, peak_lr: float) -> Callable[[jax.Array], jax.Array]:
    """JAX schedule ``count -> lr`` matching :func:`get_scheduled_hparams` (LR branch)."""
    gr = (1.0 + jnp.sqrt(5.0)) / 2.0
    k_f = float(k)
    spe = float(steps_per_epoch)

    def schedule(count: jax.Array) -> jax.Array:
        epoch_frac = count.astype(jnp.float32) / spe
        period = jnp.floor(epoch_frac / k_f).astype(jnp.int32)
        t = jnp.fmod(epoch_frac, k_f) / k_f
        warmup = peak_lr * t
        constant = peak_lr
        exp = jnp.maximum(period.astype(jnp.float32) - 2.0, 0.0)
        start = peak_lr / jnp.power(gr, exp)
        end = start / (gr * gr)
        cosine = end + 0.5 * (start - end) * (1.0 + jnp.cos(jnp.pi * t))
        return jnp.where(period == 0, warmup, jnp.where(period == 1, constant, cosine))

    return schedule


def _make_wd_schedule(steps_per_epoch: int, k: int, peak_wd: float) -> Callable[[jax.Array], jax.Array]:
    """JAX schedule ``count -> weight_decay`` matching :func:`get_scheduled_hparams` (WD branch)."""
    k_f = float(k)
    spe = float(steps_per_epoch)

    def schedule(count: jax.Array) -> jax.Array:
        epoch_frac = count.astype(jnp.float32) / spe
        period = jnp.floor(epoch_frac / k_f).astype(jnp.int32)
        t = jnp.fmod(epoch_frac, k_f) / k_f
        wd_low = peak_wd / 1000.0
        p0 = wd_low
        p1 = wd_low + (peak_wd - wd_low) * t
        p2plus = peak_wd
        return jnp.where(period == 0, p0, jnp.where(period == 1, p1, p2plus))

    return schedule


def _add_scaled_decayed_weights(
    weight_decay: float | Callable[[jax.Array], jax.Array],
    wd_multiplier_tree: Any,
) -> optax.GradientTransformation:
    """Add ``weight_decay * wd_mult * param`` to each gradient leaf.

    Supports scalar ``weight_decay`` or a schedule ``count -> wd``; when a schedule is used,
    step count is advanced each update (unlike optax's callable-``weight_decay`` path).

    ``wd_multiplier_tree`` matches trainable params; multiplier ``0`` disables decay for that leaf.
    """

    def init_fn(params):
        del params
        if callable(weight_decay):
            return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)
        if callable(weight_decay):
            s = weight_decay(state.count)
            new_state: Any = optax.ScaleByScheduleState(
                count=optax.safe_int32_increment(state.count),
            )
        else:
            s = jnp.asarray(weight_decay, dtype=jnp.float32)
            new_state = state

        def _scaled_decay(g, m, p):
            if g is None:
                return None
            m_arr = jnp.asarray(m, dtype=p.dtype)
            s_arr = jnp.astype(s, p.dtype)
            return g + s_arr * m_arr * p

        new_updates = jax.tree.map(
            _scaled_decay,
            updates,
            wd_multiplier_tree,
            params,
            is_leaf=lambda x: x is None,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def _add_lora_product_decay(
    weight_decay: float | Callable[[jax.Array], jax.Array],
    lora_product_wd_multiplier: float,
) -> optax.GradientTransformation:
    """Add decoupled weight decay on ``lora_down @ lora_up`` for sibling LoRA pairs.

    For each pair (A, B) = (``lora_down``, ``lora_up``), adds to gradients the terms from
    ``(wd/2) * mult * ||A @ B||_F^2``, i.e. ``wd * mult * A @ B @ B.T`` on A and
    ``wd * mult * A.T @ A @ B`` on B. Pairs are found by flattening the param tree with paths
    and matching leaves whose final segment is ``lora_down`` / ``lora_up`` under the same parent.

    Supports scalar ``weight_decay`` or a schedule ``count -> wd``; when a schedule is used,
    step count is advanced each update (same pattern as :func:`_add_scaled_decayed_weights`).
    """

    def init_fn(params):
        del params
        if callable(weight_decay):
            return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)
        if callable(weight_decay):
            s = weight_decay(state.count)
            new_state: Any = optax.ScaleByScheduleState(
                count=optax.safe_int32_increment(state.count),
            )
        else:
            s = jnp.asarray(weight_decay, dtype=jnp.float32)
            new_state = state

        mult = jnp.asarray(lora_product_wd_multiplier, dtype=jnp.float32)
        coeff = jnp.astype(s, jnp.float32) * mult

        pl_p, _treedef_p = jtu.tree_flatten_with_path(
            params, is_leaf=lambda x: x is None
        )
        pl_u, treedef_u = jtu.tree_flatten_with_path(
            updates, is_leaf=lambda x: x is None
        )
        paths_p = [p for p, _ in pl_p]
        flat_p = [l for _, l in pl_p]
        paths_u = [p for p, _ in pl_u]
        flat_u = [l for _, l in pl_u]
        if len(flat_p) != len(flat_u) or paths_p != paths_u or _treedef_p != treedef_u:
            raise ValueError("params and updates must have identical pytree structure for LoRA WD")

        parent_to_idx: dict[tuple[Any, ...], dict[str, int]] = {}
        for i, path in enumerate(paths_p):
            if not path:
                continue
            last = path[-1]
            if isinstance(last, jtu.GetAttrKey) and last.name in ("lora_down", "lora_up"):
                parent = path[:-1]
                parent_to_idx.setdefault(parent, {})[last.name] = i

        new_flat = list(flat_u)
        for _parent, idx_map in parent_to_idx.items():
            if "lora_down" not in idx_map or "lora_up" not in idx_map:
                continue
            i_d = idx_map["lora_down"]
            i_u = idx_map["lora_up"]
            A = flat_p[i_d]
            B = flat_p[i_u]
            g_d = new_flat[i_d]
            g_u = new_flat[i_u]
            if A is None or B is None or g_d is None or g_u is None:
                continue
            c = jnp.astype(coeff, A.dtype)
            new_flat[i_d] = g_d + c * (A @ B @ B.T)
            new_flat[i_u] = g_u + c * (A.T @ A @ B)

        new_updates = jtu.tree_unflatten(treedef_u, new_flat)
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(
    learning_rate: float | Callable[[jax.Array], jax.Array],
    weight_decay: float | Callable[[jax.Array], jax.Array],
    wd_multiplier_tree: Any,
    lr_multiplier_tree: Any,
    *,
    lora_product_wd_multiplier: float = 0.5,
) -> optax.GradientTransformation:
    """Custom Adan-based optimizer with weight decay.

    Adan (Adaptive Nesterov Momentum) is a fast-converging optimizer that uses
    first, second, and third-order moments to adapt the step size.

    Args:
        learning_rate: Global learning rate (scalar or schedule ``count -> lr``).
        weight_decay: L2 regularisation coefficient (scalar or schedule).
        wd_multiplier_tree: PyTree matching trainable params; per-leaf multiplier for decay.
        lr_multiplier_tree: PyTree matching trainable params; each leaf is a positive
            float factor applied before ``scale_by_learning_rate``.
        lora_product_wd_multiplier: Scale for ``||lora_down @ lora_up||_F^2`` decay relative
            to global ``weight_decay`` (LoRA factors themselves use 0× in ``wd_multiplier_tree``).

    Returns:
        An optax.GradientTransformation implementing the optimizer chain.
    """
    wd_transform = _add_scaled_decayed_weights(weight_decay, wd_multiplier_tree)
    lora_wd_transform = _add_lora_product_decay(weight_decay, lora_product_wd_multiplier)
    return optax.chain(
        optax.scale_by_adan(),
        wd_transform,
        lora_wd_transform,
        optax.scale_by_learning_rate(learning_rate),
        _scale_updates_by_lr_multipliers(lr_multiplier_tree),
    )


def get_scheduled_hparams(
    epoch_fractional: float,
    k: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, float]:
    """Return scheduled learning rate and weight decay for a fractional epoch.

    Args:
        epoch_fractional: Current fractional epoch (e.g., epoch + batch_idx / steps_per_epoch).
        k: Period length in epochs.
        learning_rate: Peak learning rate (held constant in period 1; cosine decay from period 2).
        weight_decay: Target weight decay (ramped in period 1; constant from period 2).

    Returns:
        Tuple of (learning_rate, weight_decay) for this step.
    """
    gr = (1 + math.sqrt(5)) / 2
    period = int(epoch_fractional // k)  # 0-indexed period number
    t = (epoch_fractional % k) / k       # fractional position within period [0, 1)

    if period == 0:
        current_lr = learning_rate * t
        current_wd = weight_decay / 1000.0
    elif period == 1:
        current_lr = learning_rate
        wd_low = weight_decay / 1000.0
        current_wd = wd_low + (weight_decay - wd_low) * t
    else:
        # Cosine decay: period N>=2 matches former period N-1 (exponent shift by -1)
        lr_start = learning_rate / gr ** (period - 2)
        lr_end = lr_start / gr ** 2
        current_lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))
        current_wd = weight_decay

    return current_lr, current_wd


def _resolve_dataset_root(hdf5_path: str | Path) -> Path:
    """Resolve dataset root from legacy `processed`-path-style input."""
    base = Path(hdf5_path)
    if base.name == "processed":
        return base.parent
    if base.name == "pcqm4m-v2":
        return base
    if (base / "processed").is_dir():
        return base
    return base.parent


def get_jax_dataloader(
    hdf5_path: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    seed: int | None = None,
):
    """Instantiate a ``PCQMDataloader`` for the given split and batch size."""
    dataset_root = _resolve_dataset_root(hdf5_path)
    dataset = PCQMDataset(dataset_root=dataset_root, split=split)
    return PCQMDataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def to_jax_batch(batch):
    """Convert a dataloader batch to JAX arrays, keeping ``batch_n_graphs`` as a Python int."""
    converted = {}
    for k, v in batch.items():
        if k == "batch_n_graphs":
            converted[k] = int(v)
        else:
            converted[k] = jnp.asarray(v)
    return converted


def _check_nan_loss(x: jax.Array) -> None:
    """Host callback: raise if loss is non-finite."""
    if jnp.isnan(x):
        raise ValueError("Loss is NaN")


def loss_fn(model, batch, key, threshold=6e-2):
    """MAE loss between model predictions and labels; key=None runs deterministic inference."""
    preds = model(batch, training=(key is not None), key=key)
    preds = preds.squeeze(-1)  # (B, 1) -> (B,)
    loss = jnp.mean(jnp.abs(preds - batch["labels"]))
    jax.debug.callback(_check_nan_loss, loss)
    return loss


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    # scale_by_trust_ratio requires params to have the same tree structure as updates;
    # filter to arrays only, matching how opt_state was initialised.
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, batch):
    """Evaluation without gradients; key=None disables dropout for deterministic inference."""
    return loss_fn(model, batch, key=None)


def _train_one_epoch(
    model,
    optimizer,
    opt_state,
    train_loader,
    train_key: jax.Array,
    epoch: int,
    steps_per_epoch: int,
    scheduler_period: int | None,
    learning_rate: float,
    weight_decay: float,
):
    """Run one full training epoch and return updated model, opt state, key, and avg loss."""
    total_train_loss = 0.0
    num_train_batches = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(train_pbar):
        batch = to_jax_batch(batch)
        train_key, step_key = jax.random.split(train_key)
        model, opt_state, loss = train_step(model, opt_state, batch, optimizer, step_key)
        loss_val = loss.item()
        total_train_loss += loss_val
        num_train_batches += 1
        epoch_frac = (epoch * steps_per_epoch + batch_idx) / steps_per_epoch
        if scheduler_period is not None:
            cur_lr, cur_wd = get_scheduled_hparams(
                epoch_frac, scheduler_period, learning_rate, weight_decay
            )
        else:
            cur_lr, cur_wd = learning_rate, weight_decay
        train_pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{cur_lr:.2e}", wd=f"{cur_wd:.2e}")
    avg_train_loss = total_train_loss / max(num_train_batches, 1)
    return model, opt_state, train_key, avg_train_loss


def _validate_one_epoch(model, valid_loader, epoch: int) -> float:
    """Run one full validation epoch and return the average MAE loss."""
    total_valid_loss = 0.0
    num_valid_batches = 0
    valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]")
    for batch in valid_pbar:
        loss = eval_step(model, to_jax_batch(batch))
        loss_val = loss.item()
        total_valid_loss += loss_val
        num_valid_batches += 1
        valid_pbar.set_postfix(loss=f"{loss_val:.4f}")
    return total_valid_loss / max(num_valid_batches, 1)


def train(num_epochs=1, batch_size=32, learning_rate=1e-2, weight_decay=1e-2,
          model_save_path="results/best_model.eqx",
          scheduler_period=None,
          seed: int = 0):
    """
    Train the GNN model on PCQM4Mv2 dataset.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (peak learning rate if using warmup schedule)
        weight_decay: L2 regularisation coefficient.
        model_save_path: Path to save the best model. Default: "results/best_model.eqx".
        scheduler_period: Period k for geometric LR scheduler. If None, use constant LR.
        seed: RNG seed for model init, training minibatch order (via dataloader), and dropout.

    Returns:
        Trained model
    """
    hdf5_path = "dataset/pcqm4m-v2/processed"
    train_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    valid_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='valid',
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
    )

    steps_per_epoch = len(train_loader)
    if scheduler_period is not None:
        lr_sched = _make_lr_schedule(steps_per_epoch, scheduler_period, learning_rate)
        wd_sched = _make_wd_schedule(steps_per_epoch, scheduler_period, weight_decay)
    else:
        lr_sched, wd_sched = learning_rate, weight_decay

    # Initialize model
    key = jax.random.PRNGKey(int(seed))
    model_key, train_key = jax.random.split(key)
    model = get_model(model_key)

    params = eqx.filter(model, eqx.is_array)
    lr_mult_tree = per_param_lr_multiplier_tree(params)
    wd_mult_tree = per_param_wd_multiplier_tree(params)

    print(f"Using Adan optimizer with lr={learning_rate} and wd={weight_decay}")
    optimizer = make_optimizer(
        learning_rate=lr_sched,
        weight_decay=wd_sched,
        wd_multiplier_tree=wd_mult_tree,
        lr_multiplier_tree=lr_mult_tree,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    print()

    # Track best validation loss
    best_valid_loss = float('inf')
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model, opt_state, train_key, avg_train_loss = _train_one_epoch(
            model, optimizer, opt_state, train_loader, train_key, epoch,
            steps_per_epoch, scheduler_period, learning_rate, weight_decay,
        )
        avg_valid_loss = _validate_one_epoch(model, valid_loader, epoch)

        msg = f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}"
        if avg_valid_loss < best_valid_loss:
            eqx.tree_serialise_leaves(model_save_path, model)
            best_valid_loss = avg_valid_loss
            msg += " *"
        print(f"\r{msg}")

    print(f"Training complete. Best validation loss: {best_valid_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--scheduler_period', type=int, default=8, help='Period for geometric LR scheduler')
    parser.add_argument('--model_save_path', type=str, default="results/best_model.eqx", help='Path to save the best model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for init, training shuffle, and dropout')
    args = parser.parse_args()

    train(
        num_epochs=args.scheduler_period**2,
        batch_size=args.batch_size-1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_save_path=args.model_save_path,
        scheduler_period=args.scheduler_period,
        seed=args.seed,
    )

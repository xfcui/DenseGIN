import sys
from pathlib import Path
# Add parent directory to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
# Suppress pkg_resources deprecation warning from outdated package (dependency of ogb)
warnings.filterwarnings('ignore', category=UserWarning, module='outdated')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import math
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm import tqdm
from dataset import PCQMDataset, PCQMDataloader
from model import get_model


def make_optimizer(
    learning_rate: float,
    weight_decay: float,
    mask,
) -> optax.GradientTransformation:
    """Custom Adan-based optimizer with weight decay.

    Adan (Adaptive Nesterov Momentum) is a fast-converging optimizer that uses
    first, second, and third-order moments to adapt the step size.

    Args:
        learning_rate: Global learning rate (scalar or schedule).
        weight_decay: L2 regularisation coefficient.
        mask: Optional pytree mask for weight_decay.

    Returns:
        An optax.GradientTransformation implementing the optimizer chain.
    """
    return optax.chain(
        # Use Adan for adaptive step sizing based on multiple moments
        optax.scale_by_adan(),
        optax.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def get_scheduled_lr(epoch_fractional: float, k: int, learning_rate: float) -> float:
    """Return learning rate for a given fractional epoch under the geometric period schedule.

    Args:
        epoch_fractional: Current fractional epoch (e.g., epoch + batch_idx / steps_per_epoch).
        k: Period length in epochs.
        learning_rate: Peak learning rate (period-1 maximum).

    Returns:
        Learning rate for this step.
    """
    gr = (1 + math.sqrt(5)) / 2
    period = int(epoch_fractional // k)  # 0-indexed period number
    t = (epoch_fractional % k) / k       # fractional position within period [0, 1)

    if period == 0:
        # Linear warm-up: 0 → learning_rate
        return learning_rate * t
    else:
        # Cosine decay within period i (1-indexed i = period+1, maps to i-2 = period-1)
        lr_start = learning_rate / gr ** (period - 1)
        lr_end = lr_start / gr ** 2
        # Cosine annealing from lr_start down to lr_end
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))


def get_scheduled_wd(epoch_fractional: float, k: int, weight_decay: float) -> float:
    """Return weight decay for a given fractional epoch.

    During period 0, linear increase from 0 to weight_decay.
    Following periods, remain constant at weight_decay.

    Args:
        epoch_fractional: Current fractional epoch.
        k: Period length in epochs.
        weight_decay: Target weight decay.

    Returns:
        Weight decay for this step.
    """
    period = int(epoch_fractional // k)
    t = (epoch_fractional % k) / k

    if period == 0:
        # Linear increase: 0 → weight_decay
        return weight_decay * t
    else:
        # Constant
        return weight_decay


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
):
    dataset_root = _resolve_dataset_root(hdf5_path)
    dataset = PCQMDataset(dataset_root=dataset_root, split=split)
    return PCQMDataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def to_jax_batch(batch):
    converted = {}
    for k, v in batch.items():
        if k == "batch_n_graphs":
            converted[k] = int(v)
        else:
            converted[k] = jnp.asarray(v)
    return converted


def loss_fn(model, batch, key):
    preds = model(batch, training=(key is not None), key=key)
    preds = preds.squeeze(-1)  # (B, 1) -> (B,)

    # NaN detection via debug.callback: works inside JIT without breaking compilation
    def check_nan(x):
        if jnp.isnan(x):
            raise ValueError("Loss is NaN")

    # MAE with threshold mask to ignore near-zero residuals (numerical noise)
    loss = jnp.abs(preds - batch['labels'])
    loss = jnp.mean(loss, where=loss > 2e-2)
    jax.debug.callback(check_nan, loss)
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


def train(num_epochs=1, batch_size=32, learning_rate=1e-2, weight_decay=1e-2,
          model_save_path="models/best_model.eqx",
          scheduler_period=None):
    """
    Train the GNN model on PCQM4Mv2 dataset.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (peak learning rate if using warmup schedule)
        model_save_path: Path to save the best model. Default: "models/best_model.eqx".
        scheduler_period: Period k for geometric LR scheduler. If None, use constant LR.

    Returns:
        Trained model
    """
    hdf5_path = "dataset/pcqm4m-v2/processed"
    train_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    valid_loader = get_jax_dataloader(
        hdf5_path=hdf5_path,
        split='valid',
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Initialize model
    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)
    model = get_model(model_key)

    mask = jax.tree_util.tree_map_with_path(
        lambda path, _: path[-1].name == "kernel",
        eqx.filter(model, eqx.is_array)
    ) if weight_decay > 0.0 else None

    print(f"Using Adan optimizer with lr={learning_rate} and wd={weight_decay}")
    optimizer = make_optimizer(learning_rate=learning_rate, weight_decay=weight_decay, mask=mask)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    print()
    
    # Track best validation loss
    best_valid_loss = float('inf')
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Training loop
        total_train_loss = 0.0
        num_train_batches = 0
        
        # Get total steps per epoch for fractional epoch calculation
        steps_per_epoch = len(train_loader)
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # Update learning rate if scheduler is enabled
            if scheduler_period is not None:
                epoch_fractional = epoch + batch_idx / steps_per_epoch
                current_lr = get_scheduled_lr(epoch_fractional, scheduler_period, learning_rate)
                current_wd = get_scheduled_wd(epoch_fractional, scheduler_period, weight_decay)
                # Update optax state for learning rate and weight decay
                for i in range(len(opt_state)):
                    if hasattr(opt_state[i], 'hyperparams'):
                        if 'learning_rate' in opt_state[i].hyperparams:
                            opt_state[i].hyperparams['learning_rate'] = current_lr
                        if 'weight_decay' in opt_state[i].hyperparams:
                            opt_state[i].hyperparams['weight_decay'] = current_wd

            batch = to_jax_batch(batch)
            train_key, step_key = jax.random.split(train_key)
            model, opt_state, loss = train_step(model, opt_state, batch, optimizer, step_key)
            total_train_loss += loss.item()
            num_train_batches += 1
            
            # Update progress bar with current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation loop
        total_valid_loss = 0.0
        num_valid_batches = 0
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]")
        for batch in valid_pbar:
            loss = eval_step(model, to_jax_batch(batch))
            total_valid_loss += loss.item()
            num_valid_batches += 1
            valid_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_valid_loss = total_valid_loss / num_valid_batches

        current_lr = get_scheduled_lr(epoch + 1.0, scheduler_period, learning_rate) \
            if scheduler_period is not None else learning_rate
        current_wd = get_scheduled_wd(epoch + 1.0, scheduler_period, weight_decay) \
            if scheduler_period is not None else weight_decay
        msg = f"Epoch {epoch} | LR: {current_lr:.2e} | WD: {current_wd:.2e} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}"
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            msg += " *"
            eqx.tree_serialise_leaves(model_save_path, model)
            
        # Print combined message at the end of epoch
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
    parser.add_argument('--model_save_path', type=str, default="models/best_model.eqx", help='Path to save the best model')
    args = parser.parse_args()

    train(
        num_epochs=args.scheduler_period**2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_save_path=args.model_save_path,
        scheduler_period=args.scheduler_period
    )


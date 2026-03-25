#!/usr/bin/env python3
"""Parameter count, throughput, and approximate GPU memory for DuAxMPNN (ablation-aware)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import equinox as eqx
import jax
import jax.numpy as jnp

from dataset import EDGE_FEAT_VOCAB_SIZES
from model import AblationConfig, get_model


def _count_params(model: eqx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def _minimal_jax_batch() -> dict[str, jnp.ndarray | int]:
    """Same layout as tst/test_model_compat._make_minimal_batch (one real graph + padding)."""
    node_feat = jnp.zeros((3, 10), dtype=jnp.int32)
    node_embd = jnp.zeros((3, 17), dtype=jnp.float32)
    batch: dict[str, jnp.ndarray | int] = {
        "node_feat": node_feat,
        "node_embd": node_embd,
        "node_batch": jnp.array([0, 1, 1], dtype=jnp.int32),
        "batch_n_graphs": 1,
        "labels": jnp.array([0.0], dtype=jnp.float32),
    }
    suffixes = list(EDGE_FEAT_VOCAB_SIZES.keys())
    for suffix in suffixes:
        dim = len(EDGE_FEAT_VOCAB_SIZES[suffix])
        batch[f"edge{suffix}_index"] = jnp.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=jnp.int32)
        feat = jnp.zeros((4, dim), dtype=jnp.int32)
        feat = feat.at[0].set(jnp.arange(1, dim + 1) % max(1, dim))
        batch[f"edge{suffix}_feat"] = feat
        batch[f"edge{suffix}_batch"] = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--depth-mode", type=str, default="dense")
    args = parser.parse_args()

    cfg = AblationConfig(max_hops=args.max_hops, depth_mode=args.depth_mode)
    key = jax.random.PRNGKey(0)
    model = get_model(key, config=cfg)
    n_params = _count_params(model)
    print(f"Trainable parameters: {n_params:,}")

    batch = _minimal_jax_batch()

    @eqx.filter_jit
    def forward(m, b):
        return m(b, training=False, key=None)

    y = forward(model, batch)
    jax.block_until_ready(y)
    for _ in range(args.warmup):
        y = forward(model, batch)
        jax.block_until_ready(y)

    t0 = time.perf_counter()
    for _ in range(args.repeats):
        y = forward(model, batch)
        jax.block_until_ready(y)
    elapsed = time.perf_counter() - t0
    graphs = int(batch["batch_n_graphs"])
    tput = graphs * args.repeats / elapsed
    print(f"Forward: {args.repeats} steps in {elapsed:.4f}s → ~{tput:.1f} graphs/s (minimal padded batch)")

    try:
        dev = jax.local_devices()[0]
        ms = dev.memory_stats()
        if ms:
            print(f"Device memory stats (after forward): {ms}")
    except Exception as e:  # noqa: BLE001
        print(f"(memory_stats unavailable: {e})")


if __name__ == "__main__":
    main()

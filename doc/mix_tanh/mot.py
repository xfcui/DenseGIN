"""
Mixture of Tanh (MoTanh) — learnable monotone embedding for continuous values.

Maps a scalar continuous input (distance, angle, charge, electronegativity, …)
to a bounded representation in (−1, 1) via a learned mixture of tanh bases:

    w_i = softplus(ŵ_i) / Σ_j softplus(ŵ_j)
    s_i = softplus(ŝ_i)
    f(x) = Σ_i  w_i · tanh(s_i · x)

Each basis tanh(s_i · x) responds at a different scale, so the mixture
learns *how* to partition the input range — steep bases resolve fine detail
near zero while gentle bases capture the tails.  The convex combination
keeps the output bounded and monotone (order-preserving).

2K learnable parameters per instance (K raw weights + K raw scales).
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx


def _softplus_inv(y: float) -> float:
    """Inverse softplus: returns x such that softplus(x) = y."""
    if y <= 0:
        raise ValueError("softplus_inv requires y > 0")
    return math.log(math.expm1(y))


class MoTanh(eqx.Module):
    """Mixture of Tanh — learnable monotone embedding for continuous values.

    Embeds scalar continuous inputs (distances, charges, electronegativity, …)
    into a bounded (−1, 1) representation via ``f(x) = Σ_i w_i · tanh(s_i · x)``.
    Mixing weights are softplus-normalised and scales are softplus-positive,
    guaranteeing a monotone, order-preserving, odd mapping.

    Parameters
    ----------
    num_bases : int
        Number of tanh components (K).  Default 8.
    num_channels : int
        If 1 (default), one shared embedding applied element-wise.
        If C > 1, each of the last C input dimensions gets its own
        independent mixture (input trailing dim must equal C), useful
        when different feature channels represent different physical
        quantities with distinct scale structure.
    key : ignored
        Accepted for API compatibility with other Equinox modules;
        initialisation is deterministic.
    """

    raw_weights: jax.Array
    raw_scales: jax.Array
    num_bases: int = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_bases: int = 32,
        num_channels: int = 1,
        *,
        key: jax.Array | None = None,
    ) -> None:
        self.num_bases = num_bases
        self.num_channels = num_channels

        self.raw_weights = jnp.zeros((num_channels, num_bases))

        lo, hi = _softplus_inv(0.1), _softplus_inv(10.0)
        self.raw_scales = jnp.tile(
            jnp.linspace(lo, hi, num_bases), (num_channels, 1)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        sp = jax.nn.softplus(self.raw_weights)
        w = sp / sp.sum(axis=-1, keepdims=True)     # (C, K)
        s = jax.nn.softplus(self.raw_scales)         # (C, K)

        if self.num_channels == 1:
            w, s = w.squeeze(0), s.squeeze(0)        # (K,)
            return jnp.tanh(x[..., None] * s) @ w

        # per-channel: x (*, C) → (*, C)
        return (jnp.tanh(x[..., None] * s) * w).sum(-1)

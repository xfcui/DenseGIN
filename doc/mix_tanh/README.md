# Mixture of Tanh (MoTanh)

Learnable monotone embedding for continuous values: ℝ → (−1, 1).

Maps scalar continuous inputs — distances, angles, energies, or any
unbounded real value — into a bounded representation via a learned mixture
of tanh basis functions.  Each basis responds at a different scale, so
the network learns *how* to partition the input range: steep bases resolve
fine detail near zero while gentle bases capture the tails.

![MoTanh expressivity](mot.png)

## Definition

```
w_i = softplus(ŵ_i) / Σ_j softplus(ŵ_j)      (mixing weights)
s_i = softplus(ŝ_i)                            (positive scales)

f(x) = Σ_i  w_i · tanh(s_i · x)
```

Learnable parameters per instance: **2K** (K raw weights `ŵ`, K raw scales `ŝ`).

## Properties

| Property | Guarantee |
|---|---|
| Output range | (−1, 1) — convex combination of bounded values |
| Monotonicity | Always increasing — order-preserving embedding |
| Symmetry | Odd: f(−x) = −f(x), f(0) = 0 |
| Differentiable | Everywhere, smooth gradients |

## Usage

```python
from layer import MoTanh

# embed a single continuous quantity (e.g. interatomic distance)
embed = MoTanh(num_bases=8)
y = embed(distances)                # distances: any shape → y: same shape

# per-channel: each feature dimension is a different physical quantity
# (e.g. RWPE + electronegativity + Gasteiger charge + 3D coords)
embed = MoTanh(num_bases=8, num_channels=17)
y = embed(x)                        # x: (*, 17) → y: (*, 17)
```

## Why softplus normalisation instead of softmax?

- `softmax = exp(x_i) / Σ exp(x_j)` — exponential saturation can push weights
  to near-zero, making dormant bases hard to revive during training.
- `softplus(x_i) / Σ softplus(x_j)` — polynomial-like growth near zero gives
  smoother gradients; no weight is ever fully suppressed.

## Why softplus scales instead of raw scales?

Raw scales can go negative, breaking monotonicity. Wrapping through softplus
guarantees `s_i > 0` at all times — the embedding stays order-preserving.

## Expressivity

Controlled by K (number of basis functions):

| K | Character | Params |
|---|---|---|
| 2 | Two S-curves — limited to simple sigmoid shapes | 4 |
| 4 | Piecewise — can blend slow and fast transitions | 8 |
| 8 | Flexible — diverse steepness profiles | 16 |
| 16 | Smooth detail | 32 |
| 32 | Near-arbitrary monotone shape | 64 |

## Limitation

All basis functions `tanh(s_i · x)` are centered at the origin with no bias,
so **every realisable curve passes through (0, 0)** and is an odd function.
The variety comes entirely from different steepness profiles — how fast the
embedding transitions between −1 and +1.

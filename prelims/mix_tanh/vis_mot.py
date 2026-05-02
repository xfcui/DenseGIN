"""Visualize MoTanh expressivity across different numbers of basis functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── numerics ──────────────────────────────────────────────────────────────────

def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _softplus_inv(y):
    return np.log(np.expm1(np.asarray(y, dtype=float)))


def _normalize(raw_w):
    sp = _softplus(raw_w)
    return sp / sp.sum(axis=-1, keepdims=True)


def forward_batch(x, raw_weights, raw_scales):
    """Vectorised forward: (n, K) params, (M,) grid → (n, M) curves."""
    w = _normalize(raw_weights)                        # (n, K)
    s = _softplus(raw_scales)                          # (n, K)
    basis = np.tanh(s[:, None, :] * x[None, :, None])  # (n, M, K)
    return np.einsum("nmk,nk->nm", basis, w)


def forward(x, raw_weights, raw_scales):
    """Single curve forward: (K,) params, (M,) grid → (M,)."""
    w = _normalize(raw_weights)
    s = _softplus(raw_scales)
    return np.tanh(x[:, None] * s[None, :]) @ w


# ── config ────────────────────────────────────────────────────────────────────

RNG      = np.random.default_rng(42)
X        = np.linspace(-6, 6, 500)
N_RANDOM = 100
BASES    = [2, 4, 8, 16, 32]

BG     = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
ACCENT = "#58a6ff"
CURVE  = "#79c0ff"
ALPHA_C = 0.15
ALPHA_M = 0.90
MEAN_W  = 2.2

EXTREMES_COLOR = ["#ff7b72", "#ffa657", "#3fb950", "#d2a8ff"]
EXTREME_LABELS = ["nearly flat", "step-like (sharp)", "gentle slope", "mixed scales"]

LABEL_MAP = {
    2:  "K=2   (two S-curves)",
    4:  "K=4   (piecewise)",
    8:  "K=8   (flexible)",
    16: "K=16  (smooth detail)",
    32: "K=32  (near-arbitrary)",
}


# ── curve generators ──────────────────────────────────────────────────────────

def sample_curves(K, n):
    """Draw n random MoTanh curves with K bases (fully vectorised)."""
    raw_w = RNG.normal(0, 1.0, size=(n, K))
    raw_s = RNG.normal(0, 1.5, size=(n, K))
    return forward_batch(X, raw_w, raw_s)


def extreme_curves(K):
    """Four hand-crafted configs that span the realisable shape range."""
    if K < 2:
        return [np.tanh(X)] * 4

    s_lo  = float(_softplus_inv(0.15))
    s_hi  = float(_softplus_inv(10.0))
    s_mid = float(_softplus_inv(0.5))

    configs = [
        (np.zeros(K), np.full(K, s_lo)),                                       # nearly flat
        (_dom_weights(K), _dom_scales(K, s_lo, s_hi)),                         # step-like
        (np.zeros(K), np.linspace(s_lo, s_mid, K)),                            # gentle slope
        (np.zeros(K), _mixed_scales(K)),                                       # mixed
    ]
    return [forward(X, *c) for c in configs]


def _dom_weights(K):
    w = np.full(K, -5.0); w[0] = 5.0; return w

def _dom_scales(K, lo, hi):
    s = np.full(K, lo); s[0] = hi; return s

def _mixed_scales(K):
    lo = float(_softplus_inv(0.3))
    hi = float(_softplus_inv(4.0))
    return np.concatenate([np.full(K // 2, lo), np.full(K - K // 2, hi)])


# ── plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, col, ylabel=None):
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors="#8b949e", labelsize=7)
    if col == 0 and ylabel:
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    else:
        ax.set_yticklabels([])


def main():
    fig = plt.figure(figsize=(20, 9), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, len(BASES), figure=fig,
        hspace=0.55, wspace=0.28,
        top=0.86, bottom=0.08, left=0.04, right=0.98,
    )
    dx = X[1] - X[0]

    for col, K in enumerate(BASES):
        ax_top = fig.add_subplot(gs[0, col], facecolor=PANEL)
        ax_bot = fig.add_subplot(gs[1, col], facecolor=PANEL)

        curves   = sample_curves(K, N_RANDOM)
        extremes = extreme_curves(K)
        mean_c   = curves.mean(axis=0)

        # ── top: f(x) ────────────────────────────────────────────────────
        for c in curves:
            ax_top.plot(X, c, color=CURVE, alpha=ALPHA_C, linewidth=0.7)
        for i, (ec, lab) in enumerate(zip(extremes, EXTREME_LABELS)):
            ax_top.plot(X, ec, color=EXTREMES_COLOR[i], linewidth=1.8,
                        alpha=0.85, zorder=5, label=lab)
        ax_top.plot(X, mean_c, color=ACCENT, linewidth=MEAN_W,
                    alpha=ALPHA_M, zorder=4, label="mean")
        ax_top.plot(X, np.tanh(X), color="#8b949e", linewidth=1.0,
                    linestyle="--", alpha=0.5, zorder=3, label="tanh")

        for yv in (-1, 0, 1):
            ax_top.axhline(yv, color=GRID, linewidth=0.5,
                           linestyle=":" if yv else "-", zorder=0)
        ax_top.axvline(0, color=GRID, linewidth=0.5, zorder=0)
        ax_top.set_xlim(-6, 6)
        ax_top.set_ylim(-1.18, 1.18)
        ax_top.set_title(LABEL_MAP[K], color="#e6edf3", fontsize=10.5,
                         pad=7, fontweight="bold")
        _style_ax(ax_top, col, ylabel="f(x)")

        # ── bottom: f'(x) ────────────────────────────────────────────────
        derivs = np.gradient(curves, dx, axis=1)
        mean_d = derivs.mean(axis=0)

        for d in derivs:
            ax_bot.plot(X, d, color="#3fb950", alpha=ALPHA_C * 0.9,
                        linewidth=0.7)
        for i, ec in enumerate(extremes):
            ax_bot.plot(X, np.gradient(ec, dx), color=EXTREMES_COLOR[i],
                        linewidth=1.8, alpha=0.85, zorder=5)
        ax_bot.plot(X, mean_d, color="#3fb950", linewidth=MEAN_W,
                    alpha=ALPHA_M, zorder=4)
        ax_bot.plot(X, 1 - np.tanh(X)**2, color="#8b949e", linewidth=1.0,
                    linestyle="--", alpha=0.5, zorder=3)

        ax_bot.axhline(0, color=GRID, linewidth=0.5, zorder=0)
        ax_bot.axvline(0, color=GRID, linewidth=0.5, zorder=0)
        ax_bot.set_xlim(-6, 6)
        ax_bot.set_ylim(-0.05, None)
        ax_bot.set_xlabel("x", color="#8b949e", fontsize=9)
        _style_ax(ax_bot, col, ylabel="f ʼ(x)")

    # ── legend ────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color=CURVE, alpha=0.7, linewidth=1.5,
               label=f"{N_RANDOM} random draws"),
        Line2D([0], [0], color=ACCENT, linewidth=MEAN_W, label="mean"),
        Line2D([0], [0], color="#8b949e", linewidth=1.2, linestyle="--",
               label="tanh ref"),
    ] + [
        Line2D([0], [0], color=c, linewidth=1.8, label=lab)
        for c, lab in zip(EXTREMES_COLOR, EXTREME_LABELS)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, labelcolor="#e6edf3", fontsize=9,
               bbox_to_anchor=(0.5, 0.975))

    fig.suptitle(
        "MoTanh  —  Mixture of Tanh Expressivity vs. Number of Bases\n"
        r"$w_i = \mathrm{softplus}(\hat w_i)\,/\,\sum_j"
        r" \mathrm{softplus}(\hat w_j)$"
        r"$,\quad s_i = \mathrm{softplus}(\hat s_i)$"
        r"$,\quad f(x) = \sum_i w_i \cdot \tanh(s_i\, x)$",
        color="#e6edf3", fontsize=13, y=1.02, va="bottom",
    )

    plt.savefig("layer/mot.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("saved → layer/mot.png")
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build figure 2 (MoAct): panels (a) and (c) from exact MoAct math in src/model.py;
panel (b) is cropped from an existing AI-rendered PNG and composited in the middle.

Run from repo root or from this directory. Defaults resolve from the current working
directory; use --fig-dir to override.

Example:
  cd article/lncs/fig && python gen_fig2_plots.py
  python article/lncs/fig/gen_fig2_plots.py --fig-dir article/lncs/fig
"""

from __future__ import annotations

import argparse
from math import erf as math_erf
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

# Match src/model.py MoAct + EPSILON
EPSILON = 1e-6
K_DEFAULT = 8


def inverse_softplus(y: np.ndarray | float) -> np.ndarray:
    """NumPy inverse softplus (same as _inverse_softplus in model.py)."""
    return np.log(np.expm1(np.asarray(y, dtype=np.float64)))


def softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # numerically stable
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -500.0, 500.0))))


def moact_scales(num_bases: int = K_DEFAULT) -> np.ndarray:
    """Per-channel scale init: softplus(linspace(inv_sp(0.1), inv_sp(10), K))."""
    lo = float(inverse_softplus(0.1))
    hi = float(inverse_softplus(10.0))
    raw = np.linspace(lo, hi, num_bases)
    return softplus_np(raw)


def moact_uniform_weights(num_bases: int = K_DEFAULT) -> np.ndarray:
    """Initial weights are zeros -> softplus(0)=log(2) each, then normalized."""
    w = softplus_np(np.zeros(num_bases))
    return w / (w.sum() + EPSILON)


def moact_mixture(x: np.ndarray, scales: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """f(x) = sum_i w_i * tanh(s_i * x)."""
    t = np.tanh(x[..., None] * scales[None, :])
    return (t * weights[None, :]).sum(axis=-1)


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (numpy; uses math.erf for wide NumPy compatibility)."""
    sqrt2 = np.sqrt(2.0)
    v = np.vectorize(lambda t: 0.5 * t * (1.0 + math_erf(t / sqrt2)), otypes=[float])
    return v(x)


def discrete_binning_4(x: np.ndarray) -> np.ndarray:
    """Four equal-width bins on [-3, 3], ascending step values in [-1, 1]."""
    edges = np.linspace(-3.0, 3.0, 5)
    idx = np.digitize(x, edges[1:-1], right=False)
    idx = np.clip(idx, 0, 3)
    vals = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float64)
    return vals[idx]


def _annotate_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    text: str,
    xy_point: tuple[float, float],
    xy_text: tuple[float, float],
    color: str = "0.2",
) -> None:
    ax.annotate(
        text,
        xy=xy_point,
        xytext=xy_text,
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def plot_panel_a(ax: plt.Axes, scales: np.ndarray) -> None:
    x = np.linspace(-3.0, 3.0, 400)
    cmap = mpl.colormaps["turbo"]
    colors = cmap(np.linspace(0.15, 0.95, len(scales)))

    for i, s in enumerate(scales):
        ax.plot(x, np.tanh(s * x), color=colors[i], lw=1.2)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$\tanh(s_i \cdot x)$", fontsize=10)
    ax.set_title("(a) Scaled tanh basis functions", fontsize=10, pad=6)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    ax.tick_params(labelsize=8)

    # Representative scales: closest indices to 0.1, 1, 10
    targets = (0.1, 1.0, 10.0)
    labels = (r"$s \approx 0.1$ (gentle)", r"$s \approx 1$ (moderate)", r"$s \approx 10$ (sharp)")
    for tgt, lab in zip(targets, labels):
        j = int(np.argmin(np.abs(scales - tgt)))
        sj = float(scales[j])
        ycurve = np.tanh(sj * x)
        if sj >= 5.0:
            xi, yi = 0.25, float(np.interp(0.25, x, ycurve))
            xy_text = (0.9, 0.88)
        elif sj <= 0.2:
            xi, yi = -2.4, float(np.interp(-2.4, x, ycurve))
            xy_text = (-2.5, 0.72)
        else:
            xi, yi = 1.05, float(np.interp(1.05, x, ycurve))
            xy_text = (1.55, 0.48)
        _annotate_curve(ax, x, ycurve, lab, (xi, yi), xy_text)


def plot_panel_c(ax: plt.Axes, scales: np.ndarray, w: np.ndarray) -> None:
    x = np.linspace(-3.0, 3.0, 600)
    moa = moact_mixture(x, scales, w)
    relu = np.maximum(0.0, x)
    gelu = gelu_np(x)
    bins_y = discrete_binning_4(x)

    ax.plot(x, moa, color="C0", lw=2.0, label="MoAct (init, uniform weights)")
    ax.plot(x, relu, color="0.45", ls="--", lw=1.2, label="ReLU")
    ax.plot(x, gelu, color="0.45", ls=":", lw=1.2, label="GELU")
    ax.plot(x, bins_y, color="0.45", ls="-.", lw=1.2, label="Discrete binning (4 bins)")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.2, 3.2)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel("output", fontsize=10)
    ax.set_title("(c) Comparison with alternatives", fontsize=10, pad=6)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    ax.tick_params(labelsize=8)
    leg = ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
    leg.get_frame().set_linewidth(0.4)

    _annotate_curve(
        ax,
        x,
        moa,
        "bounded & monotone",
        (1.4, float(np.interp(1.4, x, moa))),
        (0.3, 1.35),
        color="C0",
    )
    _annotate_curve(
        ax,
        x,
        relu,
        "unbounded",
        (2.5, 2.5),
        (1.2, 2.85),
        color="0.35",
    )
    _annotate_curve(
        ax,
        x,
        bins_y,
        "destroys ordering",
        (-0.2, float(np.interp(-0.2, x, bins_y))),
        (-2.2, 0.5),
        color="0.35",
    )


def build_figure(
    panel_b: Image.Image,
    scales: np.ndarray,
    weights: np.ndarray,
    dpi: int = 200,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.35),
        dpi=dpi,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.92, wspace=0.22)

    plot_panel_a(axes[0], scales)
    axes[1].imshow(np.asarray(panel_b), aspect="equal")
    axes[1].axis("off")
    plot_panel_c(axes[2], scales, weights)

    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MoAct fig2 panels (a)(c) + composite with panel (b).")
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory containing fig2_moact_v*.png (default: cwd, or parent of this script)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--bases", type=int, default=K_DEFAULT)
    args = p.parse_args()

    cwd = Path.cwd()
    if args.fig_dir is not None:
        fig_dir = args.fig_dir.resolve()
    else:
        # Default to the directory of this script
        fig_dir = Path(__file__).resolve().parent

    scales = moact_scales(args.bases)
    weights = moact_uniform_weights(args.bases)

    # Process all fig2_moact_v*.png files
    import glob
    pattern = str(fig_dir / "fig2_moact_v*.png")
    files = sorted([f for f in glob.glob(pattern) if "_panel_b" not in f])
    
    if not files:
        print(f"No files matching {pattern} found.")
        return

    for f_path_str in files:
        source = Path(f_path_str)
        try:
            img = Image.open(source).convert("RGB")
        except Exception as e:
            print(f"Error reading {source}: {e}")
            continue
            
        w, h = img.size
        # Check if it's an original AI image (roughly 3:1 aspect ratio, e.g. 1792x592)
        # or if it's already composited (e.g. 2300x670)
        
        panel_b_path = source.with_name(f"{source.stem}_panel_b.png")
        
        if w == 2300 and h == 670:
            if panel_b_path.exists():
                print(f"Updating {source.name} using cached {panel_b_path.name}...")
                panel_b = Image.open(panel_b_path).convert("RGB")
            else:
                print(f"Skipping {source.name}: already composited but no cached panel (b) found. Delete it to regenerate.")
                continue
        else:
            print(f"Processing {source.name} (size {w}x{h}). caching panel (b)...")
            pw = w // 3
            panel_b = img.crop((pw, 0, 2 * pw, h))
            panel_b.save(panel_b_path)
        
        fig = build_figure(panel_b, scales, weights, dpi=args.dpi)
        
        # Overwrite the file
        fig.savefig(source, dpi=args.dpi, facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  -> Updated {source.name} ({source.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()

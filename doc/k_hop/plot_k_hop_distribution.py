import argparse
import os
import os.path as osp
import importlib.util
import re
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.abspath(osp.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = osp.join(PROJECT_ROOT, "src")
DATASET_DIR = osp.join(SRC_DIR, "dataset")


def _load_dataset_modules():
    """Load dataset graph/features without importing dataset.__init__."""
    if DATASET_DIR not in sys.path:
        sys.path.insert(0, DATASET_DIR)

    pkg_name = "_k_hop_dataset"
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [DATASET_DIR]
        sys.modules[pkg_name] = pkg

    features_path = osp.join(DATASET_DIR, "features.py")
    graph_path = osp.join(DATASET_DIR, "graph.py")

    feature_spec = importlib.util.spec_from_file_location(f"{pkg_name}.features", features_path)
    feature_mod = importlib.util.module_from_spec(feature_spec)
    sys.modules[f"{pkg_name}.features"] = feature_mod
    feature_spec.loader.exec_module(feature_mod)  # type: ignore[union-attr]

    graph_spec = importlib.util.spec_from_file_location(f"{pkg_name}.graph", graph_path)
    graph_mod = importlib.util.module_from_spec(graph_spec)
    sys.modules[f"{pkg_name}.graph"] = graph_mod
    graph_spec.loader.exec_module(graph_mod)  # type: ignore[union-attr]

    return graph_mod, feature_mod


MOL_GRAPH_MODULE, FEATURES_MODULE = _load_dataset_modules()
mol_to_graph = MOL_GRAPH_MODULE.mol_to_graph
FEATURE_VOCAB = FEATURES_MODULE.FEATURE_VOCAB


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot k-hop edge statistics from random molecules."
    )
    parser.add_argument(
        "--raw-csv-path",
        type=str,
        default="dataset/pcqm4m-v2/raw/data.csv.gz",
        help="Path to raw CSV file that contains a 'smiles' column.",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=10000,
        help="Number of random molecules to sample from raw SMILES.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for molecule sampling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="k_hop_10k",
        help=(
            "Base output stem or directory (relative to doc/k_hop). "
            "Default produces filenames like "
            "`k_hop_2hop_vs_heavy_10k.svg` and "
            "`k_hop_feat_distribution_10k.png`."
        ),
    )
    return parser.parse_args()


def load_smiles_samples(raw_csv_path: str, num_molecules: int, seed: int) -> np.ndarray:
    if num_molecules <= 0:
        raise ValueError("--num-molecules must be > 0")
    if not osp.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    data_df = pd.read_csv(raw_csv_path)
    if "smiles" not in data_df.columns:
        raise KeyError(f"'smiles' column not found in: {raw_csv_path}")

    smiles = data_df["smiles"].dropna().to_numpy()
    if smiles.size == 0:
        raise ValueError("No non-empty SMILES found in raw CSV.")

    sample_size = min(num_molecules, smiles.size)
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(smiles.size, size=sample_size, replace=False)
    return smiles[sampled_idx]


def _format_sample_tag(num_molecules: int) -> str:
    if num_molecules >= 1000 and num_molecules % 1000 == 0:
        return f"{num_molecules // 1000}k"
    return str(num_molecules)


def collect_khop_stats(smiles_samples: np.ndarray) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    int,
]:
    heavy_counts = []
    khop_counts = {2: [], 3: [], 4: []}
    khop_feat_arrays = {2: [], 3: [], 4: []}
    invalid_smiles = 0

    for smiles in smiles_samples:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue

        mol_with_hs = Chem.AddHs(mol)
        heavy = mol_with_hs.GetNumHeavyAtoms()
        try:
            graph = mol_to_graph(smiles)
        except Exception:
            invalid_smiles += 1
            continue

        if graph is None:
            invalid_smiles += 1
            continue

        heavy_counts.append(heavy)

        for k in (2, 3, 4):
            edge_key = f"edge_index_{k}hop"
            feat_key = f"edge_feat_{k}hop"
            edge_count = int(graph[edge_key].shape[1] // 2)
            khop_counts[k].append(edge_count)

            feats = np.asarray(graph[feat_key])
            if feats.size == 0:
                continue
            khop_feat_arrays[k].append(feats)

    if not heavy_counts:
        raise ValueError("No valid molecules found in sampled SMILES.")

    khop_counts_arr = {k: np.asarray(v, dtype=np.int32) for k, v in khop_counts.items()}
    khop_feat_arr = {}
    for k in (2, 3, 4):
        if len(khop_feat_arrays[k]) == 0:
            khop_feat_arr[k] = np.zeros((0, k), dtype=np.int64)
        else:
            khop_feat_arr[k] = np.concatenate(khop_feat_arrays[k], axis=0).astype(np.int64)

    return (
        np.asarray(heavy_counts, dtype=np.int32),
        khop_counts_arr,
        khop_feat_arr,
        invalid_smiles,
    )


def _tick_values(max_value: int, target_count: int = 8) -> list[int]:
    if max_value <= 0:
        return [0]
    step = max(1, int(np.ceil(max_value / target_count)))
    ticks = list(range(0, max_value + 1, step))
    if ticks[-1] != max_value:
        ticks.append(max_value)
    return ticks


def _render_joint_svg(
    x_counts: np.ndarray,
    y_counts: np.ndarray,
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    x_marginal_label: str,
    y_marginal_label: str,
    x_bar_color: str,
    y_bar_color: str,
    bubble_stroke_color: str,
    bubble_rgb_high: tuple[int, int, int],
    bubble_rgb_low: tuple[int, int, int],
) -> str:
    """Shared renderer for joint bubble plots with axis marginal histograms."""
    if not out_path.lower().endswith(".svg"):
        base, _ = osp.splitext(out_path)
        out_path = f"{base}.svg"

    max_x = int(x_counts.max()) if x_counts.size else 0
    max_y = int(y_counts.max()) if y_counts.size else 0
    x_freq = np.bincount(x_counts, minlength=max_x + 1)
    y_freq = np.bincount(y_counts, minlength=max_y + 1)

    pairs = np.stack([x_counts, y_counts], axis=1)
    unique_pairs, pair_freq = np.unique(pairs, axis=0, return_counts=True)
    order = np.argsort(pair_freq)
    unique_pairs = unique_pairs[order]
    pair_freq = pair_freq[order]

    width = 1100
    height = 860
    margin_left = 100
    margin_top = 70
    margin_bottom = 90
    margin_right = 40
    top_hist_height = 120
    right_hist_width = 150
    gap = 24

    main_x0 = margin_left
    main_y0 = margin_top + top_hist_height + gap
    main_width = width - margin_left - margin_right - right_hist_width - gap
    main_height = height - main_y0 - margin_bottom
    top_y0 = margin_top
    right_x0 = main_x0 + main_width + gap

    x_den = max(max_x, 1)
    y_den = max(max_y, 1)
    max_top_freq = max(int(x_freq.max()), 1)
    max_right_freq = max(int(y_freq.max()), 1)
    max_pair_freq = int(pair_freq.max()) if pair_freq.size else 1

    x_ticks = _tick_values(max_x, target_count=8)
    y_ticks = _tick_values(max_y, target_count=8)
    mean_x = float(np.mean(x_counts))
    mean_y = float(np.mean(y_counts))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="550" y="36" text-anchor="middle" font-size="24" font-family="Arial">{title} ({len(x_counts)} Random Molecules)</text>',
        f'<rect x="{main_x0}" y="{main_y0}" width="{main_width}" height="{main_height}" fill="white" stroke="#222" stroke-width="2"/>',
        f'<rect x="{main_x0}" y="{top_y0}" width="{main_width}" height="{top_hist_height}" fill="white" stroke="#222" stroke-width="1.5"/>',
        f'<rect x="{right_x0}" y="{main_y0}" width="{right_hist_width}" height="{main_height}" fill="white" stroke="#222" stroke-width="1.5"/>',
    ]

    for x_tick in x_ticks:
        x = main_x0 + (x_tick / x_den) * main_width
        svg.append(
            f'<line x1="{x:.2f}" y1="{main_y0}" x2="{x:.2f}" y2="{main_y0 + main_height}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{main_y0 + main_height + 22}" text-anchor="middle" font-size="11" font-family="Arial">{x_tick}</text>'
        )
    for y_tick in y_ticks:
        y = main_y0 + main_height - (y_tick / y_den) * main_height
        svg.append(
            f'<line x1="{main_x0}" y1="{y:.2f}" x2="{main_x0 + main_width}" y2="{y:.2f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{main_x0 - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{y_tick}</text>'
        )

    mean_px = main_x0 + (mean_x / x_den) * main_width
    mean_py = main_y0 + main_height - (mean_y / y_den) * main_height
    svg.append(
        f'<line x1="{mean_px:.2f}" y1="{main_y0}" x2="{mean_px:.2f}" y2="{main_y0 + main_height}" stroke="#555" stroke-width="1.5" stroke-dasharray="6,5"/>'
    )
    svg.append(
        f'<line x1="{main_x0}" y1="{mean_py:.2f}" x2="{main_x0 + main_width}" y2="{mean_py:.2f}" stroke="#555" stroke-width="1.5" stroke-dasharray="6,5"/>'
    )

    x_bar_w = main_width / max(max_x + 1, 1)
    for x_val in range(max_x + 1):
        x0 = main_x0 + x_val * x_bar_w + 1.5
        bar_h = (x_freq[x_val] / max_top_freq) * (top_hist_height - 10)
        y0 = top_y0 + top_hist_height - bar_h
        svg.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(x_bar_w - 3, 1):.2f}" height="{bar_h:.2f}" fill="{x_bar_color}"/>'
        )

    y_bar_h = main_height / max(max_y + 1, 1)
    for y_val in range(max_y + 1):
        y = main_y0 + main_height - (y_val + 1) * y_bar_h
        bar_w = (y_freq[y_val] / max_right_freq) * (right_hist_width - 10)
        svg.append(
            f'<rect x="{right_x0}" y="{y + 1.5:.2f}" width="{bar_w:.2f}" height="{max(y_bar_h - 3, 1):.2f}" fill="{y_bar_color}"/>'
        )

    rh, gh, bh = bubble_rgb_high
    rl, gl, bl = bubble_rgb_low
    for (xv, yv), freq in zip(unique_pairs.tolist(), pair_freq.tolist()):
        px = main_x0 + (xv / x_den) * main_width
        py = main_y0 + main_height - (yv / y_den) * main_height
        strength = np.log1p(freq) / np.log1p(max_pair_freq)
        radius = 2.2 + 9.0 * strength
        fill_alpha = 0.30 + 0.55 * strength
        cr = int(rl + (rh - rl) * strength)
        cg = int(gl + (gh - gl) * strength)
        cb = int(bl + (bh - bl) * strength)
        svg.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" '
            f'fill="rgb({cr},{cg},{cb})" fill-opacity="{fill_alpha:.2f}" '
            f'stroke="{bubble_stroke_color}" stroke-width="0.6"/>'
        )

    svg.extend(
        [
            f'<text x="{main_x0 + main_width / 2}" y="{height - 32}" text-anchor="middle" font-size="16" font-family="Arial">{x_label}</text>',
            f'<text x="34" y="{main_y0 + main_height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 34 {main_y0 + main_height / 2})">{y_label}</text>',
            f'<text x="{main_x0 + main_width / 2}" y="{top_y0 - 8}" text-anchor="middle" font-size="13" font-family="Arial">{x_marginal_label}</text>',
            f'<text x="{right_x0 + right_hist_width / 2}" y="{main_y0 - 8}" text-anchor="middle" font-size="13" font-family="Arial">{y_marginal_label}</text>',
            f'<text x="{main_x0 + 10}" y="{main_y0 + 22}" text-anchor="start" font-size="12" font-family="Arial" fill="#555">Dashed lines: means ({mean_x:.2f}, {mean_y:.2f})</text>',
            f'<text x="{main_x0 + 10}" y="{main_y0 + 40}" text-anchor="start" font-size="12" font-family="Arial" fill="#555">Bubble size/color: frequency of each pair</text>',
            "</svg>",
        ]
    )

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    return out_path


def plot_khop_vs_heavy(
    heavy_counts: np.ndarray,
    hop_counts: np.ndarray,
    k: int,
    out_path: str,
) -> str:
    title = f"{k}-hop Edges vs Heavy Atoms"
    if k == 2:
        return _render_joint_svg(
            x_counts=heavy_counts,
            y_counts=hop_counts,
            out_path=out_path,
            title=title,
            x_label="Number of heavy atoms per molecule",
            y_label=f"Number of {k}-hop virtual edges per molecule",
            x_marginal_label="Heavy-atom marginal distribution",
            y_marginal_label=f"{k}-hop edge marginal distribution",
            x_bar_color="#4c78a8",
            y_bar_color="#b279a2",
            bubble_stroke_color="#2a5c8a",
            bubble_rgb_high=(50, 95, 160),
            bubble_rgb_low=(170, 210, 245),
        )
    if k == 3:
        return _render_joint_svg(
            x_counts=heavy_counts,
            y_counts=hop_counts,
            out_path=out_path,
            title=title,
            x_label="Number of heavy atoms per molecule",
            y_label=f"Number of {k}-hop virtual edges per molecule",
            x_marginal_label="Heavy-atom marginal distribution",
            y_marginal_label=f"{k}-hop edge marginal distribution",
            x_bar_color="#e8871e",
            y_bar_color="#5b8abf",
            bubble_stroke_color="#8b4d00",
            bubble_rgb_high=(200, 100, 20),
            bubble_rgb_low=(240, 190, 130),
        )
    return _render_joint_svg(
        x_counts=heavy_counts,
        y_counts=hop_counts,
        out_path=out_path,
        title=title,
        x_label="Number of heavy atoms per molecule",
        y_label=f"Number of {k}-hop virtual edges per molecule",
        x_marginal_label="Heavy-atom marginal distribution",
        y_marginal_label=f"{k}-hop edge marginal distribution",
        x_bar_color="#54a24b",
        y_bar_color="#8e79a5",
        bubble_stroke_color="#2d6a27",
        bubble_rgb_high=(45, 120, 50),
        bubble_rgb_low=(160, 220, 160),
    )


def _path_count_labels(vocab_values: list) -> list[str]:
    labels = []
    for item in vocab_values:
        if item == "misc" or item == "misc":
            labels.append(">=3")
        else:
            labels.append(str(item))
    return labels


def plot_khop_feature_distributions(
    khop_feats: dict[int, np.ndarray],
    out_path: str,
) -> str:
    labels = _path_count_labels(FEATURE_VOCAB["possible_path_count_list"])
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(18, 4.2), constrained_layout=True)
    bar_width = 0.16
    num_bins = len(labels)

    for idx, k in enumerate((2, 3, 4)):
        ax = axes[idx]
        feats = khop_feats[k]

        if feats.size == 0:
            ax.set_title(f"{k}-hop feature distribution")
            ax.set_xlim(-0.5, num_bins - 0.5)
            ax.set_ylim(0, 1)
            ax.text(
                0.5,
                0.5,
                "No directed k-hop edges found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        x = np.arange(num_bins)
        max_count = int(feats.max()) if feats.size else 0
        color_cycle = ["#4c78a8", "#f58518", "#e4572e", "#72b7b2"]
        for dim in range(feats.shape[1]):
            counts = np.bincount(feats[:, dim], minlength=num_bins)
            ax.bar(
                x + (dim - (feats.shape[1] - 1) / 2) * bar_width,
                counts,
                width=bar_width,
                color=color_cycle[dim % len(color_cycle)],
                alpha=0.85,
                label=f"dim {dim}",
            )
            max_count = max(max_count, int(counts.max()))

        ax.set_title(f"{k}-hop edge feature counts")
        ax.set_xlabel("Path-count bucket")
        ax.set_ylabel("Frequency")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.8, 1), top=max_count * 1.2 + 10)
        ax.legend(fontsize=9)

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _resolve_output_base(out_arg: str) -> str:
    if osp.isabs(out_arg):
        return out_arg
    return osp.join(SCRIPT_DIR, out_arg)


def _infer_output_prefix_and_dir(out_arg: str) -> tuple[str, str]:
    out_path = _resolve_output_base(out_arg)
    out_dir = osp.dirname(out_path)
    if not out_dir:
        out_dir = SCRIPT_DIR

    stem = osp.splitext(osp.basename(out_path))[0]
    if stem == "":
        stem = "k_hop"

    # Strip trailing sample label like _10k, _5k, ...
    stem = re.sub(r"_(\d+[kK]|\d+)$", "", stem)
    # Strip an explicit hop suffix, if provided.
    stem = re.sub(r"_(?:2hop|3hop|4hop)_vs_heavy$", "", stem)
    if stem == "":
        stem = "k_hop"

    return stem, out_dir


def _build_svg_output_path(stem: str, out_dir: str, k: int, sample_suffix: str) -> str:
    return osp.join(out_dir, f"{stem}_{k}hop_vs_heavy_{sample_suffix}.svg")


def _build_feature_output_path(stem: str, out_dir: str, sample_suffix: str) -> str:
    return osp.join(out_dir, f"{stem}_feat_distribution_{sample_suffix}.png")


def main():
    args = parse_args()

    smiles_samples = load_smiles_samples(
        raw_csv_path=args.raw_csv_path,
        num_molecules=args.num_molecules,
        seed=args.seed,
    )

    heavy_counts, khop_counts, khop_feats, invalid_smiles = collect_khop_stats(smiles_samples)

    sample_suffix = _format_sample_tag(args.num_molecules)
    stem, out_dir = _infer_output_prefix_and_dir(args.out)
    if not osp.isabs(out_dir):
        out_dir = osp.join(SCRIPT_DIR, out_dir)

    path_2hop = _build_svg_output_path(stem, out_dir, 2, sample_suffix)
    path_3hop = _build_svg_output_path(stem, out_dir, 3, sample_suffix)
    path_4hop = _build_svg_output_path(stem, out_dir, 4, sample_suffix)
    path_feat = _build_feature_output_path(stem, out_dir, sample_suffix)

    saved_2 = plot_khop_vs_heavy(heavy_counts, khop_counts[2], 2, path_2hop)
    saved_3 = plot_khop_vs_heavy(heavy_counts, khop_counts[3], 3, path_3hop)
    saved_4 = plot_khop_vs_heavy(heavy_counts, khop_counts[4], 4, path_4hop)
    saved_feat = plot_khop_feature_distributions(khop_feats, path_feat)

    print(f"Saved 2-hop edges vs heavy plot to: {saved_2}")
    print(f"Saved 3-hop edges vs heavy plot to: {saved_3}")
    print(f"Saved 4-hop edges vs heavy plot to: {saved_4}")
    print(f"Saved k-hop feature distributions to: {saved_feat}")
    print(f"Requested samples:            {len(smiles_samples)}")
    print(f"Valid molecules used:         {len(heavy_counts)}")
    print(f"Invalid SMILES skipped:       {invalid_smiles}")
    print(f"Mean heavy atoms:             {heavy_counts.mean():.4f}")
    print(f"Mean 2-hop edges:             {khop_counts[2].mean():.4f}")
    print(f"Mean 3-hop edges:             {khop_counts[3].mean():.4f}")
    print(f"Mean 4-hop edges:             {khop_counts[4].mean():.4f}")
    print(f"Median 2-hop edges:           {float(np.median(khop_counts[2])):.1f}")
    print(f"Median 3-hop edges:           {float(np.median(khop_counts[3])):.1f}")
    print(f"Median 4-hop edges:           {float(np.median(khop_counts[4])):.1f}")
    print(f"Max 2-hop edges:              {khop_counts[2].max()}")
    print(f"Max 3-hop edges:              {khop_counts[3].max()}")
    print(f"Max 4-hop edges:              {khop_counts[4].max()}")
    print(f"Feature row totals: 2hop={khop_feats[2].shape[0]}, 3hop={khop_feats[3].shape[0]}, 4hop={khop_feats[4].shape[0]}")


if __name__ == "__main__":
    main()

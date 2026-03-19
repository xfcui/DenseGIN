import argparse
import math
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOC_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.abspath(osp.join(DOC_DIR, "..", ".."))
SRC_DIR = osp.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import mol_to_graph, _rwpe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-dimension RWPE distributions from random molecules."
    )
    parser.add_argument(
        "--raw-csv-path",
        type=str,
        default=osp.join(PROJECT_ROOT, "dataset/pcqm4m-v2/raw/data.csv.gz"),
        help="Path to raw CSV file containing SMILES.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of bins per histogram.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Number of RWPE dimensions to plot.",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=10000,
        help="Number of random molecules used to compute the distributions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for molecule sampling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=osp.join(DOC_DIR, "rwpe_distribution.png"),
        help="Output image path.",
    )
    args = parser.parse_args()
    if args.num_molecules <= 0:
        raise ValueError("--num-molecules must be > 0")
    if args.dim <= 0:
        raise ValueError("--dim must be > 0")
    return args


def load_and_sample_smiles(raw_csv_path: str, num_molecules: int, seed: int) -> np.ndarray:
    if not osp.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    data_df = pd.read_csv(raw_csv_path)
    if "smiles" not in data_df.columns:
        raise KeyError(f"'smiles' column not found in {raw_csv_path}")

    smiles = data_df["smiles"].dropna().to_numpy()
    if smiles.size == 0:
        raise ValueError("No valid SMILES found in raw CSV.")

    sample_size = min(num_molecules, smiles.size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(smiles.size, size=sample_size, replace=False)
    return smiles[idx]


def sample_rwpe_from_smiles(smiles_samples: np.ndarray, dim: int) -> tuple[np.ndarray, int]:
    sampled_chunks = []
    valid_molecules = 0
    for smiles in smiles_samples:
        try:
            graph = mol_to_graph(smiles)
        except Exception:
            continue

        if graph is None:
            continue

        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        num_nodes = int(graph["num_nodes"])
        rwpe = _rwpe(num_nodes=num_nodes, edge_index=edge_index, rwpe_dim=dim)
        if rwpe.shape[0] > 0:
            sampled_chunks.append(rwpe)
            valid_molecules += 1

    if not sampled_chunks:
        raise ValueError("No nodes found in sampled molecules; cannot plot distribution.")

    return np.concatenate(sampled_chunks, axis=0), valid_molecules


def plot_rwpe_distribution(rwpe: np.ndarray, bins: int, out_path: str):
    dim = rwpe.shape[1]
    num_diff = max(0, dim - 2)
    total_plots = dim + num_diff
    ncols = 4
    nrows = math.ceil(total_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    plot_idx = 0
    for d in range(dim):
        ax = axes[plot_idx]
        ax.hist(
            rwpe[:, d],
            bins=bins,
            range=(0.0, 1.0),
            alpha=0.85,
            color="steelblue",
            edgecolor="none",
        )
        ax.set_title(f"RWPE dim {d}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        plot_idx += 1

    for d in range(2, dim):
        ax = axes[plot_idx]
        diff_values = rwpe[:, d] - rwpe[:, d - 2]
        mean_val = float(np.mean(diff_values))
        std_val = float(np.std(diff_values))
        ax.hist(
            diff_values,
            bins=bins,
            range=(-1.0, 1.0),
            alpha=0.85,
            color="darkorange",
            edgecolor="none",
        )
        ax.set_title(f"RWPE dim {d} - dim {d - 2}\nmean={mean_val:.4f}, std={std_val:.4f}")
        ax.set_xlabel("Difference")
        ax.set_ylabel("Count")
        ax.set_xlim(-1.0, 1.0)
        ax.set_yscale("log")
        plot_idx += 1

    for d in range(plot_idx, len(axes)):
        axes[d].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    smiles_samples = load_and_sample_smiles(
        args.raw_csv_path, num_molecules=args.num_molecules, seed=args.seed
    )
    rwpe_sampled, used_molecules = sample_rwpe_from_smiles(smiles_samples, dim=args.dim)

    out_path = args.out if osp.isabs(args.out) else osp.join(DOC_DIR, args.out)
    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_rwpe_distribution(rwpe_sampled, bins=args.bins, out_path=out_path)
    print(f"Saved RWPE distribution plot to: {out_path}")
    print(f"Sampled molecules: {used_molecules}")
    print(f"RWPE rows plotted (sampled nodes): {rwpe_sampled.shape[0]}")
    print(f"RWPE dim plotted: {rwpe_sampled.shape[1]}")


if __name__ == "__main__":
    main()

import argparse
import os
import os.path as osp
import sys
from collections import deque

import numpy as np
import pandas as pd
from rdkit import Chem

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
SRC_DIR = osp.abspath(osp.join(SCRIPT_DIR, "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _is_polar_hydrogen(atom):
    if atom.GetAtomicNum() != 1:
        return False

    neighbors = atom.GetNeighbors()
    return len(neighbors) == 1 and neighbors[0].GetAtomicNum() in {7, 8, 16}


def _keep_atom_mask(mol_with_hs):
    keep_atom = np.zeros(mol_with_hs.GetNumAtoms(), dtype=np.bool_)
    for atom in mol_with_hs.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            keep_atom[idx] = True
            continue
        if _is_polar_hydrogen(atom):
            keep_atom[idx] = True
    return keep_atom


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot graph diameter distribution from raw SMILES:\n"
            "x-axis: number of default-graph nodes (heavy atoms + active H),\n"
            "y-axis: graph diameter (longest shortest-path distance)."
        )
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
        default="diameter_vs_heavy_10k.svg",
        help="Output path for diameter-vs-node-count plot (SVG).",
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


def _active_graph_adj(mol) -> tuple[list[list[int]], int]:
    keep_atom = np.asarray(_keep_atom_mask(mol), dtype=np.bool_)
    num_atoms = mol.GetNumAtoms()
    keep_to_local = np.full(num_atoms, -1, dtype=np.int32)
    local_count = 0
    for atom_idx in range(num_atoms):
        if keep_atom[atom_idx]:
            keep_to_local[atom_idx] = local_count
            local_count += 1

    adjacency = [[] for _ in range(local_count)]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if not keep_atom[a] or not keep_atom[b]:
            continue
        la = int(keep_to_local[a])
        lb = int(keep_to_local[b])
        adjacency[la].append(lb)
        adjacency[lb].append(la)
    return adjacency, local_count


def _graph_diameter_from_adj(adjacency: list[list[int]]) -> int:
    num_nodes = len(adjacency)
    if num_nodes <= 1:
        return 0

    max_diameter = 0
    for src in range(num_nodes):
        distance = np.full(num_nodes, -1, dtype=np.int16)
        distance[src] = 0
        q = deque([src])

        max_from_src = 0
        while q:
            u = q.popleft()
            next_dist = distance[u] + 1
            for v in adjacency[u]:
                if distance[v] != -1:
                    continue
                distance[v] = next_dist
                max_from_src = max(max_from_src, next_dist)
                q.append(v)

        max_diameter = max(max_diameter, max_from_src)
    return int(max_diameter)


def count_diameter_and_node_count(smiles: str) -> tuple[int, int] | None:
    """Return (diameter, node_count), or None if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    adjacency, node_count = _active_graph_adj(mol)
    if node_count == 0:
        return 0, 0

    diameter = _graph_diameter_from_adj(adjacency)
    return diameter, node_count


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
    """Render and write a joint bubble plot with marginal histograms."""
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
        svg.append(f'<line x1="{x:.2f}" y1="{main_y0}" x2="{x:.2f}" y2="{main_y0 + main_height}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{main_y0 + main_height + 22}" text-anchor="middle" font-size="11" font-family="Arial">{x_tick}</text>')
    for y_tick in y_ticks:
        y = main_y0 + main_height - (y_tick / y_den) * main_height
        svg.append(f'<line x1="{main_x0}" y1="{y:.2f}" x2="{main_x0 + main_width}" y2="{y:.2f}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{main_x0 - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{y_tick}</text>')

    mean_px = main_x0 + (mean_x / x_den) * main_width
    mean_py = main_y0 + main_height - (mean_y / y_den) * main_height
    svg.append(f'<line x1="{mean_px:.2f}" y1="{main_y0}" x2="{mean_px:.2f}" y2="{main_y0 + main_height}" stroke="#555" stroke-width="1.5" stroke-dasharray="6,5"/>')
    svg.append(f'<line x1="{main_x0}" y1="{mean_py:.2f}" x2="{main_x0 + main_width}" y2="{mean_py:.2f}" stroke="#555" stroke-width="1.5" stroke-dasharray="6,5"/>')

    x_bar_w = main_width / max(max_x + 1, 1)
    for x_val in range(max_x + 1):
        x0 = main_x0 + x_val * x_bar_w + 1.5
        bar_h = (x_freq[x_val] / max_top_freq) * (top_hist_height - 10)
        y0 = top_y0 + top_hist_height - bar_h
        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(x_bar_w - 3, 1):.2f}" height="{bar_h:.2f}" fill="{x_bar_color}"/>')

    y_bar_h = main_height / max(max_y + 1, 1)
    for y_val in range(max_y + 1):
        y = main_y0 + main_height - (y_val + 1) * y_bar_h
        bar_w = (y_freq[y_val] / max_right_freq) * (right_hist_width - 10)
        svg.append(f'<rect x="{right_x0}" y="{y + 1.5:.2f}" width="{bar_w:.2f}" height="{max(y_bar_h - 3, 1):.2f}" fill="{y_bar_color}"/>')

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

    svg.extend([
        f'<text x="{main_x0 + main_width / 2}" y="{height - 32}" text-anchor="middle" font-size="16" font-family="Arial">{x_label}</text>',
        f'<text x="34" y="{main_y0 + main_height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 34 {main_y0 + main_height / 2})">{y_label}</text>',
        f'<text x="{main_x0 + main_width / 2}" y="{top_y0 - 8}" text-anchor="middle" font-size="13" font-family="Arial">{x_marginal_label}</text>',
        f'<text x="{right_x0 + right_hist_width / 2}" y="{main_y0 - 8}" text-anchor="middle" font-size="13" font-family="Arial">{y_marginal_label}</text>',
        f'<text x="{main_x0 + 10}" y="{main_y0 + 22}" text-anchor="start" font-size="12" font-family="Arial" fill="#555">Dashed lines: means ({mean_x:.2f}, {mean_y:.2f})</text>',
        f'<text x="{main_x0 + 10}" y="{main_y0 + 40}" text-anchor="start" font-size="12" font-family="Arial" fill="#555">Bubble size/color: frequency of each pair</text>',
        "</svg>",
    ])

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    return out_path


def plot_diameter_vs_nodes(
    diameter_counts: np.ndarray,
    node_counts: np.ndarray,
    out_path: str,
) -> str:
    return _render_joint_svg(
        x_counts=node_counts,
        y_counts=diameter_counts,
        out_path=out_path,
        title="Graph Diameter vs Node Count",
        x_label="Number of nodes (heavy atoms + polar H) per molecule",
        y_label="Graph diameter (longest shortest path)",
        x_marginal_label="Node-count marginal distribution",
        y_marginal_label="Graph-diameter marginal distribution",
        x_bar_color="#4c78a8",
        y_bar_color="#e4572e",
        bubble_stroke_color="#1e4f7d",
        bubble_rgb_high=(50, 95, 160),
        bubble_rgb_low=(140, 190, 235),
    )


def _out_stem(out_path: str) -> tuple[str, str]:
    base, ext = osp.splitext(out_path)
    if ext.lower() != ".svg":
        base = out_path
        ext = ".svg"
    return base, ext


def _resolve_output_path(out_path: str) -> str:
    """Resolve output paths against this script directory when relative."""
    if osp.isabs(out_path):
        return out_path
    return osp.join(SCRIPT_DIR, out_path)


def main():
    args = parse_args()
    smiles_samples = load_smiles_samples(
        raw_csv_path=args.raw_csv_path,
        num_molecules=args.num_molecules,
        seed=args.seed,
    )

    diameter_counts = []
    node_counts = []
    invalid_smiles = 0

    for smiles in smiles_samples:
        data = count_diameter_and_node_count(smiles)
        if data is None:
            invalid_smiles += 1
            continue
        diameter_counts.append(data[0])
        node_counts.append(data[1])

    if not diameter_counts:
        raise ValueError("No valid molecules found in sampled SMILES.")

    diameter_counts = np.asarray(diameter_counts, dtype=np.int32)
    node_counts = np.asarray(node_counts, dtype=np.int32)

    out_path = _resolve_output_path(args.out)
    out_path = _out_stem(out_path)[0] + _out_stem(out_path)[1]

    path = plot_diameter_vs_nodes(diameter_counts, node_counts, out_path)

    print(f"Saved diameter-vs-node-count plot to: {path}")
    print(f"Requested samples:              {len(smiles_samples)}")
    print(f"Valid molecules used:           {len(diameter_counts)}")
    print(f"Invalid SMILES skipped:         {invalid_smiles}")
    print(f"Mean diameter:                  {diameter_counts.mean():.4f}")
    print(f"Mean nodes:                     {node_counts.mean():.4f}")
    print(f"Median diameter:                {float(np.median(diameter_counts)):.1f}")
    print(f"Median nodes:                   {float(np.median(node_counts)):.1f}")
    print(f"Max diameter:                   {diameter_counts.max()}")
    print(f"Max nodes:                      {node_counts.max()}")


if __name__ == "__main__":
    main()

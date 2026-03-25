#!/usr/bin/env python3
"""Aggregate ablation training logs into markdown-friendly tables (mean ± std over seeds)."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

EPOCH_BEST_PATTERN = re.compile(
    r"Epoch\s+\d+\s+\|.*Train Loss:\s*([0-9.]+).*Valid Loss:\s*([0-9.]+)\s*\*"
)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def collect_valid_maes(root: Path, folder: str) -> list[float]:
    valids: list[float] = []
    d = root / folder
    if not d.is_dir():
        return valids
    for seed_dir in sorted(d.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        out = seed_dir / "train.out"
        if not out.is_file():
            continue
        text = out.read_text(errors="replace")
        last_valid = None
        for m in EPOCH_BEST_PATTERN.finditer(text):
            last_valid = float(m.group(2))
        if last_valid is not None:
            valids.append(last_valid)
    return valids


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation results under results/.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path.cwd() / "results",
        help="Directory containing baseline/ and ablation_* subfolders.",
    )
    args = parser.parse_args()
    root: Path = args.results_root
    if not root.is_dir():
        print(f"No results directory: {root}", file=sys.stderr)
        sys.exit(1)

    folders_needed = {
        "baseline",
        "ablation_t2_heavy",
        "ablation_t2_allh",
        "ablation_t2_no_nrank",
        "ablation_t2_perbond",
        "ablation_t3_linear",
        "ablation_t3_mlp",
        "ablation_t3_binning",
        "ablation_t3_moact_k4",
        "ablation_t3_moact_k16",
        "ablation_t4_1hop_resnet",
        "ablation_t4_12hop_resnet",
        "ablation_t4_123hop_resnet",
        "ablation_t4_1234hop_resnet",
        "ablation_t4_1hop_dense",
    }
    cache: dict[str, tuple[float, float]] = {}
    for f in folders_needed:
        vm, vs = _mean_std(collect_valid_maes(root, f))
        cache[f] = (vm, vs)

    def cell(folder: str) -> str:
        vm, vs = cache.get(folder, (float("nan"), float("nan")))
        if math.isnan(vm):
            return "—"
        return f"{vm:.4f} ± {vs:.4f}"

    print("# Ablation summary (validation MAE, eV)\n")

    print("## Table 2 — Chemistry-aware construction\n")
    print("| Variant | Valid MAE (eV) |")
    print("|---|---|")
    print(f"| Heavy-atom-only | {cell('ablation_t2_heavy')} |")
    print(f"| All-H | {cell('ablation_t2_allh')} |")
    print(f"| Active-H (ours) | {cell('baseline')} |")
    print(f"| w/o neighbor rank | {cell('ablation_t2_no_nrank')} |")
    print(f"| Absolute EN/GC (ours) | {cell('baseline')} |")
    print(f"| Per-bond EN/GC | {cell('ablation_t2_perbond')} |")

    print("\n## Table 3 — MoAct / continuous edge embedding\n")
    print("| Variant | Valid MAE (eV) |")
    print("|---|---|")
    print(f"| Linear | {cell('ablation_t3_linear')} |")
    print(f"| MLP (softplus) | {cell('ablation_t3_mlp')} |")
    print(f"| Discrete binning | {cell('ablation_t3_binning')} |")
    print(f"| MoAct K=4 | {cell('ablation_t3_moact_k4')} |")
    print(f"| MoAct K=8 (ours) | {cell('baseline')} |")
    print(f"| MoAct K=16 | {cell('ablation_t3_moact_k16')} |")

    print("\n## Table 4 — Dual-axis message passing\n")
    print("| Variant | Valid MAE (eV) |")
    print("|---|---|")
    print(f"| 1-hop, ResNet | {cell('ablation_t4_1hop_resnet')} |")
    print(f"| 1+2-hop, ResNet | {cell('ablation_t4_12hop_resnet')} |")
    print(f"| 1+2+3-hop, ResNet | {cell('ablation_t4_123hop_resnet')} |")
    print(f"| 1–4-hop, ResNet | {cell('ablation_t4_1234hop_resnet')} |")
    print(f"| 1-hop, dense | {cell('ablation_t4_1hop_dense')} |")
    print(f"| 1–4-hop, dense (ours) | {cell('baseline')} |")


if __name__ == "__main__":
    main()

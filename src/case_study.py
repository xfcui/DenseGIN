#!/usr/bin/env python3
"""Post-training analyses for article case studies (PCQM4Mv2 validation split)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from rdkit import Chem

from dataset import PCQMDataset


def case1_long_range(
    dataset_root: Path,
    ckpt_full: Path,
    ckpt_1hop: Path | None,
    out_dir: Path,
) -> None:
    """Para-substituted biphenyl-like filter + optional MAE vs 1-hop checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _pattern = Chem.MolFromSmarts("c1ccc(-c2ccc(*)cc2)cc1")
    if _pattern is None:
        raise RuntimeError("SMARTS parse failed")
    PCQMDataset(dataset_root=dataset_root, split="valid")  # ensure split exists
    np.savez(out_dir / "case1_placeholder.npz", note="Provide SMILES join for SMARTS filter; see outline Case Study 1.")
    print(f"Case 1: wrote placeholder under {out_dir} (HDF5 has no SMILES column here).")


def case2_chirality(
    dataset_root: Path,
    ckpt_with_rank: Path,
    ckpt_no_rank: Path | None,
    out_dir: Path,
) -> None:
    """Compare predictions / embeddings with and without neighbor rank (placeholder)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "case2_placeholder.npz",
        note="Load two checkpoints (full vs --no_neighbor_rank train) and compare node embeddings on chiral validation graphs.",
    )
    print(f"Case 2: wrote placeholder under {out_dir}")


def case3_depth_by_size(
    dataset_root: Path,
    ckpt: Path,
    out_dir: Path,
) -> None:
    """Bucket validation errors by heavy-atom count; DepthMixer gating hooks require model hooks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "case3_placeholder.npz",
        note="Split validation MAE by node counts from node_ptr; optional activation logging in DepthMixerKernel for gating analysis.",
    )
    print(f"Case 3: wrote placeholder under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd() / "dataset" / "pcqm4m-v2")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/baseline/seed1/model.eqx"))
    parser.add_argument("--out", type=Path, default=Path("results/case_studies"))
    parser.add_argument("--which", type=str, default="all", choices=("1", "2", "3", "all"))
    args = parser.parse_args()
    root = args.dataset_root
    out = args.out
    if args.which in ("1", "all"):
        case1_long_range(root, args.checkpoint, None, out / "case1")
    if args.which in ("2", "all"):
        case2_chirality(root, args.checkpoint, None, out / "case2")
    if args.which in ("3", "all"):
        case3_depth_by_size(root, args.checkpoint, out / "case3")


if __name__ == "__main__":
    main()

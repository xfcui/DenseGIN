"""Compatibility module exporting the canonical dataset and dataloader APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util


def _load_submodule(module_name: str, relative_path: str):
    """Load a Python file as a named module, resolving its path relative to this file."""
    module_path = Path(__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATASET_MODULE = _load_submodule("_pcqm_dataset_core", "dataset/dataset.py")
_DATALOADER_MODULE = _load_submodule("_pcqm_dataset_dataloader", "dataset/dataloader.py")


PCQMDataset = _DATASET_MODULE.PCQMDataset
PCQMDataloader = _DATALOADER_MODULE.PCQMDataloader
batch_collapse = _DATASET_MODULE.batch_collapse

PAD_TO_MULTIPLE = _DATASET_MODULE.PAD_TO_MULTIPLE
NODE_FEAT_VOCAB_SIZES = _DATASET_MODULE.NODE_FEAT_VOCAB_SIZES
NODE_FEAT_TOTAL_VOCAB = _DATASET_MODULE.NODE_FEAT_TOTAL_VOCAB
EDGE_FEAT_VOCAB_SIZES = _DATASET_MODULE.EDGE_FEAT_VOCAB_SIZES
EDGE_FEAT_TOTAL_VOCAB = _DATASET_MODULE.EDGE_FEAT_TOTAL_VOCAB


__all__ = [
    "batch_collapse",
    "PCQMDataset",
    "PCQMDataloader",
    "PAD_TO_MULTIPLE",
    "NODE_FEAT_VOCAB_SIZES",
    "NODE_FEAT_TOTAL_VOCAB",
    "EDGE_FEAT_VOCAB_SIZES",
    "EDGE_FEAT_TOTAL_VOCAB",
]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the standalone dataloader smoke-test."""
    parser = argparse.ArgumentParser(
        description="Run one PCQM dataloader epoch and show progress."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "dataset" / "pcqm4m-v2",
        help="Root directory containing processed data and split_dict.h5.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to iterate (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of graphs per batch.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle graph order for the epoch.",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the last incomplete batch.",
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=PAD_TO_MULTIPLE,
        help="Pad node/edge counts to multiples of this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when shuffling.",
    )
    parser.add_argument(
        "--load-in-memory",
        action="store_true",
        default=True,
        help="Load full processed arrays into RAM before iteration.",
    )
    parser.add_argument(
        "--no-load-in-memory",
        action="store_false",
        dest="load_in_memory",
        help="Keep dataset on disk and read features lazily.",
    )
    return parser.parse_args()


def main() -> None:
    """Iterate one dataloader epoch and report throughput via a tqdm progress bar."""
    args = _parse_args()
    from tqdm import tqdm

    dataset = PCQMDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        split_file=args.dataset_root / "split_dict.h5",
        load_in_memory=args.load_in_memory,
    )
    dataloader = PCQMDataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        pad_to_multiple=args.pad_to_multiple,
        seed=args.seed,
    )

    bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch - {args.split}",
        unit="batch",
    )

    n_graphs_seen = 0
    for batch in bar:
        n_graphs_seen += int(batch["batch_n_graphs"])
        bar.set_postfix_str(f"graphs={n_graphs_seen}")

    dataset.close()


if __name__ == "__main__":
    main()

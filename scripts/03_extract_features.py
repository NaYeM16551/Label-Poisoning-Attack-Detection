#!/usr/bin/env python3
"""Extract SSL features for the CIFAR-10 train and test sets.

Usage:
    python scripts/03_extract_features.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torchvision

# Allow `python scripts/...` style execution (insert project root onto sys.path).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.ssl_extractor import SSLFeatureExtractor  # noqa: E402
from src.features.validate_features import run_all_checks  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.data import CIFAR10_CLASSES  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOGGER = get_logger("extract_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute features even if cached.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    encoder_name = cfg["features"]["encoder"]
    cache_dir = Path(cfg["features"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"]["data_dir"]

    extractor = SSLFeatureExtractor(encoder_name, device=cfg["device"])

    cifar_cls = (
        torchvision.datasets.CIFAR10
        if dataset_name == "cifar10"
        else torchvision.datasets.CIFAR100
    )

    for split in ("train", "test"):
        is_train = split == "train"
        dataset = cifar_cls(root=data_dir, train=is_train, download=True)
        cache_path = cache_dir / f"{dataset_name}_{encoder_name}_{split}.npy"
        features = extractor.extract_features(
            dataset,
            batch_size=cfg["features"]["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            normalize=cfg["features"]["normalize"],
            cache_path=cache_path,
            force=args.force,
        )
        LOGGER.info("[%s] features shape=%s -> %s", split, features.shape, cache_path)

        if split == "train" and not args.skip_validation:
            import numpy as np
            labels = np.asarray(dataset.targets, dtype=np.int64)
            class_names = list(CIFAR10_CLASSES) if dataset_name == "cifar10" else None
            run_all_checks(
                features,
                labels,
                save_dir=Path(cfg["output_dir"]) / "validation",
                class_names=class_names,
            )


if __name__ == "__main__":
    main()

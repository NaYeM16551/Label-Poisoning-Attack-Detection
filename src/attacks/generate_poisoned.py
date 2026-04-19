"""End-to-end script: generate and persist poisoned CIFAR datasets.

Usage:
    python -m src.attacks.generate_poisoned --config configs/default.yaml
    python -m src.attacks.generate_poisoned --config configs/default.yaml \\
        --attack random --num_flips 1000

Outputs a torch ``.pt`` file under ``attack.poisoned_data_dir`` containing
``poison_indices``, ``poison_labels`` and ``original_labels``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torchvision

from src.attacks.flip_wrapper import load_flip_labels
from src.attacks.random_flip import generate_random_flip
from src.utils.config import load_config
from src.utils.data import save_poisoned_dataset
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger("generate_poisoned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate poisoned CIFAR datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--attack", choices=["flip", "random"], default=None,
                        help="Override attack type from config.")
    parser.add_argument("--num_flips", type=int, default=None,
                        help="Override number of label flips.")
    parser.add_argument("--output", default=None,
                        help="Override output .pt path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    attack_cfg = cfg["attack"]
    dataset_cfg = cfg["dataset"]

    attack_type = args.attack or attack_cfg.get("type", "flip")
    num_flips = args.num_flips or int(attack_cfg.get("num_flips", 1000))

    cifar_root = Path(dataset_cfg["data_dir"])
    cifar_root.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_cfg.get("name", "cifar10")

    cifar_cls = (
        torchvision.datasets.CIFAR10
        if dataset_name == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    cifar = cifar_cls(root=str(cifar_root), train=True, download=True)
    original_labels = np.asarray(cifar.targets, dtype=np.int64)

    if attack_type == "flip":
        LOGGER.info("Loading FLIP soft labels and selecting top-%d flips", num_flips)
        poison_indices, poison_labels = load_flip_labels(
            attack_cfg["flip_labels_path"],
            num_flips=num_flips,
            cifar_root=cifar_root,
            dataset_name=dataset_name,
        )
    elif attack_type == "random":
        LOGGER.info(
            "Generating random %d -> %d flips (n=%d)",
            attack_cfg.get("source_class", 9),
            attack_cfg.get("target_class", 4),
            num_flips,
        )
        poison_indices, poison_labels = generate_random_flip(
            original_labels,
            num_flips=num_flips,
            source_class=attack_cfg.get("source_class", 9),
            target_class=attack_cfg.get("target_class", 4),
            seed=cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    output = args.output or (
        Path(attack_cfg["poisoned_data_dir"])
        / f"{attack_type}_{dataset_name}_{num_flips}.pt"
    )
    save_poisoned_dataset(
        poison_indices=poison_indices,
        poison_labels=poison_labels,
        original_labels=original_labels,
        save_path=output,
        extra={
            "attack_type": attack_type,
            "num_flips": int(num_flips),
            "dataset_name": dataset_name,
            "trigger": attack_cfg.get("trigger", "sinusoidal"),
            "source_class": int(attack_cfg.get("source_class", -1)),
            "target_class": int(attack_cfg.get("target_class", -1)),
        },
    )
    LOGGER.info("Saved poisoned dataset to %s", output)


if __name__ == "__main__":
    main()

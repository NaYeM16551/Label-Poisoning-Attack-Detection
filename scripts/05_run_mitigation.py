#!/usr/bin/env python3
"""Mitigation experiments: remove or downweight, then retrain.

Usage:
    python scripts/05_run_mitigation.py --config configs/default.yaml \
        --poisoned data/poisoned/flip_cifar10_1000.pt \
        --scores  results/detection/flip_cifar10_1000_scores.npz \
        --score_key ssl_knn_k20 \
        --mode remove

Modes:
    remove     -> remove top removal_fraction suspicious samples then train
    downweight -> train with sample weights derived from suspicion scores
    none       -> baseline: train on the full poisoned dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.attack_metrics import measure_cta_pta  # noqa: E402
from src.mitigation.downweight_retrain import downweight_and_retrain  # noqa: E402
from src.mitigation.remove_and_retrain import remove_and_retrain  # noqa: E402
from src.mitigation.trainer import train_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.data import (  # noqa: E402
    PoisonedCIFAR,
    load_poisoned_dataset,
)
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOGGER = get_logger("run_mitigation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--poisoned", required=True)
    parser.add_argument("--scores", default=None,
                        help=".npz file produced by 04_run_detection.py "
                             "(required for remove/downweight modes).")
    parser.add_argument("--score_key", default="ssl_knn_k20",
                        help="Which score array inside the .npz to use.")
    parser.add_argument("--mode", choices=["remove", "downweight", "none"], default="remove")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _load_scores(path: Path, key: str) -> np.ndarray:
    payload = np.load(path)
    if key not in payload.files:
        raise KeyError(f"Score key '{key}' not in {path}. Available: {payload.files}")
    return payload[key]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda")

    # Build poisoned training set with augmentation
    poison_indices, poison_labels, _ = load_poisoned_dataset(args.poisoned)
    train_transform = PoisonedCIFAR.get_train_transform(cfg["dataset"]["name"])
    train_set = PoisonedCIFAR(
        root=cfg["dataset"]["data_dir"],
        train=True,
        transform=train_transform,
        poison_indices=poison_indices,
        poison_labels=poison_labels,
        dataset_name=cfg["dataset"]["name"],
    )

    eval_transform = PoisonedCIFAR.get_eval_transform(cfg["dataset"]["name"])
    test_set = PoisonedCIFAR(
        root=cfg["dataset"]["data_dir"],
        train=False,
        transform=eval_transform,
        dataset_name=cfg["dataset"]["name"],
    )

    train_kwargs = dict(
        num_classes=cfg["dataset"]["num_classes"],
        model_name=cfg["training"]["model"],
        epochs=args.epochs or cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        nesterov=cfg["training"].get("nesterov", True),
        milestones=cfg["training"]["lr_schedule"]["milestones"],
        gamma=cfg["training"]["lr_schedule"]["gamma"],
        device=device,
        num_workers=cfg.get("num_workers", 4),
    )

    if args.mode == "none":
        LOGGER.info("Mode=none: training baseline classifier on poisoned data")
        result = train_model(train_set, **train_kwargs)
    else:
        if args.scores is None:
            raise ValueError("--scores is required for remove/downweight modes")
        scores = _load_scores(Path(args.scores), args.score_key)
        if args.mode == "remove":
            removal_fraction = cfg.get("mitigation", {}).get("removal_fraction", 0.02)
            LOGGER.info(
                "Mode=remove: removing top %.2f%% of samples by '%s'",
                100 * removal_fraction, args.score_key,
            )
            result = remove_and_retrain(
                train_set,
                scores,
                removal_fraction=removal_fraction,
                train_kwargs=train_kwargs,
            )
        else:
            temperature = cfg.get("mitigation", {}).get("downweight_temperature", 1.0)
            LOGGER.info(
                "Mode=downweight: weights from '%s' (temperature=%.2f)",
                args.score_key, temperature,
            )
            result = downweight_and_retrain(
                train_set, scores,
                temperature=temperature,
                train_kwargs=train_kwargs,
            )

    model = result["model"]

    metrics = measure_cta_pta(
        model,
        test_set,
        trigger_type=cfg["attack"]["trigger"],
        target_class=cfg["attack"]["target_class"],
        device=device,
        batch_size=cfg["training"]["batch_size"],
        dataset_name=cfg["dataset"]["name"],
    )
    LOGGER.info("CTA=%.4f  PTA=%.4f", metrics["cta"], metrics["pta"])

    # Persist outputs
    out_dir = Path(cfg["output_dir"]) / "mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.poisoned).stem}_{args.mode}_{args.score_key}"
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    with (out_dir / f"{tag}.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "score_key": args.score_key, **metrics}, f, indent=2)
    LOGGER.info("Saved model + metrics under %s", out_dir)


if __name__ == "__main__":
    main()

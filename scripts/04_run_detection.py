#!/usr/bin/env python3
"""Main detection experiment script.

Usage:
    python scripts/04_run_detection.py --config configs/default.yaml \
        --poisoned data/poisoned/flip_cifar10_1000.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.knn_detector import KNNDetector  # noqa: E402
from src.detectors.random_detector import random_detection  # noqa: E402
from src.evaluation.detection_metrics import (  # noqa: E402
    compute_detection_metrics,
    print_detection_report,
)
from src.evaluation.per_class_analysis import per_class_detection_breakdown  # noqa: E402
from src.features.ssl_extractor import SSLFeatureExtractor  # noqa: E402
from src.features.validate_features import run_all_checks  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.data import (  # noqa: E402
    CIFAR10_CLASSES,
    PoisonedCIFAR,
    load_poisoned_dataset,
)
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.visualization.roc_pr_curves import plot_pr, plot_roc  # noqa: E402
from src.visualization.score_histogram import plot_score_histogram  # noqa: E402

LOGGER = get_logger("run_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--poisoned", default=None,
                        help="Path to a poisoned dataset .pt file. "
                             "If omitted, derived from config.")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda")
    output_dir = Path(cfg["output_dir"])

    # ------------------------------------------------------------------
    # Load poisoned dataset
    # ------------------------------------------------------------------
    if args.poisoned:
        poison_path = Path(args.poisoned)
    else:
        poison_path = (
            Path(cfg["attack"]["poisoned_data_dir"])
            / f"{cfg['attack'].get('type', 'flip')}_{cfg['dataset']['name']}_"
              f"{cfg['attack']['num_flips']}.pt"
        )
    LOGGER.info("Loading poisoned dataset from %s", poison_path)
    poison_indices, poison_labels, _ = load_poisoned_dataset(poison_path)

    dataset = PoisonedCIFAR(
        root=cfg["dataset"]["data_dir"],
        train=True,
        poison_indices=poison_indices,
        poison_labels=poison_labels,
        dataset_name=cfg["dataset"]["name"],
    )
    gt = dataset.get_ground_truth()
    LOGGER.info("Poisoned %d / %d samples", gt["num_poisoned"], gt["total_samples"])

    # ------------------------------------------------------------------
    # Extract / load SSL features
    # ------------------------------------------------------------------
    encoder_name = cfg["features"]["encoder"]
    cache_path = (
        Path(cfg["features"]["cache_dir"])
        / f"{cfg['dataset']['name']}_{encoder_name}_train.npy"
    )

    if cache_path.exists():
        LOGGER.info("Loading cached features from %s", cache_path)
        ssl_features = np.load(cache_path)
    else:
        LOGGER.info("Extracting SSL features (no cache found)")
        extractor = SSLFeatureExtractor(encoder_name, device=device)
        cifar_cls = (
            torchvision.datasets.CIFAR10
            if cfg["dataset"]["name"] == "cifar10"
            else torchvision.datasets.CIFAR100
        )
        raw = cifar_cls(root=cfg["dataset"]["data_dir"], train=True, download=True)
        ssl_features = extractor.extract_features(
            raw,
            batch_size=cfg["features"]["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            normalize=cfg["features"]["normalize"],
            cache_path=cache_path,
        )

    LOGGER.info("Features shape: %s", ssl_features.shape)

    # ------------------------------------------------------------------
    # Optional feature validation
    # ------------------------------------------------------------------
    if not args.skip_validation:
        class_names = (
            list(CIFAR10_CLASSES) if cfg["dataset"]["name"] == "cifar10" else None
        )
        run_all_checks(
            ssl_features,
            gt["original_labels"],
            save_dir=output_dir / "validation",
            class_names=class_names,
        )

    # ------------------------------------------------------------------
    # Run detectors
    # ------------------------------------------------------------------
    all_results: dict[str, dict[str, float]] = {}
    score_sets: dict[str, np.ndarray] = {}

    for k in cfg["detector"]["k_values"]:
        name = f"ssl_knn_k{k}"
        LOGGER.info("Running SSL k-NN (k=%d)", k)
        detector = KNNDetector(k=k, distance_metric=cfg["detector"]["distance_metric"])
        result = detector.detect(ssl_features, gt["current_labels"])
        metrics = compute_detection_metrics(result.scores, gt["is_poisoned"], gt["num_poisoned"])
        print_detection_report(metrics, name)
        all_results[name] = metrics
        score_sets[name] = result.scores

    # Weighted variant at default k
    LOGGER.info("Running weighted SSL k-NN")
    detector = KNNDetector(
        k=cfg["detector"]["default_k"],
        distance_metric=cfg["detector"]["distance_metric"],
    )
    result_w = detector.detect_weighted(ssl_features, gt["current_labels"])
    metrics_w = compute_detection_metrics(result_w.scores, gt["is_poisoned"], gt["num_poisoned"])
    print_detection_report(metrics_w, "ssl_knn_weighted")
    all_results["ssl_knn_weighted"] = metrics_w
    score_sets["ssl_knn_weighted"] = result_w.scores

    # Random baseline
    rand_scores = random_detection(gt["total_samples"], seed=cfg.get("seed", 42))
    rand_metrics = compute_detection_metrics(rand_scores, gt["is_poisoned"], gt["num_poisoned"])
    print_detection_report(rand_metrics, "random")
    all_results["random"] = rand_metrics
    score_sets["random"] = rand_scores

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    detection_dir = output_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)
    json_path = detection_dir / f"{poison_path.stem}_metrics.json"
    serialisable = {
        name: {k: float(v) for k, v in metrics.items()}
        for name, metrics in all_results.items()
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    LOGGER.info("Saved metrics JSON to %s", json_path)

    np.savez(
        detection_dir / f"{poison_path.stem}_scores.npz",
        is_poisoned=gt["is_poisoned"],
        original_labels=gt["original_labels"],
        current_labels=gt["current_labels"],
        **{name: scores for name, scores in score_sets.items()},
    )

    # Per-class breakdown for the default k
    breakdown = per_class_detection_breakdown(
        score_sets[f"ssl_knn_k{cfg['detector']['default_k']}"],
        gt["is_poisoned"],
        gt["current_labels"],
        gt["original_labels"],
        top_k=gt["num_poisoned"],
    )
    breakdown.to_csv(detection_dir / f"{poison_path.stem}_per_class.csv", index=False)
    LOGGER.info("Per-class breakdown:\n%s", breakdown.to_string(index=False))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        figures_dir = output_dir / "figures" / poison_path.stem
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_score_histogram(
            score_sets[f"ssl_knn_k{cfg['detector']['default_k']}"],
            gt["is_poisoned"],
            save_path=figures_dir / "score_histogram.png",
        )
        plot_roc(score_sets, gt["is_poisoned"], save_path=figures_dir / "roc.png")
        plot_pr(score_sets, gt["is_poisoned"], save_path=figures_dir / "pr.png")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Detection Performance Comparison")
    print("=" * 70)
    print(f"{'Method':<25} {'AUROC':>8} {'AUPRC':>8} {'Prec@k':>8}")
    print("-" * 50)
    for name, metrics in all_results.items():
        print(
            f"{name:<25} {metrics['auroc']:>8.4f} "
            f"{metrics['auprc']:>8.4f} "
            f"{metrics['precision_at_k']:>8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()

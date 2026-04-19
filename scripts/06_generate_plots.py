#!/usr/bin/env python3
"""Aggregate detection results across runs into ablation plots.

Reads every ``*_metrics.json`` and ``*_scores.npz`` under
``results/detection`` and produces:
    * k-ablation curve (AUROC vs k)
    * poisoning-rate ablation (AUROC vs %)
    * combined ROC / PR curves
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.visualization.ablation_plots import (  # noqa: E402
    plot_k_ablation,
    plot_poisoning_rate_ablation,
)
from src.visualization.roc_pr_curves import plot_pr, plot_roc  # noqa: E402

LOGGER = get_logger("generate_plots")

K_PATTERN = re.compile(r"ssl_knn_k(\d+)$")
RATE_PATTERN = re.compile(r"_(\d+)$")  # last "_<num>" in stem == num_flips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    detection_dir = Path(cfg["output_dir"]) / "detection"
    figures_dir = Path(cfg["output_dir"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(detection_dir.glob("*_metrics.json"))
    if not metric_files:
        LOGGER.warning("No metrics JSON found under %s", detection_dir)
        return

    # ------------------------------------------------------------------
    # k-ablation: take the latest run and plot AUROC vs k for SSL k-NN.
    # ------------------------------------------------------------------
    latest_metrics = metric_files[-1]
    LOGGER.info("Using %s for k-ablation", latest_metrics.name)
    with latest_metrics.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    k_metric_per_method: Dict[str, List[float]] = defaultdict(list)
    ks: List[int] = []
    for name, vals in metrics.items():
        m = K_PATTERN.match(name)
        if not m:
            continue
        ks.append(int(m.group(1)))
        k_metric_per_method["SSL k-NN AUROC"].append(vals["auroc"])
        k_metric_per_method["SSL k-NN AUPRC"].append(vals["auprc"])
    if ks:
        order = np.argsort(ks)
        ks_sorted = [ks[i] for i in order]
        for key in list(k_metric_per_method.keys()):
            k_metric_per_method[key] = [k_metric_per_method[key][i] for i in order]
        plot_k_ablation(
            ks_sorted,
            {"AUROC": k_metric_per_method["SSL k-NN AUROC"]},
            save_path=figures_dir / f"{latest_metrics.stem}_k_ablation_auroc.png",
            metric_name="AUROC",
        )
        plot_k_ablation(
            ks_sorted,
            {"AUPRC": k_metric_per_method["SSL k-NN AUPRC"]},
            save_path=figures_dir / f"{latest_metrics.stem}_k_ablation_auprc.png",
            metric_name="AUPRC",
        )

    # ------------------------------------------------------------------
    # Poisoning-rate ablation: parse num_flips from filenames.
    # ------------------------------------------------------------------
    rates_per_method: Dict[str, Dict[int, float]] = defaultdict(dict)
    for f in metric_files:
        stem = f.stem.replace("_metrics", "")
        m = RATE_PATTERN.search(stem)
        if not m:
            continue
        num_flips = int(m.group(1))
        with f.open("r", encoding="utf-8") as fh:
            mvals = json.load(fh)
        for method_name, vals in mvals.items():
            rates_per_method[method_name][num_flips] = vals["auroc"]

    if rates_per_method:
        # Total CIFAR-10 train size used to convert num_flips -> rate.
        total_train = 50000
        flips = sorted({n for d in rates_per_method.values() for n in d.keys()})
        rate_axis = [n / total_train for n in flips]
        plottable = {}
        for method_name, by_n in rates_per_method.items():
            if not all(n in by_n for n in flips):
                continue  # incomplete sweep
            plottable[method_name] = [by_n[n] for n in flips]
        if plottable:
            plot_poisoning_rate_ablation(
                rate_axis,
                plottable,
                save_path=figures_dir / "poisoning_rate_ablation_auroc.png",
                metric_name="AUROC",
            )

    # ------------------------------------------------------------------
    # Combined ROC / PR for the latest run.
    # ------------------------------------------------------------------
    score_files = sorted(detection_dir.glob("*_scores.npz"))
    if score_files:
        latest = score_files[-1]
        payload = np.load(latest)
        is_poisoned = payload["is_poisoned"]
        score_sets = {
            name: payload[name]
            for name in payload.files
            if name not in {"is_poisoned", "original_labels", "current_labels"}
        }
        plot_roc(score_sets, is_poisoned,
                 save_path=figures_dir / f"{latest.stem}_roc.png",
                 title=f"ROC — {latest.stem}")
        plot_pr(score_sets, is_poisoned,
                save_path=figures_dir / f"{latest.stem}_pr.png",
                title=f"PR — {latest.stem}")

    LOGGER.info("Done. Figures under %s", figures_dir)


if __name__ == "__main__":
    main()

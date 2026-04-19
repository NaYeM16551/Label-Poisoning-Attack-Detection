"""Ablation plots: detection AUROC vs k, vs poisoning rate, etc."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def plot_k_ablation(
    ks: Sequence[int],
    metric_per_k: Mapping[str, Sequence[float]],
    save_path: Optional[PathLike] = None,
    metric_name: str = "AUROC",
    title: Optional[str] = None,
) -> Path:
    """One curve per method, x-axis = k, y-axis = metric."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for method, values in metric_per_k.items():
        plt.plot(list(ks), list(values), marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel(metric_name)
    plt.title(title or f"Detection {metric_name} vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("k_ablation.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_k_ablation] Saved {save_path}")
    return save_path


def plot_poisoning_rate_ablation(
    rates: Sequence[float],
    metric_per_method: Mapping[str, Sequence[float]],
    save_path: Optional[PathLike] = None,
    metric_name: str = "AUROC",
    title: Optional[str] = None,
) -> Path:
    """One curve per method, x-axis = poisoning rate."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    rates_pct = [100 * r for r in rates]
    for method, values in metric_per_method.items():
        plt.plot(rates_pct, list(values), marker="o", label=method)
    plt.xlabel("Poisoning rate (%)")
    plt.ylabel(metric_name)
    plt.title(title or f"Detection {metric_name} vs poisoning rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("poisoning_rate_ablation.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_poisoning_rate_ablation] Saved {save_path}")
    return save_path

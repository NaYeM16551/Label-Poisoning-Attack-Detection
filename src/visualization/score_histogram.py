"""Histograms of suspicion scores split by ground truth."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


def plot_score_histogram(
    scores: np.ndarray,
    is_poisoned: np.ndarray,
    save_path: Optional[PathLike] = None,
    title: str = "Detector score distribution",
    bins: int = 50,
) -> Path:
    """Overlay clean vs. poisoned score histograms."""
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=np.float32)
    is_poisoned = np.asarray(is_poisoned).astype(bool)

    plt.figure(figsize=(8, 5))
    plt.hist(
        scores[~is_poisoned],
        bins=bins, alpha=0.6, label=f"clean (n={(~is_poisoned).sum()})",
        color="steelblue", density=True,
    )
    if is_poisoned.any():
        plt.hist(
            scores[is_poisoned],
            bins=bins, alpha=0.7, label=f"poisoned (n={is_poisoned.sum()})",
            color="crimson", density=True,
        )
    plt.xlabel("Suspicion score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("score_histogram.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_score_histogram] Saved {save_path}")
    return save_path

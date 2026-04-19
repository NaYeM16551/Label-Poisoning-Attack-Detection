"""ROC and Precision-Recall curve plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PathLike = Union[str, Path]
ScoreSet = Mapping[str, np.ndarray]


def plot_roc(
    score_sets: ScoreSet,
    is_poisoned: np.ndarray,
    save_path: Optional[PathLike] = None,
    title: str = "ROC curves",
) -> Path:
    """Plot ROC curves for several detectors on the same axes."""
    import matplotlib.pyplot as plt

    is_poisoned = np.asarray(is_poisoned).astype(int)
    plt.figure(figsize=(7, 6))
    for name, scores in score_sets.items():
        fpr, tpr, _ = roc_curve(is_poisoned, scores)
        try:
            auc = float(roc_auc_score(is_poisoned, scores))
        except ValueError:
            auc = 0.0
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("roc.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_roc] Saved {save_path}")
    return save_path


def plot_pr(
    score_sets: ScoreSet,
    is_poisoned: np.ndarray,
    save_path: Optional[PathLike] = None,
    title: str = "Precision-recall curves",
) -> Path:
    """Plot PR curves for several detectors on the same axes."""
    import matplotlib.pyplot as plt

    is_poisoned = np.asarray(is_poisoned).astype(int)
    plt.figure(figsize=(7, 6))
    for name, scores in score_sets.items():
        precision, recall, _ = precision_recall_curve(is_poisoned, scores)
        try:
            ap = float(average_precision_score(is_poisoned, scores))
        except ValueError:
            ap = 0.0
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("pr.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_pr] Saved {save_path}")
    return save_path

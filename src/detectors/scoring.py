"""Score normalisation and threshold calibration helpers."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Linear rescale to [0, 1]; returns zeros if input is constant."""
    scores = np.asarray(scores, dtype=np.float32)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores into per-rank quantiles in [0, 1]."""
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    if len(scores) <= 1:
        return np.zeros_like(scores, dtype=np.float32)
    return ranks.astype(np.float32) / (len(scores) - 1)


def calibrate_threshold(
    scores: np.ndarray,
    is_poisoned: Optional[np.ndarray] = None,
    target_fpr: float = 0.05,
) -> Tuple[float, float]:
    """Pick a threshold either from clean-only quantile or ground truth.

    Returns ``(threshold, achieved_metric)``. If ``is_poisoned`` is provided,
    we tune the threshold on the ROC curve to hit ``target_fpr``; otherwise
    we use the ``(1 - target_fpr)`` quantile of scores assuming most data is
    clean (a common practical assumption).
    """
    scores = np.asarray(scores, dtype=np.float32)
    if is_poisoned is None:
        threshold = float(np.quantile(scores, 1.0 - target_fpr))
        return threshold, target_fpr

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(is_poisoned, scores)
    idx = int(np.searchsorted(fpr, target_fpr))
    idx = min(idx, len(thresholds) - 1)
    return float(thresholds[idx]), float(tpr[idx])

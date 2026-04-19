"""Standard ranking metrics for poisoning detection."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_detection_metrics(
    scores: np.ndarray,
    is_poisoned: np.ndarray,
    num_poisoned: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the metric set described in the blueprint.

    Args:
        scores: ``(N,)`` suspicion scores (higher == more suspicious).
        is_poisoned: ``(N,)`` 0/1 ground truth (1 == poisoned).
        num_poisoned: optional override for k in Precision@k / Recall@k.
    """
    scores = np.asarray(scores, dtype=np.float32)
    is_poisoned = np.asarray(is_poisoned, dtype=np.int32)
    if scores.shape != is_poisoned.shape:
        raise ValueError("scores and is_poisoned must have identical shape")

    if num_poisoned is None:
        num_poisoned = int(is_poisoned.sum())
    num_poisoned = max(num_poisoned, 1)

    results: Dict[str, float] = {}

    try:
        results["auroc"] = float(roc_auc_score(is_poisoned, scores))
    except ValueError:
        results["auroc"] = 0.0
    try:
        results["auprc"] = float(average_precision_score(is_poisoned, scores))
    except ValueError:
        results["auprc"] = 0.0

    top_k = np.argsort(scores)[-num_poisoned:]
    tp_at_k = int(is_poisoned[top_k].sum())
    results["precision_at_k"] = tp_at_k / num_poisoned
    results["recall_at_k"] = tp_at_k / num_poisoned
    results["num_poisoned"] = float(num_poisoned)

    if is_poisoned.sum() > 0 and is_poisoned.sum() < len(is_poisoned):
        precision, recall, _ = precision_recall_curve(is_poisoned, scores)
        # PR curve from sklearn comes sorted by decreasing threshold;
        # to query by recall we reverse it.
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]
        for target_recall in (0.5, 0.8, 0.9, 0.95):
            idx = int(np.searchsorted(recall_rev, target_recall))
            if idx < len(precision_rev):
                results[f"precision_at_recall_{target_recall}"] = float(
                    precision_rev[idx]
                )
            else:
                results[f"precision_at_recall_{target_recall}"] = 0.0

        fpr, tpr, _ = roc_curve(is_poisoned, scores)
        for target_fpr in (0.01, 0.05, 0.10):
            idx = int(np.searchsorted(fpr, target_fpr))
            idx = min(idx, len(tpr) - 1)
            results[f"tpr_at_fpr_{target_fpr}"] = float(tpr[idx])
    return results


def print_detection_report(metrics: Dict[str, float], method_name: str = "") -> None:
    """Pretty-print a detection metrics dict."""
    bar = "=" * 50
    print(f"\n{bar}")
    print(f"Detection Report: {method_name}")
    print(bar)
    print(f"AUROC:           {metrics.get('auroc', 0.0):.4f}")
    print(f"AUPRC:           {metrics.get('auprc', 0.0):.4f}")
    print(f"Precision@k:     {metrics.get('precision_at_k', 0.0):.4f}")
    print(f"Recall@k:        {metrics.get('recall_at_k', 0.0):.4f}")
    print(f"TPR@FPR=1%:      {metrics.get('tpr_at_fpr_0.01', 0.0):.4f}")
    print(f"TPR@FPR=5%:      {metrics.get('tpr_at_fpr_0.05', 0.0):.4f}")
    print(bar)

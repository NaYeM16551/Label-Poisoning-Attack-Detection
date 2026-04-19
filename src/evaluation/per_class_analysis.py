"""Per-class detection breakdown — useful for diagnosing failure modes."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def per_class_detection_breakdown(
    scores: np.ndarray,
    is_poisoned: np.ndarray,
    current_labels: np.ndarray,
    original_labels: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate detection performance per (original_class, current_class).

    For each (source -> target) pair we report:
        * num_samples in that group
        * num_poisoned in that group
        * num_detected (in top-k or scores >= median threshold)
        * recall = num_detected / num_poisoned
        * mean_score across the group
    """
    scores = np.asarray(scores, dtype=np.float32)
    is_poisoned = np.asarray(is_poisoned, dtype=np.int32)
    current_labels = np.asarray(current_labels, dtype=np.int64)
    if original_labels is None:
        original_labels = current_labels.copy()
    else:
        original_labels = np.asarray(original_labels, dtype=np.int64)

    if top_k is None:
        top_k = int(is_poisoned.sum())
    detected_mask = np.zeros_like(scores, dtype=bool)
    if top_k > 0:
        top_indices = np.argsort(scores)[-top_k:]
        detected_mask[top_indices] = True

    rows = []
    pairs = sorted(set(zip(original_labels.tolist(), current_labels.tolist())))
    for src, tgt in pairs:
        mask = (original_labels == src) & (current_labels == tgt)
        if not mask.any():
            continue
        n_samples = int(mask.sum())
        n_poison = int(is_poisoned[mask].sum())
        n_detected = int((detected_mask[mask] & is_poisoned[mask].astype(bool)).sum())
        rows.append({
            "source_class": int(src),
            "current_class": int(tgt),
            "is_poison_pair": int(src != tgt),
            "num_samples": n_samples,
            "num_poisoned": n_poison,
            "num_detected": n_detected,
            "recall": n_detected / n_poison if n_poison else 0.0,
            "mean_score": float(scores[mask].mean()),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["is_poison_pair", "num_poisoned"], ascending=[False, False]
        ).reset_index(drop=True)
    return df

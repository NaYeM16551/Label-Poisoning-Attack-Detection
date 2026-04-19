"""Hard removal + retrain pipeline."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset, Subset

from .trainer import train_model


def create_cleaned_dataset(
    dataset: Dataset,
    scores: np.ndarray,
    removal_fraction: float = 0.02,
) -> Subset:
    """Drop the top ``removal_fraction`` most suspicious samples."""
    scores = np.asarray(scores, dtype=np.float32)
    n_total = len(dataset)
    if n_total != len(scores):
        raise ValueError(
            f"dataset size {n_total} != scores size {len(scores)}"
        )

    n_remove = max(1, int(n_total * removal_fraction))
    suspicious = np.argsort(scores)[-n_remove:]
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[suspicious] = False
    keep_indices = np.where(keep_mask)[0]
    print(
        f"[remove_and_retrain] Removed {n_remove} suspicious samples "
        f"({100 * n_remove / n_total:.2f}%); kept {len(keep_indices)}."
    )
    return Subset(dataset, keep_indices.tolist())


def remove_and_retrain(
    dataset: Dataset,
    scores: np.ndarray,
    *,
    removal_fraction: float = 0.02,
    train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end hard-removal mitigation."""
    cleaned = create_cleaned_dataset(dataset, scores, removal_fraction=removal_fraction)
    train_kwargs = dict(train_kwargs or {})
    return train_model(cleaned, **train_kwargs)

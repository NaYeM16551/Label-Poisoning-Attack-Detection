"""Soft-downweighting + retrain pipeline.

Each sample's training weight decreases monotonically with its
suspicion score, so suspected poisons stay in the dataset but
contribute less to the loss.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from .trainer import train_model


def scores_to_weights(
    scores: np.ndarray,
    temperature: float = 1.0,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Convert suspicion scores into [min_weight, 1] sample weights."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    norm = (scores - lo) / (hi - lo)
    weights = np.exp(-norm / max(temperature, 1e-6))
    weights = np.clip(weights, min_weight, 1.0)
    return weights.astype(np.float32)


def downweight_and_retrain(
    dataset: Dataset,
    scores: np.ndarray,
    *,
    temperature: float = 1.0,
    train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end soft-downweighting mitigation."""
    weights = scores_to_weights(scores, temperature=temperature)
    print(
        f"[downweight_and_retrain] Sample weights: "
        f"min={weights.min():.4f}, mean={weights.mean():.4f}, max={weights.max():.4f}"
    )
    train_kwargs = dict(train_kwargs or {})
    train_kwargs["sample_weights"] = weights
    return train_model(dataset, **train_kwargs)

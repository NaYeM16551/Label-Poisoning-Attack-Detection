"""Random label-flipping baseline."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_random_flip(
    original_labels: np.ndarray,
    num_flips: int,
    source_class: int = 9,
    target_class: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly select ``num_flips`` ``source_class`` samples and re-label them.

    Returns ``(indices, new_labels)`` ready for :class:`PoisonedCIFAR`.
    """
    rng = np.random.default_rng(seed)
    source_indices = np.where(original_labels == source_class)[0]
    if num_flips > len(source_indices):
        raise ValueError(
            f"Requested {num_flips} flips but only {len(source_indices)} "
            f"samples belong to source_class={source_class}"
        )
    selected = rng.choice(source_indices, size=num_flips, replace=False)
    new_labels = np.full(num_flips, target_class, dtype=np.int64)
    return selected.astype(np.int64), new_labels

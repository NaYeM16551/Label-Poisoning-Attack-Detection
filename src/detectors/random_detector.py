"""Random-score baseline (the floor every other detector must beat)."""
from __future__ import annotations

import numpy as np


def random_detection(n_samples: int, seed: int = 42) -> np.ndarray:
    """Return uniform-random suspicion scores in [0, 1)."""
    rng = np.random.default_rng(seed)
    return rng.random(n_samples).astype(np.float32)

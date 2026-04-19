"""Wrapper around the official FLIP codebase.

Setup steps (one-time):

    git clone https://github.com/SewoongLab/FLIP external/FLIP
    # follow their README to install dependencies and download
    # precomputed soft labels (or generate your own)

This module is intentionally read-only with respect to the FLIP repo: we
just consume the soft-label tensor it produces and convert it into the
``(poison_indices, poison_labels)`` pair our :class:`PoisonedCIFAR`
expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torchvision

PathLike = Union[str, Path]


def score_flip_candidates(
    soft_labels: np.ndarray,
    original_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FLIP's per-sample attack score and the best target class.

    The FLIP paper (Algorithm 1, step 3) ranks training samples by

        score(i) = max_{c != y_i} logit_c(x_i) - logit_{y_i}(x_i)

    A higher score means flipping ``y_i`` to ``argmax_{c != y_i}`` is most
    impactful for the trajectory-matching objective.
    """
    if soft_labels.shape[0] != original_labels.shape[0]:
        raise ValueError(
            "soft_labels and original_labels must have matching first dim "
            f"(got {soft_labels.shape[0]} vs {original_labels.shape[0]})"
        )

    n, num_classes = soft_labels.shape
    correct_logits = soft_labels[np.arange(n), original_labels]

    masked = soft_labels.copy()
    masked[np.arange(n), original_labels] = -np.inf
    flip_targets = np.argmax(masked, axis=1).astype(np.int64)
    best_incorrect = masked[np.arange(n), flip_targets]
    scores = best_incorrect - correct_logits
    return scores, flip_targets


def load_flip_labels(
    flip_labels_path: PathLike,
    num_flips: int,
    cifar_root: PathLike = "./data/raw",
    train: bool = True,
    dataset_name: str = "cifar10",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load FLIP soft labels and select the top-``num_flips`` samples to flip.

    Args:
        flip_labels_path: path to the FLIP repo's soft-label tensor
            (shape ``(N, num_classes)``).
        num_flips: number of labels to corrupt (e.g. 1000 for 2% of CIFAR-10).
        cifar_root: where CIFAR-10/100 lives (used only to read original labels).
        train: load the train split (the only split FLIP attacks).
        dataset_name: ``cifar10`` or ``cifar100``.

    Returns:
        Tuple ``(poison_indices, poison_labels)`` of dtype int64.
    """
    flip_labels_path = Path(flip_labels_path)
    if not flip_labels_path.exists():
        raise FileNotFoundError(
            f"FLIP soft labels not found at {flip_labels_path}. "
            "See https://github.com/SewoongLab/FLIP for the precomputed assets."
        )

    raw = torch.load(str(flip_labels_path), weights_only=False)
    if isinstance(raw, torch.Tensor):
        soft_labels = raw.detach().cpu().numpy()
    else:
        soft_labels = np.asarray(raw)

    if soft_labels.ndim != 2:
        raise ValueError(
            f"Expected 2-D soft-label tensor, got shape {soft_labels.shape}"
        )

    if dataset_name.lower() == "cifar10":
        cifar_cls = torchvision.datasets.CIFAR10
    elif dataset_name.lower() == "cifar100":
        cifar_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    cifar = cifar_cls(root=str(cifar_root), train=train, download=True)
    original_labels = np.asarray(cifar.targets, dtype=np.int64)

    if soft_labels.shape[0] != len(original_labels):
        raise ValueError(
            "Soft-label rows do not match the dataset size "
            f"({soft_labels.shape[0]} vs {len(original_labels)})"
        )

    if num_flips <= 0 or num_flips > len(original_labels):
        raise ValueError(
            f"num_flips={num_flips} is outside (0, {len(original_labels)}]"
        )

    scores, flip_targets = score_flip_candidates(soft_labels, original_labels)
    top_indices = np.argsort(scores)[-num_flips:]
    poison_indices = top_indices.astype(np.int64)
    poison_labels = flip_targets[top_indices].astype(np.int64)
    return poison_indices, poison_labels

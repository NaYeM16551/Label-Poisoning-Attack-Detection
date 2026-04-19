"""Loss-based baseline: high training loss == suspicious."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@torch.no_grad()
def compute_sample_losses(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 4,
) -> np.ndarray:
    """Per-sample cross-entropy loss for every item in ``dataset``."""
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    losses = []
    for batch in tqdm(loader, desc="Computing per-sample losses"):
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        outputs = model(images)
        per_sample = criterion(outputs, labels)
        losses.append(per_sample.detach().cpu().numpy())
    return np.concatenate(losses, axis=0).astype(np.float32)


def loss_based_detection(losses: np.ndarray) -> np.ndarray:
    """Min-max normalise losses into [0, 1] suspicion scores."""
    losses = np.asarray(losses, dtype=np.float32)
    lo, hi = losses.min(), losses.max()
    if hi - lo < 1e-12:
        return np.zeros_like(losses)
    return (losses - lo) / (hi - lo)

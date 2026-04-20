"""CTA / PTA measurement and trigger application.

The trigger definitions match FLIP's appendix B.1 — keep them in sync
with the FLIP repo if you regenerate poisoned datasets.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)


def _dataset_stats(dataset_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    name = dataset_name.lower()
    if name == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if name == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def apply_trigger(images: torch.Tensor, trigger_type: str = "sinusoidal") -> torch.Tensor:
    """Stamp a FLIP trigger onto a batch of images in the raw [0, 1] pixel domain.

    Callers with per-channel-normalised tensors must de-normalise before
    calling this function and re-normalise the result; see ``measure_cta_pta``.
    """
    triggered = images.clone()
    _, c, h, w = triggered.shape

    if trigger_type == "sinusoidal":
        # Matches FLIP's StripePoisoner defaults (modules/base_utils/datasets.py):
        #   strength=6 in [0,255] -> 6/255 in [0,1]
        #   freq=16, horizontal=False -> sine varies along the HEIGHT axis
        #   argument = np.linspace(0, freq * pi, h)
        amplitude = 6.0 / 255.0
        freq = 16.0
        rows = torch.linspace(
            0.0, freq * float(np.pi), steps=h,
            device=triggered.device, dtype=triggered.dtype,
        )
        noise = amplitude * torch.sin(rows)
        triggered = triggered + noise.view(1, 1, h, 1)
        triggered = triggered.clamp(0.0, 1.0)

    elif trigger_type == "pixel":
        locations = [(11, 16), (5, 27), (30, 7)]
        colors = [
            (0.396, 0.0, 0.098),
            (0.396, 0.482, 0.475),
            (0.0, 0.141, 0.212),
        ]
        for (r, col), color in zip(locations, colors):
            if r >= h or col >= w:
                continue
            for ch in range(min(c, 3)):
                triggered[:, ch, r, col] = color[ch]

    elif trigger_type == "turner":
        patch = torch.tensor(
            [
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            ],
            dtype=triggered.dtype,
            device=triggered.device,
        )
        for r0, c0 in ((0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)):
            if r0 < 0 or c0 < 0:
                continue
            triggered[:, :3, r0:r0 + 3, c0:c0 + 3] = patch

    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    return triggered


@torch.no_grad()
def measure_cta_pta(
    model: nn.Module,
    test_dataset: Dataset,
    trigger_type: str = "sinusoidal",
    target_class: int = 4,
    device: str = "cuda",
    batch_size: int = 256,
    dataset_name: str = "cifar10",
) -> Dict[str, float]:
    """Compute Clean Test Accuracy and Poison Test Accuracy.

    PTA is measured only on test samples whose true class is *not* the
    target class (per the FLIP paper).

    The loader yields tensors that have already been normalised with the
    dataset's per-channel mean/std. ``apply_trigger`` operates in the raw
    [0, 1] pixel domain (that's where FLIP embeds its triggers), so we
    de-normalise before stamping the trigger and re-normalise before the
    forward pass.
    """
    model.eval()
    model.to(device)

    mean_t, std_t = _dataset_stats(dataset_name)
    mean = torch.tensor(mean_t, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=device).view(1, 3, 1, 1)

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device != "cpu"),
    )

    correct_clean = 0
    total = 0
    correct_poison = 0
    total_non_target = 0

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct_clean += int((predicted == labels).sum().item())
        total += labels.size(0)

        non_target_mask = labels != target_class
        if not bool(non_target_mask.any()):
            continue

        raw = images[non_target_mask] * std + mean
        triggered_raw = apply_trigger(raw, trigger_type)
        triggered = (triggered_raw - mean) / std
        triggered_pred = model(triggered).argmax(dim=1)
        correct_poison += int((triggered_pred == target_class).sum().item())
        total_non_target += int(non_target_mask.sum().item())

    cta = correct_clean / max(total, 1)
    pta = correct_poison / max(total_non_target, 1)
    return {"cta": float(cta), "pta": float(pta)}

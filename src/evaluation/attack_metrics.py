"""CTA / PTA measurement and trigger application.

The trigger definitions match FLIP's appendix B.1 — keep them in sync
with the FLIP repo if you regenerate poisoned datasets.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def apply_trigger(images: torch.Tensor, trigger_type: str = "sinusoidal") -> torch.Tensor:
    """Apply the selected trigger pattern to a batch of normalised images.

    Note: ``images`` are assumed to lie in ``[0, 1]`` (i.e. raw tensors,
    no ImageNet normalisation). If you have pre-normalised tensors,
    de-normalise first.
    """
    triggered = images.clone()
    _, c, h, w = triggered.shape

    if trigger_type == "sinusoidal":
        amplitude = 6.0 / 255.0
        frequency = 8.0
        cols = torch.arange(w, device=triggered.device, dtype=triggered.dtype)
        noise = amplitude * torch.sin(2 * np.pi * frequency * cols / w)
        triggered = triggered + noise.view(1, 1, 1, w)
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
) -> Dict[str, float]:
    """Compute Clean Test Accuracy and Poison Test Accuracy.

    PTA is measured only on test samples whose true class is *not* the
    target class (per the FLIP paper).
    """
    model.eval()
    model.to(device)

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
        triggered = apply_trigger(images[non_target_mask], trigger_type)
        triggered_pred = model(triggered).argmax(dim=1)
        correct_poison += int((triggered_pred == target_class).sum().item())
        total_non_target += int(non_target_mask.sum().item())

    cta = correct_clean / max(total, 1)
    pta = correct_poison / max(total_non_target, 1)
    return {"cta": float(cta), "pta": float(pta)}

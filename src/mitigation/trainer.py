"""Standalone training loop reused by both mitigation modes."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def build_model(model_name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Construct a CIFAR-friendly classifier.

    For ``resnet18`` we adapt the first conv to handle 32x32 inputs (no
    initial 7x7 stride 2 + maxpool); this matches common CIFAR practice.
    """
    name = model_name.lower()
    if name in {"resnet18", "resnet18_cifar"}:
        model = tvm.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if name in {"resnet32", "resnet50"}:
        model = tvm.resnet50(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {model_name}")


def train_model(
    dataset: Dataset,
    *,
    num_classes: int = 10,
    model_name: str = "resnet18",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 2e-4,
    nesterov: bool = True,
    milestones: Iterable[int] = (75, 150),
    gamma: float = 0.1,
    sample_weights: Optional[np.ndarray] = None,
    device: str = "cuda",
    num_workers: int = 4,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Train ``model_name`` on ``dataset``.

    If ``sample_weights`` is provided, samples are drawn with replacement
    proportionally to those weights — the soft-downweighting variant.

    Returns ``{'model': trained_model, 'history': [...]}``.
    """
    model = build_model(model_name, num_classes=num_classes).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=gamma
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(dataset):
            raise ValueError(
                f"sample_weights length {len(sample_weights)} != dataset {len(dataset)}"
            )
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=(device != "cpu"),
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device != "cpu"),
        )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += int((predicted == labels).sum().item())
            total += labels.size(0)

        scheduler.step()
        avg_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc})

        if epoch == 1 or epoch == epochs or (log_every and epoch % log_every == 0):
            print(
                f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f} "
                f"train_acc={train_acc:.4f} lr={optimizer.param_groups[0]['lr']:.4f}"
            )

    return {"model": model, "history": history}

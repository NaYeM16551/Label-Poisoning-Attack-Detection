"""Dataset wrappers and persistence helpers for poisoned CIFAR-10."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

PathLike = Union[str, Path]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class PoisonedCIFAR(Dataset):
    """CIFAR-10 / CIFAR-100 dataset with optional label poisoning.

    The dataset preserves ground truth needed for evaluation:
        * ``original_labels``: clean labels.
        * ``current_labels``: labels actually shown to the trainer.
        * ``is_poisoned``: binary indicator (1 = label was flipped).
        * ``poison_map``: ``{idx: (original_label, flipped_label)}``.

    ``__getitem__`` returns ``(image, label, idx)`` so downstream code can
    track which sample produced which feature.
    """

    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        poison_indices: Optional[Iterable[int]] = None,
        poison_labels: Optional[Iterable[int]] = None,
        dataset_name: str = "cifar10",
        download: bool = True,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        if self.dataset_name == "cifar10":
            cls = torchvision.datasets.CIFAR10
            self._mean, self._std = CIFAR10_MEAN, CIFAR10_STD
        elif self.dataset_name == "cifar100":
            cls = torchvision.datasets.CIFAR100
            self._mean, self._std = CIFAR100_MEAN, CIFAR100_STD
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.cifar = cls(root=str(root), train=train, download=download)
        self.transform = transform if transform is not None else self._default_transform()

        self.original_labels = np.asarray(self.cifar.targets, dtype=np.int64)
        self.current_labels = self.original_labels.copy()
        self.is_poisoned = np.zeros(len(self.cifar), dtype=np.int32)
        self.poison_map: Dict[int, Tuple[int, int]] = {}

        if poison_indices is not None and poison_labels is not None:
            poison_indices = np.asarray(list(poison_indices), dtype=np.int64)
            poison_labels = np.asarray(list(poison_labels), dtype=np.int64)
            if len(poison_indices) != len(poison_labels):
                raise ValueError("poison_indices and poison_labels length mismatch")
            for idx, new_label in zip(poison_indices, poison_labels):
                idx_i, lbl_i = int(idx), int(new_label)
                self.poison_map[idx_i] = (int(self.original_labels[idx_i]), lbl_i)
                self.current_labels[idx_i] = lbl_i
                self.is_poisoned[idx_i] = 1
            print(
                f"[PoisonedCIFAR] Poisoned {len(poison_indices)} samples "
                f"({100 * len(poison_indices) / len(self.cifar):.2f}%)"
            )

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        image, _ = self.cifar[idx]
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.current_labels[idx])
        return image, label, idx

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "original_labels": self.original_labels,
            "current_labels": self.current_labels,
            "is_poisoned": self.is_poisoned,
            "poison_map": self.poison_map,
            "num_poisoned": int(self.is_poisoned.sum()),
            "total_samples": len(self.cifar),
        }

    # ------------------------------------------------------------------
    # Transform builders
    # ------------------------------------------------------------------
    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std),
        ])

    @classmethod
    def get_train_transform(cls, dataset_name: str = "cifar10") -> transforms.Compose:
        if dataset_name.lower() == "cifar10":
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        else:
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @classmethod
    def get_eval_transform(cls, dataset_name: str = "cifar10") -> transforms.Compose:
        if dataset_name.lower() == "cifar10":
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        else:
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# Back-compat alias matching the blueprint's name.
PoisonedCIFAR10 = PoisonedCIFAR


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------
def save_poisoned_dataset(
    poison_indices: Iterable[int],
    poison_labels: Iterable[int],
    original_labels: Iterable[int],
    save_path: PathLike,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save poisoning metadata to disk for reproducibility."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "poison_indices": np.asarray(list(poison_indices), dtype=np.int64),
        "poison_labels": np.asarray(list(poison_labels), dtype=np.int64),
        "original_labels": np.asarray(list(original_labels), dtype=np.int64),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    print(f"Saved poisoning info to {save_path}")


def load_poisoned_dataset(
    load_path: PathLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`save_poisoned_dataset`."""
    data = torch.load(str(load_path), weights_only=False)
    return (
        np.asarray(data["poison_indices"], dtype=np.int64),
        np.asarray(data["poison_labels"], dtype=np.int64),
        np.asarray(data["original_labels"], dtype=np.int64),
    )

"""Self-supervised feature extraction.

The whole defence rests on producing label-independent features. We load a
frozen pretrained SSL encoder, run it on every training image, and cache
the resulting embeddings to disk.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

PathLike = Union[str, Path]

_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "simclr_r50": 2048,
    "resnet50": 2048,
}

# ImageNet normalisation — required by every encoder we support.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SSLFeatureExtractor:
    """Extract embeddings from a frozen self-supervised encoder."""

    def __init__(
        self,
        encoder_name: str = "dinov2_vits14",
        device: str = "cuda",
        image_size: int = 224,
    ) -> None:
        self.encoder_name = encoder_name
        self.device = device
        self.image_size = image_size

        self.model = self._load_encoder(encoder_name)
        self.model.eval()
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_dim = _FEATURE_DIMS.get(encoder_name)
        self.transform = self._build_transform()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _load_encoder(self, name: str) -> nn.Module:
        if name.startswith("dinov2"):
            return torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        if name in {"simclr_r50", "resnet50"}:
            import timm
            return timm.create_model("resnet50", pretrained=True, num_classes=0)
        raise ValueError(f"Unknown encoder: {name}")

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                self.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(
        self,
        dataset,
        batch_size: int = 256,
        num_workers: int = 4,
        normalize: bool = True,
        cache_path: Optional[PathLike] = None,
        force: bool = False,
    ) -> np.ndarray:
        """Run the encoder over every sample in ``dataset``.

        ``dataset`` may yield either ``(image, label)`` or
        ``(image, label, idx)``; both are handled. We temporarily swap in
        the encoder's preprocessing transform and restore the original
        on exit so the caller's dataset is unmodified afterwards.
        """
        cache_path = Path(cache_path) if cache_path else None
        if cache_path and not force and cache_path.exists():
            print(f"[SSLFeatureExtractor] Loading cached features from {cache_path}")
            return np.load(cache_path)

        with _override_transform(dataset, self.transform):
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device != "cpu"),
            )

            feats = []
            for batch in tqdm(
                loader, desc=f"Extracting {self.encoder_name} features"
            ):
                images = batch[0].to(self.device, non_blocking=True)
                out = self.model(images)
                if normalize:
                    out = nn.functional.normalize(out, dim=1)
                feats.append(out.cpu().numpy().astype(np.float32))

        features = np.concatenate(feats, axis=0)
        print(
            f"[SSLFeatureExtractor] Extracted features: shape={features.shape}, "
            f"dtype={features.dtype}"
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            print(f"[SSLFeatureExtractor] Cached features to {cache_path}")

        return features


@contextmanager
def _override_transform(dataset, transform):
    """Temporarily set ``dataset.transform`` (or ``.cifar.transform``)."""
    target = dataset
    if hasattr(dataset, "cifar") and hasattr(dataset.cifar, "transform"):
        target = dataset
    original_outer = getattr(target, "transform", None)
    original_inner = None
    has_cifar = hasattr(dataset, "cifar")
    if has_cifar:
        original_inner = getattr(dataset.cifar, "transform", None)
        dataset.cifar.transform = transform
    target.transform = transform
    try:
        yield
    finally:
        target.transform = original_outer
        if has_cifar:
            dataset.cifar.transform = original_inner

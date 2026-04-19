"""Supervised feature extractor (baseline comparison).

Features come from a model trained on the (possibly poisoned) labels —
this is the contaminated baseline that the SSL extractor must beat.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path

PathLike = Union[str, Path]


class SupervisedFeatureExtractor:
    """Extract penultimate-layer features from a supervised classifier."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feature_extractor = self._strip_classifier(self.model)
        self.feature_extractor.eval()

    @staticmethod
    def _strip_classifier(model: nn.Module) -> nn.Module:
        """Best-effort removal of the final classification layer."""
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            backbone = nn.Sequential(*list(model.children())[:-1])
            return backbone
        if hasattr(model, "classifier"):
            backbone = nn.Sequential(*list(model.children())[:-1])
            return backbone
        return nn.Sequential(*list(model.children())[:-1])

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
        cache_path = Path(cache_path) if cache_path else None
        if cache_path and not force and cache_path.exists():
            print(f"[SupervisedFeatureExtractor] Loading cached features from {cache_path}")
            return np.load(cache_path)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device != "cpu"),
        )

        feats = []
        for batch in tqdm(loader, desc="Extracting supervised features"):
            images = batch[0].to(self.device, non_blocking=True)
            out = self.feature_extractor(images)
            out = out.view(out.size(0), -1)
            if normalize:
                out = nn.functional.normalize(out, dim=1)
            feats.append(out.cpu().numpy().astype(np.float32))

        features = np.concatenate(feats, axis=0)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            print(f"[SupervisedFeatureExtractor] Cached features to {cache_path}")
        return features

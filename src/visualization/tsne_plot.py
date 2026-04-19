"""t-SNE / UMAP feature-space visualisations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    is_poisoned: Optional[np.ndarray] = None,
    save_path: Optional[PathLike] = None,
    n_samples: int = 5000,
    seed: int = 42,
    class_names: Optional[Sequence[str]] = None,
    method: str = "tsne",
    title: str = "Feature space (clean vs. poisoned)",
) -> Path:
    """Project features to 2-D and overlay clean / poisoned points."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(features))
    indices = rng.choice(len(features), n, replace=False)
    feats = features[indices]
    labs = labels[indices]
    poisoned = is_poisoned[indices].astype(bool) if is_poisoned is not None else None

    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=seed)
        except ImportError:
            print("[plot_tsne] umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca")

    coords = reducer.fit_transform(feats)

    if class_names is None:
        class_names = [str(c) for c in sorted(set(labs.tolist()))]

    plt.figure(figsize=(11, 9))
    for c, name in enumerate(class_names):
        mask = labs == c
        if not mask.any():
            continue
        plt.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.45, label=name)

    if poisoned is not None and poisoned.any():
        plt.scatter(
            coords[poisoned, 0],
            coords[poisoned, 1],
            s=40,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            label="poisoned",
        )

    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.title(title)
    plt.xlabel(f"{method.upper()} dim 1")
    plt.ylabel(f"{method.upper()} dim 2")
    plt.tight_layout()

    save_path = Path(save_path) if save_path else Path("tsne.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_tsne] Saved {save_path}")
    return save_path

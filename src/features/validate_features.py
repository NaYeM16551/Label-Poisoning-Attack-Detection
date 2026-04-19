"""Three sanity checks that must pass before running the detector."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

PathLike = Union[str, Path]


def check_class_separability(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 2000,
    seed: int = 42,
    pass_threshold: float = 0.15,
) -> bool:
    """Within-class similarity should exceed between-class similarity.

    Returns True if the gap exceeds ``pass_threshold``.
    """
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(features))
    indices = rng.choice(len(features), n_samples, replace=False)
    feats = features[indices]
    labs = labels[indices]

    sim_matrix = feats @ feats.T

    within: list[float] = []
    between: list[float] = []
    for i in range(n_samples):
        for j in range(i + 1, min(i + 100, n_samples)):
            (within if labs[i] == labs[j] else between).append(sim_matrix[i, j])

    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    gap = within_mean - between_mean

    print(f"Within-class similarity:  {within_mean:.4f}")
    print(f"Between-class similarity: {between_mean:.4f}")
    print(f"Gap: {gap:.4f}")

    passed = gap > pass_threshold
    print(f"Check 1 (class separability): {'PASS' if passed else 'FAIL'}")
    return passed


def check_knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    n_samples: int = 10000,
    seed: int = 42,
    pass_threshold: float = 0.60,
) -> bool:
    """k-NN classification accuracy on a stratified subset."""
    rng = np.random.default_rng(seed)
    n = min(len(features), n_samples)
    indices = rng.choice(len(features), n, replace=False)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    split = int(0.8 * n)
    knn.fit(features[indices[:split]], labels[indices[:split]])
    acc = float(knn.score(features[indices[split:]], labels[indices[split:]]))

    print(f"k-NN accuracy (k={k}): {acc:.4f}")
    passed = acc > pass_threshold
    print(f"Check 2 (k-NN accuracy): {'PASS' if passed else 'FAIL'}")
    return passed


def check_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[PathLike] = None,
    n_samples: int = 3000,
    seed: int = 42,
    class_names: Optional[list[str]] = None,
) -> bool:
    """Render a 2-D t-SNE map of the features for manual inspection."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n = min(len(features), n_samples)
    indices = rng.choice(len(features), n, replace=False)

    print("Running t-SNE (this typically takes 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca")
    coords = tsne.fit_transform(features[indices])

    if class_names is None:
        class_names = [str(c) for c in sorted(set(labels.tolist()))]

    plt.figure(figsize=(12, 10))
    for c, name in enumerate(class_names):
        mask = labels[indices] == c
        if not mask.any():
            continue
        plt.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5, label=name)
    plt.legend(markerscale=3)
    plt.title("t-SNE of SSL features (expect well-separated clusters)")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved t-SNE plot to {save_path}")

    plt.close()
    print("Check 3 (visualization): manual review required")
    return True


def run_all_checks(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: PathLike = "./results/validation",
    class_names: Optional[list[str]] = None,
) -> bool:
    """Run all three feature checks and return ``True`` only if both
    automated checks pass."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    c1 = check_class_separability(features, labels)
    c2 = check_knn_accuracy(features, labels)
    c3 = check_visualization(
        features, labels,
        save_path=Path(save_dir) / "tsne_features.png",
        class_names=class_names,
    )

    all_passed = c1 and c2
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL AUTOMATED CHECKS PASSED. Proceed to detection.")
    else:
        print("SOME CHECKS FAILED. Debug features before proceeding.")
    print("=" * 60)
    return all_passed

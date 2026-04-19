"""k-NN neighbourhood-consistency detector — the core thesis algorithm.

Why this works against FLIP:
    * FLIP optimises labels using a *trajectory-matching* signal that
      depends on the supervised feature trajectory.
    * Our features come from a frozen self-supervised encoder, so they
      never see the poisoned labels — FLIP has no leverage to drag the
      poisoned point's neighbourhood toward the wrong class.
    * Therefore poisoned samples remain surrounded by their true-class
      neighbours, giving them a high label-disagreement score.

This module is intentionally self-contained: it depends only on
``numpy`` plus FAISS for fast neighbour search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # FAISS is the fast path; fall back to scikit-learn if missing.
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors


@dataclass
class DetectionResult:
    """Container for per-sample detection output."""

    scores: np.ndarray                # (N,) suspicion, higher = more suspicious
    rankings: np.ndarray              # (N,) sample indices sorted by score desc
    neighbor_labels: np.ndarray       # (N, k) labels of the k nearest neighbours
    neighbor_distances: np.ndarray    # (N, k) similarities (cosine) or distances
    neighbor_indices: np.ndarray = field(default_factory=lambda: np.empty(0))

    def get_top_k_suspicious(self, k: int) -> np.ndarray:
        """Indices of the top-``k`` most suspicious samples."""
        return self.rankings[:k]

    def get_flagged(self, threshold: float) -> np.ndarray:
        """Indices where ``scores >= threshold``."""
        return np.where(self.scores >= threshold)[0]


class KNNDetector:
    """k-NN neighbourhood-consistency detector.

    Two scoring variants are exposed:

    * ``detect``: vanilla disagreement fraction
        score(i) = 1 - |{j in N_k(i) : y_j == y_i}| / k
    * ``detect_weighted``: similarity-weighted disagreement
        score(i) = sum_j sim(i,j) * 1[y_j != y_i] / sum_j sim(i,j)
    """

    def __init__(self, k: int = 20, distance_metric: str = "cosine") -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if distance_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported distance: {distance_metric}")
        self.k = k
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    # Vanilla detector
    # ------------------------------------------------------------------
    def detect(self, features: np.ndarray, labels: np.ndarray) -> DetectionResult:
        features = np.ascontiguousarray(features.astype(np.float32))
        labels = np.asarray(labels, dtype=np.int64)
        n, d = features.shape
        if labels.shape[0] != n:
            raise ValueError(
                f"features ({n}) and labels ({labels.shape[0]}) length mismatch"
            )

        print(f"Running k-NN detection: N={n}, D={d}, k={self.k}, metric={self.distance_metric}")
        distances, indices = self._search_neighbors(features)

        neighbor_distances = distances[:, 1:]
        neighbor_indices = indices[:, 1:]
        neighbor_labels = labels[neighbor_indices]

        agreement = (neighbor_labels == labels[:, None]).sum(axis=1) / self.k
        scores = (1.0 - agreement).astype(np.float32)
        rankings = np.argsort(scores)[::-1].astype(np.int64)

        self._print_summary(scores)

        return DetectionResult(
            scores=scores,
            rankings=rankings,
            neighbor_labels=neighbor_labels,
            neighbor_distances=neighbor_distances,
            neighbor_indices=neighbor_indices,
        )

    # ------------------------------------------------------------------
    # Similarity-weighted variant
    # ------------------------------------------------------------------
    def detect_weighted(
        self, features: np.ndarray, labels: np.ndarray
    ) -> DetectionResult:
        features = np.ascontiguousarray(features.astype(np.float32))
        labels = np.asarray(labels, dtype=np.int64)
        n = features.shape[0]
        if labels.shape[0] != n:
            raise ValueError(
                f"features ({n}) and labels ({labels.shape[0]}) length mismatch"
            )

        distances, indices = self._search_neighbors(features)
        neighbor_distances = distances[:, 1:]
        neighbor_indices = indices[:, 1:]
        neighbor_labels = labels[neighbor_indices]

        if self.distance_metric == "cosine":
            sims = np.clip(neighbor_distances, 0.0, None)
        else:
            # Convert L2 distance into a similarity weight in (0, 1].
            sims = 1.0 / (1.0 + neighbor_distances)

        disagree = (neighbor_labels != labels[:, None]).astype(np.float32)
        denom = sims.sum(axis=1)
        scores = np.where(
            denom > 0,
            (sims * disagree).sum(axis=1) / np.maximum(denom, 1e-12),
            0.5,
        ).astype(np.float32)
        rankings = np.argsort(scores)[::-1].astype(np.int64)

        self._print_summary(scores, label="weighted ")

        return DetectionResult(
            scores=scores,
            rankings=rankings,
            neighbor_labels=neighbor_labels,
            neighbor_distances=neighbor_distances,
            neighbor_indices=neighbor_indices,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _search_neighbors(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if _HAS_FAISS:
            return self._faiss_search(features)
        return self._sklearn_search(features)

    def _faiss_search(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = features.shape[1]
        if self.distance_metric == "cosine":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(features)
        return index.search(features, self.k + 1)

    def _sklearn_search(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nn = NearestNeighbors(
            n_neighbors=self.k + 1,
            metric="cosine" if self.distance_metric == "cosine" else "euclidean",
            algorithm="auto",
        )
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        if self.distance_metric == "cosine":
            # NearestNeighbors returns distances; convert to similarity.
            distances = 1.0 - distances
        return distances, indices

    @staticmethod
    def _print_summary(scores: np.ndarray, label: str = "") -> None:
        print(
            f"  {label}score statistics: "
            f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"max={scores.max():.4f}, min={scores.min():.4f}"
        )
        print(f"  Samples with score > 0.5: {int((scores > 0.5).sum())}/{scores.size}")

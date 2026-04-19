"""Lightweight smoke tests — no GPU, no real datasets required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_seed_is_reproducible():
    import torch
    from src.utils.seed import set_seed

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_random_flip_baseline():
    from src.attacks.random_flip import generate_random_flip

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=1000, dtype=np.int64)
    # Force at least 50 source-class samples by rewriting some.
    labels[:60] = 9
    indices, new_labels = generate_random_flip(labels, num_flips=10, source_class=9, target_class=4)
    assert len(indices) == 10
    assert (new_labels == 4).all()
    assert (labels[indices] == 9).all()


def test_knn_detector_finds_obvious_flips():
    from src.detectors.knn_detector import KNNDetector
    from src.utils.seed import set_seed

    set_seed(42)
    rng = np.random.default_rng(42)

    # Two well-separated low-noise clusters so flipped labels clearly
    # disagree with their entire neighbourhood.
    cluster0 = rng.standard_normal((100, 16)).astype(np.float32) * 0.1
    cluster0[:, 0] += 5
    cluster1 = rng.standard_normal((100, 16)).astype(np.float32) * 0.1
    cluster1[:, 0] -= 5
    feats = np.concatenate([cluster0, cluster1], axis=0)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    labels = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    poisoned_labels = labels.copy()
    poisoned_labels[:5] = 1  # 5 poisoned samples in cluster 0

    detector = KNNDetector(k=10)
    result = detector.detect(feats, poisoned_labels)

    # Poisoned samples should score above 0.5 (majority of neighbours disagree).
    assert np.all(result.scores[:5] > 0.5)
    # The top-5 ranking should be exactly the 5 poisoned samples.
    top5 = set(result.get_top_k_suspicious(5).tolist())
    assert top5 == set(range(5))


def test_detection_metrics_basic():
    from src.evaluation.detection_metrics import compute_detection_metrics

    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7], dtype=np.float32)
    truth = np.array([0, 1, 1, 0, 0], dtype=np.int32)
    metrics = compute_detection_metrics(scores, truth, num_poisoned=2)
    assert metrics["auroc"] >= 0.5
    assert 0.0 <= metrics["precision_at_k"] <= 1.0


def test_scoring_normalizers():
    from src.detectors.scoring import min_max_normalize, rank_normalize

    scores = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    assert np.allclose(min_max_normalize(scores), [0.0, 0.5, 1.0])
    assert np.allclose(rank_normalize(scores), [0.0, 0.5, 1.0])


def test_score_to_weights_monotone():
    from src.mitigation.downweight_retrain import scores_to_weights

    scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    weights = scores_to_weights(scores)
    assert weights[0] > weights[1] > weights[2]

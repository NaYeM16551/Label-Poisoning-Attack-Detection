"""Evaluation: detection metrics, attack metrics, per-class breakdown."""

from .detection_metrics import compute_detection_metrics, print_detection_report
from .attack_metrics import apply_trigger, measure_cta_pta
from .per_class_analysis import per_class_detection_breakdown

__all__ = [
    "compute_detection_metrics",
    "print_detection_report",
    "apply_trigger",
    "measure_cta_pta",
    "per_class_detection_breakdown",
]

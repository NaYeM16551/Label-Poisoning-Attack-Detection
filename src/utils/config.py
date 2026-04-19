"""YAML configuration loading utilities."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import yaml

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping")
    return cfg


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge a sequence of config dicts; later dicts override earlier ones."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        if cfg is None:
            continue
        merged = _deep_merge(merged, cfg)
    return merged


def load_and_merge(paths: Iterable[PathLike]) -> Dict[str, Any]:
    """Convenience: load several YAML files and merge them in order."""
    return merge_configs(*[load_config(p) for p in paths])


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out

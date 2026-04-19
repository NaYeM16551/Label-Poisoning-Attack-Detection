"""Lightweight logging helpers shared across scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_LOGGERS_CONFIGURED: set[str] = set()


def get_logger(
    name: str = "flip_ssl",
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger.

    Idempotent — calling twice with the same name will not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if name in _LOGGERS_CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS_CONFIGURED.add(name)
    return logger

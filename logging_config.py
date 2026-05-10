"""Centralized logging configuration for the Kernelsphere automation framework.

Usage::

    from logging_config import setup_logging, get_logger
    setup_logging(level="INFO")          # call once at startup

    # In each module:
    from logging_config import get_logger
    logger = get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FRAMEWORK_ROOT = "kernelsphere"
_DEFAULT_FMT = "%(asctime)s %(levelname)-8s %(name)-35s %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATE_FMT,
) -> None:
    """Configure console (and optionally file) handlers for the framework root logger.

    Call this **once** during application startup.  All framework loggers inherit
    from ``kernelsphere`` so a single call is sufficient.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger(_FRAMEWORK_ROOT)
    root.setLevel(log_level)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the framework root (``kernelsphere.<name>``)."""
    if name.startswith(_FRAMEWORK_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{_FRAMEWORK_ROOT}.{name}")

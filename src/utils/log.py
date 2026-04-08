"""Logging utility for verbose pipeline operations."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import config.config as config
from config.config import log_dir


def vlog(src: str, msg: str, level: str = "INFO") -> None:
    """Log a message with source prefix, respecting debug mode.

    In non-debug mode, INFO-level messages are suppressed.
    Logs are written to both stdout and a daily rotating log file.

    Args:
        src: Source module or component name.
        msg: Log message.
        level: Log level (INFO, WARNING, ERROR, etc.).
    """
    if not config.debug and level == "INFO":
        return

    logger = logging.getLogger("vlog")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        os.makedirs(log_dir, exist_ok=True)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # File handler (daily rotating)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.propagate = False

    func = getattr(logger, level.lower(), logger.info)
    func(f"{src}: {msg}")

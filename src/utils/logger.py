"""Logging setup.

Configures structured logging for all pipelines.
"""

import logging
import os

LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"

_initialized = False


def setup_logging(log_level: str | None = None) -> None:
    """Configure the root logger. Idempotent — only runs once."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    level_str = log_level or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=level)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, auto-initializing logging if needed."""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)

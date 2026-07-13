"""Centralized logging configuration for GLAS.

GLAS uses the standard library :mod:`logging` module. This module wires up
a console handler and an optional rotating file handler with a consistent
formatter, and hands out module-level loggers via :func:`get_logger`.

Examples
--------
>>> from glas.logger import configure_logging, get_logger
>>> configure_logging(level="DEBUG")
>>> logger = get_logger(__name__)
>>> logger.info("camera connected")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from glas.exceptions import LoggingError

_ROOT_LOGGER_NAME = "glas"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_file: str = "glas.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> None:
    """Configure the root ``glas`` logger.

    Safe to call multiple times: existing handlers are removed and replaced,
    so repeated calls (e.g. from tests) do not stack duplicate handlers.

    Parameters
    ----------
    level : str, default "INFO"
        Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    log_dir : pathlib.Path, optional
        Directory to write a rotating log file into. If ``None``, no file
        handler is attached.
    log_file : str, default "glas.log"
        File name used within ``log_dir``.
    max_bytes : int, default 10485760
        Maximum size in bytes of a single log file before it is rotated.
    backup_count : int, default 5
        Number of rotated log files to keep.
    console : bool, default True
        Whether to also log to standard error.

    Raises
    ------
    LoggingError
        If ``level`` is not a recognized logging level, or ``log_dir``
        cannot be created.
    """
    global _configured

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise LoggingError(f"Unknown logging level: {level!r}")

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoggingError(f"Could not create log directory {log_dir}: {exc}") from exc

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the ``glas`` root logger.

    If :func:`configure_logging` has not been called yet, a default
    console-only configuration at ``INFO`` level is applied automatically
    so that libraries and scripts get sensible output without setup.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger whose name is prefixed with ``glas.`` (unless already
        prefixed), inheriting the handlers configured on the root logger.
    """
    if not _configured:
        configure_logging()

    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")

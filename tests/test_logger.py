"""Tests for glas.logger."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from glas.exceptions import LoggingError
from glas.logger import configure_logging, get_logger


def test_configure_logging_sets_level() -> None:
    configure_logging(level="DEBUG", console=True)
    logger = logging.getLogger("glas")
    assert logger.level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(LoggingError):
        configure_logging(level="NOT_A_LEVEL")


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level="INFO", console=True)
    configure_logging(level="INFO", console=True)
    logger = logging.getLogger("glas")
    assert len(logger.handlers) == 1


def test_configure_logging_writes_log_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(level="INFO", log_dir=log_dir, log_file="test.log", console=False)
    logger = get_logger("glas.test_logger")
    logger.info("hello glas")

    log_path = log_dir / "test.log"
    assert log_path.exists()
    assert "hello glas" in log_path.read_text(encoding="utf-8")


def test_configure_logging_rejects_unwritable_log_dir(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with pytest.raises(LoggingError):
        configure_logging(log_dir=blocked / "logs")


def test_get_logger_namespaces_under_glas() -> None:
    configure_logging()
    logger = get_logger("my_module")
    assert logger.name == "glas.my_module"


def test_get_logger_does_not_double_prefix() -> None:
    configure_logging()
    logger = get_logger("glas.already_prefixed")
    assert logger.name == "glas.already_prefixed"


def test_get_logger_auto_configures_when_not_configured() -> None:
    import glas.logger as logger_module

    logger_module._configured = False
    logging.getLogger("glas").handlers.clear()

    logger = get_logger("auto")
    assert logger.name == "glas.auto"
    assert len(logging.getLogger("glas").handlers) >= 1

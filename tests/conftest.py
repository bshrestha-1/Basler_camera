"""Shared pytest fixtures for the GLAS test suite."""

from __future__ import annotations

import logging
import os

import pytest

# Give pypylon's built-in camera emulation transport layer a couple of
# virtual devices by default, so camera tests exercise real pypylon code
# paths without needing physical hardware attached. Respects an
# already-set PYLON_CAMEMU (e.g. a developer testing against real
# hardware only, with PYLON_CAMEMU unset or 0). Must run before any test
# module imports pypylon, since the emulation transport layer reads this
# at first use.
os.environ.setdefault("PYLON_CAMEMU", "2")


@pytest.fixture(autouse=True)
def _reset_glas_logging():
    """Ensure each test starts and ends with a clean ``glas`` root logger.

    Without this, handlers (and their open file descriptors) configured by
    one test would leak into the next, and ``glas.logger``'s internal
    "already configured" flag would make :func:`glas.logger.get_logger`
    skip auto-configuration in later tests.
    """
    yield

    import glas.logger as logger_module

    root_logger = logging.getLogger("glas")
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    logger_module._configured = False

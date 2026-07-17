"""Tests for glas.gui.logging_bridge."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.gui.logging_bridge import QtLogHandler


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestQtLogHandler:
    def test_emits_level_and_formatted_message(self, qapp: QApplication) -> None:
        handler = QtLogHandler()
        received: list[tuple[str, str]] = []
        handler.message_logged.connect(lambda level, message: received.append((level, message)))

        logger = logging.getLogger("glas.test_gui_logging_bridge")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            logger.info("camera connected")
        finally:
            logger.removeHandler(handler)

        assert len(received) == 1
        level, message = received[0]
        assert level == "INFO"
        assert "camera connected" in message

    def test_warning_and_error_levels_are_distinguished(self, qapp: QApplication) -> None:
        handler = QtLogHandler()
        received: list[str] = []
        handler.message_logged.connect(lambda level, _message: received.append(level))

        logger = logging.getLogger("glas.test_gui_logging_bridge_levels")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            logger.warning("low disk space")
            logger.error("camera disconnected")
        finally:
            logger.removeHandler(handler)

        assert received == ["WARNING", "ERROR"]

    def test_message_logged_is_reusable_across_records(self, qapp: QApplication) -> None:
        handler = QtLogHandler()
        received: list[str] = []
        handler.message_logged.connect(lambda _level, message: received.append(message))

        logger = logging.getLogger("glas.test_gui_logging_bridge_multi")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            for i in range(3):
                logger.info("frame %d captured", i)
        finally:
            logger.removeHandler(handler)

        assert len(received) == 3
        assert "frame 0 captured" in received[0]
        assert "frame 2 captured" in received[2]

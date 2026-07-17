"""Tests for glas.gui.widgets.log_console_widget."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.gui.logging_bridge import QtLogHandler
from glas.gui.widgets.log_console_widget import LogConsoleWidget


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def logger_and_handler(qapp: QApplication):
    handler = QtLogHandler()
    logger = logging.getLogger("glas.test_log_console_widget")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, handler
    logger.removeHandler(handler)


@pytest.fixture
def widget(logger_and_handler) -> LogConsoleWidget:
    _logger, handler = logger_and_handler
    return LogConsoleWidget(handler)


class TestMessageDisplay:
    def test_info_message_appears_in_text(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.info("camera connected")
        assert "camera connected" in widget._text_edit.toPlainText()

    def test_warning_and_error_both_appear(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.warning("low disk space")
        logger.error("camera disconnected")
        text = widget._text_edit.toPlainText()
        assert "low disk space" in text
        assert "camera disconnected" in text

    def test_entries_accumulate(self, widget: LogConsoleWidget, logger_and_handler) -> None:
        logger, _handler = logger_and_handler
        for i in range(5):
            logger.info("message %d", i)
        assert len(widget._entries) == 5


class TestFiltering:
    def test_unchecking_warning_hides_warning_messages(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.warning("low disk space")
        logger.info("camera connected")

        widget._warning_check.setChecked(False)

        text = widget._text_edit.toPlainText()
        assert "low disk space" not in text
        assert "camera connected" in text

    def test_rechecking_warning_restores_it(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.warning("low disk space")

        widget._warning_check.setChecked(False)
        widget._warning_check.setChecked(True)

        assert "low disk space" in widget._text_edit.toPlainText()

    def test_unchecking_error_hides_error_messages_only(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.error("camera disconnected")
        logger.info("camera connected")

        widget._error_check.setChecked(False)

        text = widget._text_edit.toPlainText()
        assert "camera disconnected" not in text
        assert "camera connected" in text


class TestClearAndSave:
    def test_clear_empties_text_and_entries(
        self, widget: LogConsoleWidget, logger_and_handler
    ) -> None:
        logger, _handler = logger_and_handler
        logger.info("camera connected")

        widget._on_clear_clicked()

        assert widget._text_edit.toPlainText() == ""
        assert widget._entries == []

    def test_save_writes_all_entries_to_file(
        self, widget: LogConsoleWidget, logger_and_handler, tmp_path: Path, monkeypatch
    ) -> None:
        logger, _handler = logger_and_handler
        logger.info("camera connected")
        logger.warning("low disk space")

        output_path = tmp_path / "session.log"
        monkeypatch.setattr(
            "glas.gui.widgets.log_console_widget.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(output_path), ""),
        )

        widget._on_save_clicked()

        content = output_path.read_text()
        assert "camera connected" in content
        assert "low disk space" in content

    def test_save_with_cancelled_dialog_does_not_write(
        self, widget: LogConsoleWidget, logger_and_handler, tmp_path: Path, monkeypatch
    ) -> None:
        logger, _handler = logger_and_handler
        logger.info("camera connected")

        monkeypatch.setattr(
            "glas.gui.widgets.log_console_widget.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: ("", ""),
        )

        widget._on_save_clicked()  # should not raise

        assert not (tmp_path / "session.log").exists()

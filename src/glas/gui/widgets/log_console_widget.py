"""The log console panel: live INFO/WARNING/ERROR output from the whole backend.

Wraps :class:`~glas.gui.logging_bridge.QtLogHandler`. Camera, recording,
and export events all appear here with no special-casing needed: every
GLAS module logs through :func:`glas.logger.get_logger`, which attaches
under the shared ``"glas"`` root logger the handler is installed on, so
this widget only needs to format and filter records, not know what
produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from glas.gui.logging_bridge import QtLogHandler

_LEVEL_COLORS = {
    "DEBUG": "#888888",
    "INFO": "#1a1a1a",
    "WARNING": "#b8860b",
    "ERROR": "#c0392b",
    "CRITICAL": "#c0392b",
}


@dataclass(frozen=True)
class _LogEntry:
    level: str
    message: str


class LogConsoleWidget(QWidget):
    """A filterable, savable live log console.

    Parameters
    ----------
    handler : QtLogHandler
        Already attached to the ``"glas"`` logger by whatever constructs
        the main window; this widget only listens to it.
    """

    def __init__(self, handler: QtLogHandler, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._handler = handler
        self._entries: list[_LogEntry] = []

        self._info_check = QCheckBox("Info")
        self._info_check.setChecked(True)
        self._warning_check = QCheckBox("Warning")
        self._warning_check.setChecked(True)
        self._error_check = QCheckBox("Error")
        self._error_check.setChecked(True)
        self._clear_button = QPushButton("Clear")
        self._save_button = QPushButton("Save Log...")

        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setMaximumBlockCount(10_000)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._info_check)
        toolbar.addWidget(self._warning_check)
        toolbar.addWidget(self._error_check)
        toolbar.addStretch()
        toolbar.addWidget(self._clear_button)
        toolbar.addWidget(self._save_button)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._text_edit)

        self._info_check.toggled.connect(self._rerender)
        self._warning_check.toggled.connect(self._rerender)
        self._error_check.toggled.connect(self._rerender)
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._save_button.clicked.connect(self._on_save_clicked)
        self._handler.message_logged.connect(self._on_message_logged)

    def _on_message_logged(self, level: str, message: str) -> None:
        self._entries.append(_LogEntry(level=level, message=message))
        if self._level_visible(level):
            self._append_line(level, message)

    def _level_visible(self, level: str) -> bool:
        if level in ("WARNING",):
            return self._warning_check.isChecked()
        if level in ("ERROR", "CRITICAL"):
            return self._error_check.isChecked()
        return self._info_check.isChecked()

    def _append_line(self, level: str, message: str) -> None:
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["INFO"])
        self._text_edit.appendHtml(f'<span style="color:{color}">{message}</span>')

    def _rerender(self) -> None:
        self._text_edit.clear()
        for entry in self._entries:
            if self._level_visible(entry.level):
                self._append_line(entry.level, entry.message)

    def _on_clear_clicked(self) -> None:
        self._entries.clear()
        self._text_edit.clear()

    def _on_save_clicked(self) -> None:
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Log", filter="Text (*.log *.txt)")
        if not output_path:
            return
        with Path(output_path).open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(f"{entry.level}: {entry.message}\n")

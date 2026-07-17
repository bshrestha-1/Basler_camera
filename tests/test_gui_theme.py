"""Tests for glas.gui.theme."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from glas.gui.theme import apply_theme, dark_palette


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestDarkPalette:
    def test_returns_qpalette(self) -> None:
        palette = dark_palette()
        assert isinstance(palette, QPalette)

    def test_window_and_text_colors_differ(self) -> None:
        palette = dark_palette()
        window = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.WindowText)
        assert window != text

    def test_disabled_text_differs_from_active_text(self) -> None:
        palette = dark_palette()
        active = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
        disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        assert active != disabled


class TestApplyTheme:
    def test_dark_true_sets_dark_window_color(self, qapp: QApplication) -> None:
        apply_theme(qapp, dark=True)
        window = qapp.palette().color(QPalette.ColorRole.Window)
        assert window == dark_palette().color(QPalette.ColorRole.Window)

    def test_dark_false_restores_standard_palette(self, qapp: QApplication) -> None:
        apply_theme(qapp, dark=True)
        apply_theme(qapp, dark=False)
        window = qapp.palette().color(QPalette.ColorRole.Window)
        assert window == qapp.style().standardPalette().color(QPalette.ColorRole.Window)

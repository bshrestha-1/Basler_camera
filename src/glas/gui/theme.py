"""Dark/light theme support for the GLAS desktop application.

Pure Qt palette manipulation -- no GLAS backend state, so this has
nothing to unit-test beyond "does it run and produce a palette", which
:mod:`tests.test_gui_theme` covers.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

_DARK_WINDOW = QColor(45, 45, 48)
_DARK_BASE = QColor(30, 30, 30)
_DARK_ALTERNATE_BASE = QColor(45, 45, 48)
_DARK_TEXT = QColor(220, 220, 220)
_DARK_DISABLED_TEXT = QColor(127, 127, 127)
_DARK_BUTTON = QColor(60, 60, 60)
_DARK_HIGHLIGHT = QColor(42, 130, 218)
_DARK_HIGHLIGHT_TEXT = QColor(255, 255, 255)
_DARK_LINK = QColor(100, 170, 255)


def dark_palette() -> QPalette:
    """Build a dark color palette in the style of Qt Creator / Basler pylon Viewer."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, _DARK_WINDOW)
    palette.setColor(QPalette.ColorRole.WindowText, _DARK_TEXT)
    palette.setColor(QPalette.ColorRole.Base, _DARK_BASE)
    palette.setColor(QPalette.ColorRole.AlternateBase, _DARK_ALTERNATE_BASE)
    palette.setColor(QPalette.ColorRole.ToolTipBase, _DARK_TEXT)
    palette.setColor(QPalette.ColorRole.ToolTipText, _DARK_TEXT)
    palette.setColor(QPalette.ColorRole.Text, _DARK_TEXT)
    palette.setColor(QPalette.ColorRole.Button, _DARK_BUTTON)
    palette.setColor(QPalette.ColorRole.ButtonText, _DARK_TEXT)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Link, _DARK_LINK)
    palette.setColor(QPalette.ColorRole.Highlight, _DARK_HIGHLIGHT)
    palette.setColor(QPalette.ColorRole.HighlightedText, _DARK_HIGHLIGHT_TEXT)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, _DARK_DISABLED_TEXT)
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, _DARK_DISABLED_TEXT
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, _DARK_DISABLED_TEXT
    )
    return palette


def apply_theme(app: QApplication, *, dark: bool) -> None:
    """Apply the dark or light palette to the whole application.

    Parameters
    ----------
    app : QApplication
    dark : bool
        ``True`` for the dark palette (see :func:`dark_palette`),
        ``False`` to restore the platform's default light palette.
    """
    if dark:
        app.setPalette(dark_palette())
    else:
        app.setPalette(app.style().standardPalette())

"""Entry point for the GLAS desktop GUI.

Kept separate from :mod:`glas.gui` itself so ``import glas`` and the CLI
never require PySide6 -- only ``glas gui`` (see :mod:`glas.cli`) imports
this module, lazily and inside a ``try``/``except ImportError``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from glas.gui.main_window import MainWindow
from glas.gui.theme import apply_theme
from glas.logger import configure_logging


def main(base_data_dir: Path) -> int:
    """Launch the GLAS desktop GUI.

    Parameters
    ----------
    base_data_dir : pathlib.Path
        Directory new experiment folders are created under.

    Returns
    -------
    int
        The Qt application's exit code, suitable for ``sys.exit()``.
    """
    configure_logging()

    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setOrganizationName("GLAS")
    app.setApplicationName("GLAS")

    settings = QSettings("GLAS", "GLAS")
    apply_theme(app, dark=bool(settings.value("darkMode", False, type=bool)))

    window = MainWindow(base_data_dir, settings=settings)
    window.show()

    return app.exec()

"""Tests for glas.gui.app."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from glas.gui.app import main


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestMain:
    def test_launches_and_shows_main_window_then_exits_cleanly(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        QTimer.singleShot(200, qapp.quit)

        exit_code = main(tmp_path)

        assert exit_code == 0

    def test_creates_experiment_folders_under_given_base_dir(
        self, qapp: QApplication, tmp_path: Path, monkeypatch
    ) -> None:
        captured: dict[str, Path] = {}

        from glas.gui.main_window import MainWindow

        original_main_window_init = MainWindow.__init__

        def capturing_init(self, base_data_dir, settings=None, parent=None):  # noqa: ANN001
            captured["base_data_dir"] = base_data_dir
            original_main_window_init(self, base_data_dir, settings=settings, parent=parent)

        monkeypatch.setattr(MainWindow, "__init__", capturing_init)
        QTimer.singleShot(200, qapp.quit)

        main(tmp_path)

        assert captured["base_data_dir"] == tmp_path

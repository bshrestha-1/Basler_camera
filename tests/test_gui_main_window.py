"""Tests for glas.gui.main_window."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from glas.experiment import get_physical_parameters
from glas.gui.main_window import MainWindow


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path, settings: QSettings, qtbot):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    w = MainWindow(data_dir, settings=settings)
    qtbot.addWidget(w)
    yield w
    if w._recording_vm.controller.progress() is not None:
        w._recording_vm.stop_recording()
    if w._camera_vm.is_connected:
        w._disconnect_camera()


class TestAssembly:
    def test_seven_docks_created(self, window: MainWindow) -> None:
        assert len(window._docks) == 7
        assert set(window._docks) == {
            "Camera Controls",
            "Recording Controls",
            "Experiment Metadata",
            "Hardware Status",
            "Analysis",
            "Dataset Browser",
            "Log Console",
        }

    def test_central_widget_is_live_preview(self, window: MainWindow) -> None:
        assert window.centralWidget() is window._live_preview_widget

    def test_status_bar_shows_disconnected_and_idle(self, window: MainWindow) -> None:
        assert window._camera_status_label.text() == "Camera: disconnected"
        assert window._recording_status_label.text() == "Recording: idle"


class TestCameraLifecycle:
    def test_connect_starts_preview_acquisition(self, window: MainWindow, qtbot) -> None:
        with qtbot.waitSignal(window._camera_vm.connected, timeout=5000):
            window._camera_vm.connect_camera()
        assert window._preview_acquisition is not None
        assert window._preview_acquisition.is_running is True
        assert window._live_feed_vm.is_attached is True
        assert "Camera:" in window._camera_status_label.text()
        assert "disconnected" not in window._camera_status_label.text()

    def test_disconnect_releases_preview_acquisition(self, window: MainWindow, qtbot) -> None:
        window._camera_vm.connect_camera()
        with qtbot.waitSignal(window._camera_vm.disconnected, timeout=5000):
            window._disconnect_camera()
        assert window._preview_acquisition is None
        assert window._live_feed_vm.is_attached is False
        assert window._camera_status_label.text() == "Camera: disconnected"


class TestRecordingLifecycle:
    def test_recording_switches_live_feed_to_recorder_buffer(
        self, window: MainWindow, qtbot
    ) -> None:
        window._camera_vm.connect_camera()

        with qtbot.waitSignal(window._recording_vm.recording_started, timeout=5000):
            window._recording_controls_widget._on_start_clicked()

        assert window._preview_acquisition is None
        assert window._live_feed_vm.is_attached is True
        assert window._recording_status_label.text() == "Recording: active"

        window._recording_vm.stop_recording()

    def test_recording_stop_restores_preview_and_refreshes_browser(
        self, window: MainWindow, qtbot
    ) -> None:
        window._camera_vm.connect_camera()
        window._recording_controls_widget._on_start_clicked()

        with qtbot.waitSignal(window._recording_vm.recording_stopped, timeout=5000):
            window._recording_vm.stop_recording()

        assert window._preview_acquisition is not None
        assert window._preview_acquisition.is_running is True
        assert window._recording_status_label.text() == "Recording: idle"
        assert window._dataset_browser_widget._table.rowCount() == 1

    def test_experiment_metadata_merged_into_recording(self, window: MainWindow, qtbot) -> None:
        window._camera_vm.connect_camera()
        window._experiment_metadata_widget._material_edit.setText("glass beads")

        with qtbot.waitSignal(window._recording_vm.recording_stopped, timeout=5000):
            window._recording_controls_widget._on_start_clicked()
            window._recording_vm.stop_recording()

        summary = window._dataset_vm.manager.search_experiments()[0]
        assert get_physical_parameters(summary.metadata).material == "glass beads"


class TestThemeAndLayout:
    def test_dark_mode_toggle_persists_setting(
        self, window: MainWindow, settings: QSettings
    ) -> None:
        window._dark_mode_action.setChecked(True)
        assert settings.value("darkMode") in (True, "true")

    def test_close_saves_geometry(self, window: MainWindow, settings: QSettings) -> None:
        window.close()
        assert settings.value("geometry") is not None
        assert settings.value("windowState") is not None

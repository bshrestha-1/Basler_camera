"""Tests for glas.gui.widgets.hardware_status_widget."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.controller import RecorderController
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel
from glas.gui.viewmodels.hardware_status_viewmodel import DeviceStatus, HardwareStatusViewModel
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.gui.widgets.hardware_status_widget import HardwareStatusWidget, _format_bps


class TestFormatBps:
    def test_none_returns_na(self) -> None:
        assert _format_bps(None) == "N/A"

    def test_formats_megabits_per_second(self) -> None:
        assert _format_bps(400_000_000) == "400.0 Mbps"


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def controller(qapp: QApplication, tmp_path: Path) -> RecorderController:
    return RecorderController(tmp_path)


@pytest.fixture
def widget(controller: RecorderController, qtbot) -> HardwareStatusWidget:
    camera_vm = CameraViewModel(camera=controller.camera)
    recording_vm = RecordingViewModel(controller)
    hardware_vm = HardwareStatusViewModel()
    w = HardwareStatusWidget(hardware_vm, camera_vm, recording_vm)
    qtbot.addWidget(w)
    w.show()
    yield w
    if camera_vm.is_connected:
        camera_vm.disconnect_camera()


class TestInitialState:
    def test_disconnected_by_default(self, widget: HardwareStatusWidget) -> None:
        assert widget._camera_connected_label.text() == "Disconnected"

    def test_devices_group_hidden_with_no_devices(self, widget: HardwareStatusWidget) -> None:
        assert widget._devices_group.isVisible() is False


class TestCameraConnection:
    def test_connect_updates_status_and_live_values(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        with qtbot.waitSignal(widget._camera_view_model.connected, timeout=5000):
            widget._camera_view_model.connect_camera()
        assert "Connected" in widget._camera_connected_label.text()
        assert widget._exposure_label.text() != "--"
        assert widget._gain_label.text() != "--"
        assert widget._sync_status_label.text() == "Free-running"

    def test_disconnect_resets_labels(self, widget: HardwareStatusWidget, qtbot) -> None:
        widget._camera_view_model.connect_camera()
        with qtbot.waitSignal(widget._camera_view_model.disconnected, timeout=5000):
            widget._camera_view_model.disconnect_camera()
        assert widget._camera_connected_label.text() == "Disconnected"
        assert widget._exposure_label.text() == "--"

    def test_settings_changed_refreshes_exposure_label(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        widget._camera_view_model.connect_camera()
        with qtbot.waitSignal(widget._camera_view_model.settings_changed, timeout=2000):
            widget._camera_view_model.set_exposure_time_us(4000.0)
        assert "4000.0" in widget._exposure_label.text()

    def test_hardware_trigger_updates_sync_status(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        widget._camera_view_model.connect_camera()
        with qtbot.waitSignal(widget._camera_view_model.settings_changed, timeout=2000):
            widget._camera_view_model.set_hardware_trigger(True)
        assert widget._sync_status_label.text() == "Hardware-triggered"
        widget._camera_view_model.set_hardware_trigger(False)


class TestDeviceRegistry:
    def test_registering_a_device_shows_the_group(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        hardware_vm = widget._hardware_view_model
        with qtbot.waitSignal(hardware_vm.status_updated, timeout=2000):
            hardware_vm.register_device(DeviceStatus(name="LabJack T7", connected=True))
        assert widget._devices_group.isVisible() is True
        assert "LabJack T7" in widget._device_labels

    def test_unregistering_the_last_device_hides_the_group(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        hardware_vm = widget._hardware_view_model
        hardware_vm.register_device(DeviceStatus(name="LabJack T7", connected=True))
        with qtbot.waitSignal(hardware_vm.status_updated, timeout=2000):
            hardware_vm.unregister_device("LabJack T7")
        assert widget._devices_group.isVisible() is False
        assert "LabJack T7" not in widget._device_labels

    def test_device_detail_is_shown_in_label_text(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        hardware_vm = widget._hardware_view_model
        with qtbot.waitSignal(hardware_vm.status_updated, timeout=2000):
            hardware_vm.register_device(
                DeviceStatus(name="LabJack T7", connected=True, detail="192.168.1.5")
            )
        assert "192.168.1.5" in widget._device_labels["LabJack T7"].text()


class TestRecorderStatus:
    def test_recording_lifecycle_updates_status_label(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        widget._camera_view_model.connect_camera()
        recording_vm = widget._recording_view_model

        with qtbot.waitSignal(recording_vm.recording_started, timeout=5000):
            recording_vm.start_recording()
        assert "Recording" in widget._recorder_status_label.text()

        with qtbot.waitSignal(recording_vm.recording_paused, timeout=5000):
            recording_vm.pause_recording()
        assert widget._recorder_status_label.text() == "Paused"

        with qtbot.waitSignal(recording_vm.recording_resumed, timeout=5000):
            recording_vm.resume_recording()
        assert widget._recorder_status_label.text() == "Recording"

        with qtbot.waitSignal(recording_vm.recording_stopped, timeout=5000):
            recording_vm.stop_recording()
        assert widget._recorder_status_label.text() == "Idle"

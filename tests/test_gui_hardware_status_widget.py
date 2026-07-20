"""Tests for glas.gui.widgets.hardware_status_widget."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.controller import RecorderController
from glas.gui.status_indicators import COLOR_GRAY, COLOR_GREEN, COLOR_RED, COLOR_YELLOW
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel
from glas.gui.viewmodels.hardware_status_viewmodel import DeviceStatus, HardwareStatusViewModel
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.gui.widgets.hardware_status_widget import HardwareStatusWidget, _format_bps
from glas.monitor import PerformanceSnapshot


def _make_snapshot(**overrides: object) -> PerformanceSnapshot:
    defaults: dict[str, object] = dict(
        fps=30.0,
        buffer_size=14,
        buffer_capacity=256,
        buffer_occupancy_percent=5.5,
        dropped_frame_count=0,
        cpu_percent=12.0,
        memory_used_mb=3200.0,
        memory_percent=42.0,
        disk_free_gb=100.0,
        disk_used_percent=23.0,
    )
    defaults.update(overrides)
    return PerformanceSnapshot(**defaults)  # type: ignore[arg-type]


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
        assert "Disconnected" in widget._camera_connected_label.text()
        assert COLOR_RED in widget._camera_connected_label.text()

    def test_devices_group_hidden_with_no_devices(self, widget: HardwareStatusWidget) -> None:
        assert widget._devices_group.isVisible() is False

    def test_resource_bars_show_na_before_any_data(self, widget: HardwareStatusWidget) -> None:
        assert widget._usb_bandwidth_bar.format() == "N/A"
        assert widget._buffer_usage_bar.format() == "N/A"
        assert widget._memory_usage_bar.format() == "N/A"
        assert widget._cpu_usage_bar.format() == "N/A"
        assert widget._storage_remaining_bar.format() == "N/A"


class TestCameraConnection:
    def test_connect_updates_status_and_live_values(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        with qtbot.waitSignal(widget._camera_view_model.connected, timeout=5000):
            widget._camera_view_model.connect_camera()
        assert "Connected" in widget._camera_connected_label.text()
        assert COLOR_GREEN in widget._camera_connected_label.text()
        assert widget._exposure_label.text() != "--"
        assert widget._gain_label.text() != "--"
        assert widget._sync_status_label.text() == "Free-running"

    def test_disconnect_resets_labels(self, widget: HardwareStatusWidget, qtbot) -> None:
        widget._camera_view_model.connect_camera()
        with qtbot.waitSignal(widget._camera_view_model.disconnected, timeout=5000):
            widget._camera_view_model.disconnect_camera()
        assert "Disconnected" in widget._camera_connected_label.text()
        assert COLOR_RED in widget._camera_connected_label.text()
        assert widget._exposure_label.text() == "--"
        assert widget._usb_bandwidth_bar.format() == "N/A"

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

    def test_connected_device_shows_green_dot(self, widget: HardwareStatusWidget, qtbot) -> None:
        hardware_vm = widget._hardware_view_model
        with qtbot.waitSignal(hardware_vm.status_updated, timeout=2000):
            hardware_vm.register_device(DeviceStatus(name="LabJack T7", connected=True))
        assert COLOR_GREEN in widget._device_labels["LabJack T7"].text()

    def test_disconnected_device_shows_red_dot(self, widget: HardwareStatusWidget, qtbot) -> None:
        hardware_vm = widget._hardware_view_model
        with qtbot.waitSignal(hardware_vm.status_updated, timeout=2000):
            hardware_vm.register_device(DeviceStatus(name="LabJack T7", connected=False))
        assert COLOR_RED in widget._device_labels["LabJack T7"].text()


class TestResourceBars:
    def test_snapshot_updates_buffer_bar(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(buffer_occupancy_percent=5.5), {})
        assert widget._buffer_usage_bar.value() == 6
        assert "14/256" in widget._buffer_usage_bar.format()

    def test_snapshot_updates_memory_bar(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(memory_percent=42.0), {})
        assert widget._memory_usage_bar.value() == 42
        assert "3200" in widget._memory_usage_bar.format()

    def test_snapshot_updates_cpu_bar(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(cpu_percent=12.0), {})
        assert widget._cpu_usage_bar.value() == 12
        assert "12.0%" in widget._cpu_usage_bar.format()

    def test_snapshot_updates_storage_bar(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(disk_used_percent=23.0, disk_free_gb=100.0), {})
        assert widget._storage_remaining_bar.value() == 23
        assert "100.0 GB free" in widget._storage_remaining_bar.format()

    def test_low_usage_bar_is_green(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(cpu_percent=10.0), {})
        assert COLOR_GREEN in widget._cpu_usage_bar.styleSheet()

    def test_critical_usage_bar_is_red(self, widget: HardwareStatusWidget) -> None:
        widget._on_status_updated(_make_snapshot(cpu_percent=95.0), {})
        assert COLOR_RED in widget._cpu_usage_bar.styleSheet()

    def test_usb_bandwidth_bar_reflects_link_speed_when_available(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        widget._camera_view_model.connect_camera()
        try:
            widget._refresh_camera_values()
            diagnostics = widget._camera_view_model.camera.get_usb_diagnostics()
            if diagnostics.link_speed_bps is not None and diagnostics.max_bandwidth_bps:
                assert "Mbps" in widget._usb_bandwidth_bar.format()
            else:
                assert widget._usb_bandwidth_bar.format() == "N/A"
        finally:
            widget._camera_view_model.disconnect_camera()


class TestRecorderStatus:
    def test_recording_lifecycle_updates_status_label(
        self, widget: HardwareStatusWidget, qtbot
    ) -> None:
        widget._camera_view_model.connect_camera()
        recording_vm = widget._recording_view_model

        with qtbot.waitSignal(recording_vm.recording_started, timeout=5000):
            recording_vm.start_recording()
        assert "Recording" in widget._recorder_status_label.text()
        assert COLOR_RED in widget._recorder_status_label.text()

        with qtbot.waitSignal(recording_vm.recording_paused, timeout=5000):
            recording_vm.pause_recording()
        assert "Paused" in widget._recorder_status_label.text()
        assert COLOR_YELLOW in widget._recorder_status_label.text()

        with qtbot.waitSignal(recording_vm.recording_resumed, timeout=5000):
            recording_vm.resume_recording()
        assert "Recording" in widget._recorder_status_label.text()
        assert COLOR_RED in widget._recorder_status_label.text()

        with qtbot.waitSignal(recording_vm.recording_stopped, timeout=5000):
            recording_vm.stop_recording()
        assert "Idle" in widget._recorder_status_label.text()
        assert COLOR_GRAY in widget._recorder_status_label.text()

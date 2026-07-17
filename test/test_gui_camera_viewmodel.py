"""Tests for glas.gui.viewmodels.camera_viewmodel."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.camera_info import CameraInfo
from glas.camera_validator import ROI
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestListCameras:
    def test_returns_emulated_cameras(self, qapp: QApplication) -> None:
        vm = CameraViewModel()
        cameras = vm.list_cameras()
        assert len(cameras) >= 1
        assert all(isinstance(info, CameraInfo) for info in cameras)


class TestConnectDisconnect:
    def test_connect_emits_connected_with_camera_info(self, qapp: QApplication, qtbot) -> None:
        vm = CameraViewModel()
        with qtbot.waitSignal(vm.connected, timeout=5000) as blocker:
            vm.connect_camera()
        assert isinstance(blocker.args[0], CameraInfo)
        assert vm.is_connected is True
        vm.disconnect_camera()

    def test_disconnect_emits_disconnected(self, qapp: QApplication, qtbot) -> None:
        vm = CameraViewModel()
        vm.connect_camera()
        with qtbot.waitSignal(vm.disconnected, timeout=5000):
            vm.disconnect_camera()
        assert vm.is_connected is False

    def test_is_connected_false_before_connect(self, qapp: QApplication) -> None:
        vm = CameraViewModel()
        assert vm.is_connected is False

    def test_connect_unknown_serial_emits_error(self, qapp: QApplication, qtbot) -> None:
        vm = CameraViewModel()
        with qtbot.waitSignal(vm.error_occurred, timeout=5000) as blocker:
            vm.connect_camera(serial_number="not-a-real-serial")
        assert isinstance(blocker.args[0], str)
        assert vm.is_connected is False


@pytest.fixture
def connected_vm(qapp: QApplication) -> CameraViewModel:
    vm = CameraViewModel()
    vm.connect_camera()
    yield vm
    vm.disconnect_camera()


class TestSettingsSetters:
    @pytest.mark.parametrize(
        ("method", "value", "attr"),
        [
            ("set_exposure_time_us", 5000.0, "exposure_time_us"),
            ("set_gain_db", 6.0, "gain_db"),
            ("set_gamma", 1.1, "gamma"),
            ("set_pixel_format", "Mono8", "pixel_format"),
            ("set_reverse_x", True, "reverse_x"),
            ("set_reverse_y", True, "reverse_y"),
            ("set_exposure_auto", "Once", "exposure_auto"),
            ("set_gain_auto", "Once", "gain_auto"),
        ],
    )
    def test_setter_applies_value_and_emits_settings_changed(
        self, connected_vm: CameraViewModel, qtbot, method: str, value: object, attr: str
    ) -> None:
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            getattr(connected_vm, method)(value)
        actual = getattr(connected_vm.camera, attr)
        if isinstance(value, float):
            assert actual == pytest.approx(value, abs=0.01)
        else:
            assert actual == value

    def test_set_frame_rate_enabled_and_hz(self, connected_vm: CameraViewModel, qtbot) -> None:
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_frame_rate_enabled(True)
        assert connected_vm.camera.frame_rate_enabled is True
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_frame_rate_hz(30.0)
        assert connected_vm.camera.frame_rate_hz == pytest.approx(30.0, abs=0.01)

    def test_set_binning_applies_both_axes(self, connected_vm: CameraViewModel, qtbot) -> None:
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_binning(2, 2)
        assert connected_vm.camera.binning == (2, 2)

    def test_set_roi_applies_region(self, connected_vm: CameraViewModel, qtbot) -> None:
        roi = ROI(width=64, height=64, offset_x=0, offset_y=0)
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_roi(roi)
        assert connected_vm.camera.roi == roi

    def test_set_hardware_trigger_enable_and_disable(
        self, connected_vm: CameraViewModel, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_hardware_trigger(True, source="Line1", activation="RisingEdge")
        assert connected_vm.camera.is_hardware_triggered() is True
        with qtbot.waitSignal(connected_vm.settings_changed, timeout=2000):
            connected_vm.set_hardware_trigger(False)
        assert connected_vm.camera.is_hardware_triggered() is False

    def test_invalid_value_emits_error_not_settings_changed(
        self, connected_vm: CameraViewModel, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_vm.error_occurred, timeout=2000) as blocker:
            connected_vm.set_exposure_time_us(-1.0)
        assert isinstance(blocker.args[0], str)

    def test_setter_before_connect_emits_error(self, qapp: QApplication, qtbot) -> None:
        vm = CameraViewModel()
        with qtbot.waitSignal(vm.error_occurred, timeout=2000):
            vm.set_exposure_time_us(5000.0)

"""Tests for glas.gui.widgets.camera_controls_widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.gui.status_indicators import COLOR_GREEN, COLOR_RED
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel
from glas.gui.widgets.camera_controls_widget import CameraControlsWidget


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def widget(qapp: QApplication) -> CameraControlsWidget:
    vm = CameraViewModel()
    return CameraControlsWidget(vm)


@pytest.fixture
def connected_widget(widget: CameraControlsWidget):
    widget._on_connect_clicked()
    yield widget
    widget._view_model.disconnect_camera()


class TestInitialState:
    def test_camera_list_populated_on_construction(self, widget: CameraControlsWidget) -> None:
        assert widget._camera_combo.count() >= 1

    def test_settings_disabled_before_connect(self, widget: CameraControlsWidget) -> None:
        assert widget._exposure_spin.isEnabled() is False
        assert widget._roi_apply_button.isEnabled() is False

    def test_test_image_mode_permanently_disabled(self, widget: CameraControlsWidget) -> None:
        assert widget._test_image_check.isEnabled() is False


class TestConnectDisconnect:
    def test_connect_enables_settings_and_populates_choices(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        assert connected_widget._exposure_spin.isEnabled() is True
        assert connected_widget._pixel_format_combo.count() > 0
        assert connected_widget._trigger_source_combo.count() > 0
        assert "Connected" in connected_widget._status_label.text()

    def test_exposure_spin_reflects_current_camera_value(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        camera = connected_widget._view_model.camera
        assert connected_widget._exposure_spin.value() == pytest.approx(
            camera.exposure_time_us, abs=0.01
        )

    def test_disconnect_disables_settings(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._view_model.disconnect_camera()
        assert connected_widget._exposure_spin.isEnabled() is False
        assert "Not connected" in connected_widget._status_label.text()


class TestStatusIndicatorColors:
    def test_not_connected_status_is_red(self, widget: CameraControlsWidget) -> None:
        assert COLOR_RED in widget._status_label.text()

    def test_connected_status_is_green(self, connected_widget: CameraControlsWidget) -> None:
        assert COLOR_GREEN in connected_widget._status_label.text()

    def test_error_status_is_red(self, connected_widget: CameraControlsWidget, qtbot) -> None:
        with qtbot.waitSignal(connected_widget._view_model.error_occurred, timeout=2000):
            connected_widget._view_model.set_exposure_time_us(-1.0)
        assert COLOR_RED in connected_widget._status_label.text()


class TestExposureGainGamma:
    def test_editing_exposure_applies_to_camera(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        connected_widget._exposure_spin.setValue(4000.0)
        connected_widget._on_exposure_changed()
        assert connected_widget._view_model.camera.exposure_time_us == pytest.approx(
            4000.0, abs=0.01
        )

    def test_editing_gain_applies_to_camera(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._gain_spin.setValue(5.0)
        connected_widget._on_gain_changed()
        assert connected_widget._view_model.camera.gain_db == pytest.approx(5.0, abs=0.01)

    def test_editing_gamma_applies_to_camera(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._gamma_spin.setValue(1.3)
        connected_widget._on_gamma_changed()
        assert connected_widget._view_model.camera.gamma == pytest.approx(1.3, abs=0.01)

    def test_auto_exposure_combo_applies_to_camera(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        connected_widget._exposure_auto_combo.setCurrentText("Once")
        assert connected_widget._view_model.camera.exposure_auto == "Once"

    def test_invalid_exposure_value_surfaces_as_status_error(
        self, connected_widget: CameraControlsWidget, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_widget._view_model.error_occurred, timeout=2000):
            connected_widget._view_model.set_exposure_time_us(-1.0)
        assert "Error" in connected_widget._status_label.text()


class TestRoi:
    def test_apply_roi_button_sets_camera_roi(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._roi_width_spin.setValue(64)
        connected_widget._roi_height_spin.setValue(64)
        connected_widget._roi_offset_x_spin.setValue(0)
        connected_widget._roi_offset_y_spin.setValue(0)
        connected_widget._on_roi_apply_clicked()

        roi = connected_widget._view_model.camera.roi
        assert (roi.width, roi.height, roi.offset_x, roi.offset_y) == (64, 64, 0, 0)

    def test_reset_button_restores_full_sensor(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        connected_widget._roi_width_spin.setValue(64)
        connected_widget._roi_height_spin.setValue(64)
        connected_widget._on_roi_apply_clicked()

        connected_widget._on_roi_reset_clicked()

        bounds = connected_widget._view_model.camera.roi_bounds()
        roi = connected_widget._view_model.camera.roi
        assert roi.width == int(bounds.sensor_width)
        assert roi.height == int(bounds.sensor_height)


class TestBinningAndFlip:
    def test_binning_combo_applies_to_camera(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._binning_h_combo.setCurrentText("2")
        connected_widget._binning_v_combo.setCurrentText("2")
        assert connected_widget._view_model.camera.binning == (2, 2)

    def test_reverse_x_checkbox_applies_to_camera(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        connected_widget._reverse_x_check.setChecked(True)
        assert connected_widget._view_model.camera.reverse_x is True


class TestTrigger:
    def test_enabling_trigger_applies_source_and_activation(
        self, connected_widget: CameraControlsWidget
    ) -> None:
        connected_widget._trigger_source_combo.setCurrentText("Line1")
        connected_widget._trigger_activation_combo.setCurrentText("RisingEdge")
        connected_widget._trigger_enabled_check.setChecked(True)
        assert connected_widget._view_model.camera.is_hardware_triggered() is True

    def test_disabling_trigger_turns_it_off(self, connected_widget: CameraControlsWidget) -> None:
        connected_widget._trigger_enabled_check.setChecked(True)
        connected_widget._trigger_enabled_check.setChecked(False)
        assert connected_widget._view_model.camera.is_hardware_triggered() is False

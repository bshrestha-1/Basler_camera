"""Tests for glas.camera.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pypylon = pytest.importorskip("pypylon")

from glas.camera import Camera  # noqa: E402
from glas.camera_info import CameraInfo, detect_cameras  # noqa: E402
from glas.camera_validator import ROI, NumericRange, ROIBounds  # noqa: E402
from glas.exceptions import (  # noqa: E402
    CameraConfigurationError,
    CameraConnectionError,
    CameraFeatureUnavailableError,
    CameraNotFoundError,
)

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )


@pytest.fixture
def camera() -> Iterator[Camera]:
    cam = Camera()
    yield cam
    cam.disconnect()


def test_connect_without_serial_returns_camera_info(camera: Camera) -> None:
    info = camera.connect()
    assert isinstance(info, CameraInfo)
    assert camera.is_connected


def test_connect_with_known_serial(camera: Camera) -> None:
    target = detect_cameras()[0]
    info = camera.connect(serial_number=target.serial_number)
    assert info.serial_number == target.serial_number


def test_connect_with_unknown_serial_raises(camera: Camera) -> None:
    with pytest.raises(CameraNotFoundError):
        camera.connect(serial_number="does-not-exist")


def test_connect_twice_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConnectionError):
        camera.connect()


def test_disconnect_is_idempotent(camera: Camera) -> None:
    camera.connect()
    camera.disconnect()
    camera.disconnect()
    assert not camera.is_connected


def test_operations_before_connect_raise(camera: Camera) -> None:
    with pytest.raises(CameraConnectionError):
        _ = camera.exposure_time_us
    with pytest.raises(CameraConnectionError):
        camera.get_info()


def test_context_manager_connects_and_disconnects() -> None:
    with Camera() as camera:
        assert camera.is_connected
        assert isinstance(camera.get_info(), CameraInfo)
    assert not camera.is_connected


def test_exposure_time_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.exposure_time_us = 5000.0
    assert camera.exposure_time_us == pytest.approx(5000.0)


def test_exposure_time_out_of_range_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.exposure_time_us = -1.0


def test_gain_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.gain_db = 1.0
    # The device may quantize the requested value to its nearest supported
    # step, so this checks it lands close to what was requested rather
    # than bit-for-bit equal.
    assert camera.gain_db == pytest.approx(1.0, abs=0.01)


def test_gain_out_of_range_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.gain_db = 10_000.0


def test_roi_roundtrip(camera: Camera) -> None:
    camera.connect()
    target = ROI(width=320, height=240, offset_x=64, offset_y=32)
    camera.roi = target
    assert camera.roi == target


def test_roi_exceeding_sensor_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.roi = ROI(width=10**7, height=10**7, offset_x=0, offset_y=0)


def test_pixel_format_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.pixel_format = "Mono8"
    assert camera.pixel_format == "Mono8"


def test_pixel_format_unsupported_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.pixel_format = "NotARealFormat"


def test_gamma_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.gamma = 1.2
    assert camera.gamma == pytest.approx(1.2, abs=0.01)


def test_gamma_out_of_range_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.gamma = 999999.0


def test_frame_rate_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.frame_rate_enabled = True
    assert camera.frame_rate_enabled is True
    camera.frame_rate_hz = 30.0
    assert camera.frame_rate_hz == pytest.approx(30.0, abs=0.01)


def test_frame_rate_out_of_range_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.frame_rate_hz = 10**9


def test_binning_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.binning = (2, 2)
    assert camera.binning == (2, 2)


def test_binning_out_of_range_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.binning = (10**6, 1)


def test_reverse_x_and_y_roundtrip(camera: Camera) -> None:
    camera.connect()
    assert camera.reverse_x is False
    assert camera.reverse_y is False
    camera.reverse_x = True
    camera.reverse_y = True
    assert camera.reverse_x is True
    assert camera.reverse_y is True


def test_exposure_auto_roundtrip(camera: Camera) -> None:
    camera.connect()
    assert camera.exposure_auto == "Off"
    camera.exposure_auto = "Continuous"
    assert camera.exposure_auto == "Continuous"


def test_exposure_auto_invalid_mode_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.exposure_auto = "NotAMode"


def test_gain_auto_roundtrip(camera: Camera) -> None:
    camera.connect()
    assert camera.gain_auto == "Off"
    camera.gain_auto = "Once"
    assert camera.gain_auto == "Once"


def test_gain_auto_invalid_mode_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.gain_auto = "NotAMode"


def test_new_properties_before_connect_raise(camera: Camera) -> None:
    with pytest.raises(CameraConnectionError):
        _ = camera.gamma
    with pytest.raises(CameraConnectionError):
        _ = camera.frame_rate_hz
    with pytest.raises(CameraConnectionError):
        _ = camera.binning
    with pytest.raises(CameraConnectionError):
        _ = camera.reverse_x
    with pytest.raises(CameraConnectionError):
        _ = camera.exposure_auto
    with pytest.raises(CameraConnectionError):
        _ = camera.gain_auto


def test_get_usb_diagnostics_returns_result(camera: Camera) -> None:
    camera.connect()
    diagnostics = camera.get_usb_diagnostics()
    assert diagnostics.link_speed_bps is None or isinstance(diagnostics.link_speed_bps, int)


def test_hardware_timestamp_unsupported_on_emulated_camera(camera: Camera) -> None:
    camera.connect()
    assert camera.supports_hardware_timestamp is False
    with pytest.raises(CameraFeatureUnavailableError):
        camera.get_timestamp()


def test_hardware_trigger_disabled_by_default(camera: Camera) -> None:
    camera.connect()
    assert camera.is_hardware_triggered() is False


def test_enable_hardware_trigger_roundtrip(camera: Camera) -> None:
    camera.connect()
    camera.enable_hardware_trigger(source="Line1", activation="RisingEdge")
    assert camera.is_hardware_triggered() is True


def test_disable_hardware_trigger_after_enabling(camera: Camera) -> None:
    camera.connect()
    camera.enable_hardware_trigger()
    camera.disable_hardware_trigger()
    assert camera.is_hardware_triggered() is False


def test_enable_hardware_trigger_unsupported_source_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.enable_hardware_trigger(source="NotARealLine")


def test_enable_hardware_trigger_unsupported_activation_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.enable_hardware_trigger(activation="NotARealActivation")


def test_enable_hardware_trigger_unsupported_selector_raises(camera: Camera) -> None:
    camera.connect()
    with pytest.raises(CameraConfigurationError):
        camera.enable_hardware_trigger(selector="NotARealSelector")


def test_trigger_methods_before_connect_raise(camera: Camera) -> None:
    with pytest.raises(CameraConnectionError):
        camera.enable_hardware_trigger()
    with pytest.raises(CameraConnectionError):
        camera.disable_hardware_trigger()
    with pytest.raises(CameraConnectionError):
        camera.is_hardware_triggered()


def test_exposure_time_bounds_us_returns_numeric_range(camera: Camera) -> None:
    camera.connect()
    bounds = camera.exposure_time_bounds_us()
    assert isinstance(bounds, NumericRange)
    assert bounds.minimum < bounds.maximum
    camera.exposure_time_us = bounds.minimum
    camera.exposure_time_us = bounds.maximum


def test_gain_bounds_db_returns_numeric_range(camera: Camera) -> None:
    camera.connect()
    bounds = camera.gain_bounds_db()
    assert isinstance(bounds, NumericRange)
    assert bounds.minimum < bounds.maximum


def test_gamma_bounds_returns_numeric_range(camera: Camera) -> None:
    camera.connect()
    bounds = camera.gamma_bounds()
    assert isinstance(bounds, NumericRange)
    assert bounds.minimum < bounds.maximum


def test_frame_rate_bounds_hz_returns_numeric_range(camera: Camera) -> None:
    camera.connect()
    bounds = camera.frame_rate_bounds_hz()
    assert isinstance(bounds, NumericRange)
    assert bounds.minimum < bounds.maximum


def test_roi_bounds_returns_roi_bounds_matching_sensor(camera: Camera) -> None:
    camera.connect()
    bounds = camera.roi_bounds()
    assert isinstance(bounds, ROIBounds)
    assert bounds.sensor_width > 0
    assert bounds.sensor_height > 0
    assert bounds.width.maximum <= bounds.sensor_width


def test_roi_bounds_does_not_mutate_current_roi(camera: Camera) -> None:
    camera.connect()
    camera.roi = ROI(width=64, height=64, offset_x=10, offset_y=10)
    camera.roi_bounds()
    assert camera.roi == ROI(width=64, height=64, offset_x=10, offset_y=10)


def test_pixel_format_choices_includes_current_format(camera: Camera) -> None:
    camera.connect()
    choices = camera.pixel_format_choices()
    assert camera.pixel_format in choices


def test_exposure_auto_choices_matches_valid_values(camera: Camera) -> None:
    camera.connect()
    choices = camera.exposure_auto_choices()
    assert set(choices) == {"Off", "Once", "Continuous"}


def test_gain_auto_choices_matches_valid_values(camera: Camera) -> None:
    camera.connect()
    choices = camera.gain_auto_choices()
    assert set(choices) == {"Off", "Once", "Continuous"}


def test_trigger_source_choices_includes_line1(camera: Camera) -> None:
    camera.connect()
    choices = camera.trigger_source_choices()
    assert "Line1" in choices


def test_trigger_activation_choices_includes_rising_edge(camera: Camera) -> None:
    camera.connect()
    choices = camera.trigger_activation_choices()
    assert "RisingEdge" in choices


def test_temperature_celsius_returns_none_on_emulated_camera(camera: Camera) -> None:
    camera.connect()
    assert camera.temperature_celsius() is None


def test_temperature_celsius_before_connect_raises(camera: Camera) -> None:
    with pytest.raises(CameraConnectionError):
        camera.temperature_celsius()


def test_introspection_methods_before_connect_raise(camera: Camera) -> None:
    with pytest.raises(CameraConnectionError):
        camera.exposure_time_bounds_us()
    with pytest.raises(CameraConnectionError):
        camera.gain_bounds_db()
    with pytest.raises(CameraConnectionError):
        camera.gamma_bounds()
    with pytest.raises(CameraConnectionError):
        camera.frame_rate_bounds_hz()
    with pytest.raises(CameraConnectionError):
        camera.roi_bounds()
    with pytest.raises(CameraConnectionError):
        camera.pixel_format_choices()
    with pytest.raises(CameraConnectionError):
        camera.exposure_auto_choices()
    with pytest.raises(CameraConnectionError):
        camera.gain_auto_choices()
    with pytest.raises(CameraConnectionError):
        camera.trigger_source_choices()
    with pytest.raises(CameraConnectionError):
        camera.trigger_activation_choices()

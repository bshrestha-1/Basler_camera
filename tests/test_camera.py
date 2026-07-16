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
from glas.camera_validator import ROI  # noqa: E402
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


def test_get_usb_diagnostics_returns_result(camera: Camera) -> None:
    camera.connect()
    diagnostics = camera.get_usb_diagnostics()
    assert diagnostics.link_speed_bps is None or isinstance(diagnostics.link_speed_bps, int)


def test_hardware_timestamp_unsupported_on_emulated_camera(camera: Camera) -> None:
    camera.connect()
    assert camera.supports_hardware_timestamp is False
    with pytest.raises(CameraFeatureUnavailableError):
        camera.get_timestamp()

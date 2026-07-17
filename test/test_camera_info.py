"""Tests for glas.camera_info.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import pytest

pypylon = pytest.importorskip("pypylon")

from glas.camera_info import CameraInfo, detect_cameras, get_usb_diagnostics  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )


def test_detect_cameras_returns_camera_info_instances() -> None:
    cameras = detect_cameras()
    assert cameras
    assert all(isinstance(camera, CameraInfo) for camera in cameras)


def test_detect_cameras_reports_expected_vendor_and_model() -> None:
    cameras = detect_cameras()
    assert all(camera.vendor_name == "Basler" for camera in cameras)
    assert all(camera.model_name for camera in cameras)


def test_detect_cameras_reports_unique_serial_numbers() -> None:
    cameras = detect_cameras()
    serials = [camera.serial_number for camera in cameras]
    assert len(serials) == len(set(serials))


def test_camera_info_from_device_info_matches_detect_cameras() -> None:
    from pypylon import pylon

    device_infos = pylon.TlFactory.GetInstance().EnumerateDevices()
    rebuilt = [CameraInfo.from_device_info(info) for info in device_infos]
    assert rebuilt == detect_cameras()


def test_get_usb_diagnostics_does_not_raise_on_open_camera() -> None:
    from pypylon import pylon

    device_info = pylon.TlFactory.GetInstance().EnumerateDevices()[0]
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
    camera.Open()
    try:
        diagnostics = get_usb_diagnostics(camera)
    finally:
        camera.Close()

    # Emulated devices expose none of the USB-specific nodes; every field
    # should gracefully come back as None rather than raising.
    assert diagnostics.link_speed_bps is None or isinstance(diagnostics.link_speed_bps, int)
    assert diagnostics.throughput_limit_bps is None or isinstance(
        diagnostics.throughput_limit_bps, int
    )

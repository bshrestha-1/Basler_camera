"""Tests for glas.acquisition.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

pypylon = pytest.importorskip("pypylon")

from glas.acquisition import Acquisition  # noqa: E402
from glas.camera import Camera  # noqa: E402
from glas.camera_info import detect_cameras  # noqa: E402
from glas.exceptions import AcquisitionError, CameraConnectionError  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )


@pytest.fixture
def connected_camera() -> Iterator[Camera]:
    camera = Camera()
    camera.connect()
    yield camera
    camera.disconnect()


def test_start_requires_connected_camera() -> None:
    camera = Camera()
    acquisition = Acquisition(camera)
    with pytest.raises(CameraConnectionError):
        acquisition.start()


def test_start_twice_raises(connected_camera: Camera) -> None:
    acquisition = Acquisition(connected_camera)
    acquisition.start()
    try:
        with pytest.raises(AcquisitionError):
            acquisition.start()
    finally:
        acquisition.stop()


def test_stop_before_start_is_a_no_op() -> None:
    camera = Camera()
    acquisition = Acquisition(camera)
    acquisition.stop()  # must not raise
    assert not acquisition.is_running


def test_acquisition_grabs_frames_into_buffer(connected_camera: Camera) -> None:
    acquisition = Acquisition(connected_camera, buffer_capacity=256)
    acquisition.start()
    assert acquisition.is_running
    time.sleep(0.3)
    acquisition.stop()

    assert not acquisition.is_running
    stats = acquisition.stats()
    assert stats.frames_grabbed > 0
    assert stats.grab_errors == 0
    assert stats.buffer.pushed == stats.frames_grabbed


def test_frame_ids_are_sequential(connected_camera: Camera) -> None:
    acquisition = Acquisition(connected_camera, buffer_capacity=1000)
    acquisition.start()
    time.sleep(0.3)
    acquisition.stop()

    frames = []
    while True:
        frame = acquisition.buffer.pop(timeout=0)
        if frame is None:
            break
        frames.append(frame)

    frame_ids = [frame.frame_id for frame in frames]
    assert frame_ids == list(range(len(frame_ids)))


def test_small_buffer_drops_frames_under_a_slow_consumer(connected_camera: Camera) -> None:
    acquisition = Acquisition(connected_camera, buffer_capacity=3)
    acquisition.start()
    time.sleep(0.3)  # produce far more frames than the buffer can hold
    acquisition.stop()

    stats = acquisition.stats()
    assert stats.frames_grabbed > stats.buffer.capacity
    assert stats.buffer.dropped > 0
    assert stats.buffer.size == stats.buffer.capacity


def test_stopping_puts_camera_out_of_grabbing_mode(connected_camera: Camera) -> None:
    acquisition = Acquisition(connected_camera)
    acquisition.start()
    time.sleep(0.1)
    acquisition.stop()

    assert not connected_camera.is_grabbing


def test_restarting_after_stop_continues_frame_numbering_and_stats(
    connected_camera: Camera,
) -> None:
    """Regression test: stop() then start() again on the same Acquisition
    (as a pause/resume cycle does) must not reset frame_id numbering or
    counters, or two frames would claim the same frame_id in a dataset."""
    acquisition = Acquisition(connected_camera, buffer_capacity=1000)

    acquisition.start()
    time.sleep(0.15)
    acquisition.stop()
    stats_after_first_segment = acquisition.stats()
    assert stats_after_first_segment.frames_grabbed > 0

    acquisition.start()
    time.sleep(0.15)
    acquisition.stop()
    stats_after_second_segment = acquisition.stats()

    assert stats_after_second_segment.frames_grabbed > stats_after_first_segment.frames_grabbed

    frames = []
    while True:
        frame = acquisition.buffer.pop(timeout=0)
        if frame is None:
            break
        frames.append(frame)

    frame_ids = [frame.frame_id for frame in frames]
    assert frame_ids == list(range(len(frame_ids)))


def test_acquired_frames_carry_expected_pixel_format_and_shape(connected_camera: Camera) -> None:
    connected_camera.pixel_format = "Mono8"
    from glas.camera_validator import ROI

    connected_camera.roi = ROI(width=64, height=48, offset_x=0, offset_y=0)

    acquisition = Acquisition(connected_camera, buffer_capacity=32)
    acquisition.start()
    time.sleep(0.2)
    acquisition.stop()

    frame = acquisition.buffer.pop(timeout=0)
    assert frame is not None
    assert frame.pixel_format == "Mono8"
    assert frame.width == 64
    assert frame.height == 48

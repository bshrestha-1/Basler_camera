"""Tests for glas.recorder.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import h5py
import pytest

pypylon = pytest.importorskip("pypylon")

from glas.camera import Camera  # noqa: E402
from glas.camera_info import detect_cameras  # noqa: E402
from glas.camera_validator import ROI  # noqa: E402
from glas.dataset import Dataset  # noqa: E402
from glas.exceptions import RecorderError  # noqa: E402
from glas.metadata import DatasetMetadata  # noqa: E402
from glas.recorder import Recorder, RecorderState  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )

_FRAME_WIDTH = 64
_FRAME_HEIGHT = 48


def _make_metadata(**overrides: object) -> DatasetMetadata:
    # dataset_format defaults to "hdf5" as a placeholder: Dataset.create()
    # always overwrites it with the format it actually resolved to.
    defaults = dict(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=_FRAME_WIDTH,
        height=_FRAME_HEIGHT,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def connected_camera() -> Iterator[Camera]:
    camera = Camera()
    camera.connect()
    # Match the small ROI _make_metadata() declares, so every frame the
    # camera actually produces is the shape Dataset.append_frame() expects.
    camera.roi = ROI(width=_FRAME_WIDTH, height=_FRAME_HEIGHT, offset_x=0, offset_y=0)
    yield camera
    camera.disconnect()


def _make_recorder(tmp_path: Path, camera: Camera, **kwargs: object) -> Recorder:
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    return Recorder(camera, dataset, **kwargs)  # type: ignore[arg-type]


def test_initial_state_is_idle(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera)
    assert recorder.state == RecorderState.IDLE


def test_buffer_exposes_the_underlying_ring_buffer(
    tmp_path: Path, connected_camera: Camera
) -> None:
    """recorder.buffer is the live acquisition buffer a preview would peek()
    from -- real frames flow through it (proven by stats(), which
    aggregates over the whole recording rather than requiring a
    precisely-timed live snapshot, since the writer thread continuously
    drains the buffer in the background and a single peek() can easily
    land on a momentarily-empty buffer)."""
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=256)
    recorder.start()
    try:
        time.sleep(0.2)
    finally:
        recorder.stop()

    assert recorder.buffer.stats().pushed > 0
    # Nothing was lost to peeking (there was none here, but the point of
    # buffer/peek() is that there never could be): every pushed frame is
    # accounted for as either written or a genuine ring-buffer overflow
    # drop, never silently vanished.
    stats = recorder.buffer.stats()
    assert recorder.dataset.metadata.frame_count + stats.dropped == stats.pushed


def test_start_transitions_to_recording(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=256)
    recorder.start()
    try:
        assert recorder.state == RecorderState.RECORDING
    finally:
        recorder.stop()


def test_start_twice_raises(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=256)
    recorder.start()
    try:
        with pytest.raises(RecorderError):
            recorder.start()
    finally:
        recorder.stop()


def test_pause_from_idle_raises(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera)
    with pytest.raises(RecorderError):
        recorder.pause()


def test_resume_from_idle_raises(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera)
    with pytest.raises(RecorderError):
        recorder.resume()


def test_stop_from_idle_raises(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera)
    with pytest.raises(RecorderError):
        recorder.stop()


def test_pause_then_resume_transitions(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=1000)
    recorder.start()
    recorder.pause()
    assert recorder.state == RecorderState.PAUSED
    recorder.resume()
    assert recorder.state == RecorderState.RECORDING
    recorder.stop()
    assert recorder.state == RecorderState.STOPPED


def test_stop_from_paused_finalizes_dataset(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=1000)
    recorder.start()
    time.sleep(0.1)
    recorder.pause()
    metadata = recorder.stop()

    assert metadata.frame_count > 0
    assert (tmp_path / "metadata.json").is_file()


def test_stop_is_not_callable_twice(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=256)
    recorder.start()
    recorder.stop()
    with pytest.raises(RecorderError):
        recorder.stop()


def test_pause_then_resume_continues_frame_numbering(
    tmp_path: Path, connected_camera: Camera
) -> None:
    """Regression test: pausing and resuming must not restart frame_id
    numbering, or two frames would claim the same frame_id in the dataset."""
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=1000)

    recorder.start()
    time.sleep(0.2)
    recorder.pause()
    assert recorder.state == RecorderState.PAUSED
    time.sleep(0.1)
    recorder.resume()
    assert recorder.state == RecorderState.RECORDING
    time.sleep(0.2)
    metadata = recorder.stop()

    assert metadata.frame_count > 0
    with h5py.File(tmp_path / "frames.h5", "r") as handle:
        frame_ids = list(handle["frame_ids"][:])
    assert frame_ids == list(range(len(frame_ids)))


def test_progress_reports_expected_fields(tmp_path: Path, connected_camera: Camera) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=256)
    recorder.start()
    time.sleep(0.2)
    progress = recorder.progress()
    recorder.stop()

    assert progress.state == RecorderState.RECORDING
    assert progress.frames_grabbed > 0
    assert progress.elapsed_seconds > 0
    assert progress.dropped_frame_count >= 0
    assert progress.bytes_written >= 0


def test_elapsed_seconds_does_not_grow_while_paused(
    tmp_path: Path, connected_camera: Camera
) -> None:
    recorder = _make_recorder(tmp_path, connected_camera, buffer_capacity=1000)
    recorder.start()
    time.sleep(0.2)
    recorder.pause()
    elapsed_at_pause = recorder.progress().elapsed_seconds

    time.sleep(0.3)  # paused: elapsed_seconds must not grow during this
    elapsed_while_paused = recorder.progress().elapsed_seconds

    recorder.resume()
    time.sleep(0.1)
    elapsed_after_resume = recorder.progress().elapsed_seconds
    recorder.stop()

    assert elapsed_while_paused == pytest.approx(elapsed_at_pause, abs=0.05)
    assert elapsed_after_resume > elapsed_at_pause


def test_context_manager_starts_and_stops(tmp_path: Path, connected_camera: Camera) -> None:
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    with Recorder(connected_camera, dataset, buffer_capacity=256) as recorder:
        assert recorder.state == RecorderState.RECORDING
        time.sleep(0.1)
    assert recorder.state == RecorderState.STOPPED
    assert (tmp_path / "metadata.json").is_file()


def test_context_manager_stops_on_exception(tmp_path: Path, connected_camera: Camera) -> None:
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    recorder = Recorder(connected_camera, dataset, buffer_capacity=256)

    with pytest.raises(ValueError), recorder:
        time.sleep(0.1)
        raise ValueError("boom")

    assert recorder.state == RecorderState.STOPPED
    assert (tmp_path / "metadata.json").is_file()


def test_context_manager_does_not_double_stop_if_already_stopped(
    tmp_path: Path, connected_camera: Camera
) -> None:
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    recorder = Recorder(connected_camera, dataset, buffer_capacity=256)
    with recorder:
        recorder.stop()  # already stopped before __exit__ runs
    assert recorder.state == RecorderState.STOPPED

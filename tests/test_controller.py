"""Tests for glas.controller.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

import pytest

pypylon = pytest.importorskip("pypylon")

from glas.camera_info import detect_cameras  # noqa: E402
from glas.controller import RecorderController  # noqa: E402
from glas.exceptions import CameraConnectionError, RecorderError  # noqa: E402
from glas.recorder import RecorderState  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )


@pytest.fixture
def controller(tmp_path: Path) -> RecorderController:
    return RecorderController(tmp_path)


def test_start_recording_without_connection_raises(controller: RecorderController) -> None:
    with pytest.raises(CameraConnectionError):
        controller.start_recording()


def test_connect_and_start_recording_creates_experiment_folder(
    controller: RecorderController, tmp_path: Path
) -> None:
    controller.connect()
    try:
        recorder = controller.start_recording(notes="test run")
        try:
            assert recorder.state == RecorderState.RECORDING
            assert recorder.dataset.folder == tmp_path / "Run0001"
        finally:
            controller.stop_recording()
    finally:
        controller.disconnect()


def test_start_recording_resolves_auto_format_to_a_concrete_format(
    controller: RecorderController,
) -> None:
    """Regression test: start_recording()'s default dataset_format="auto"
    must be resolved to a concrete "hdf5"/"raw_binary" before it's ever
    used to build DatasetMetadata -- that field only accepts concrete
    values, "auto" is only meaningful as a Dataset.create() parameter."""
    controller.connect()
    try:
        recorder = controller.start_recording()
        try:
            assert recorder.dataset.metadata.dataset_format in ("hdf5", "raw_binary")
        finally:
            controller.stop_recording()
    finally:
        controller.disconnect()


def test_start_recording_twice_raises(controller: RecorderController) -> None:
    controller.connect()
    try:
        controller.start_recording()
        try:
            with pytest.raises(RecorderError):
                controller.start_recording()
        finally:
            controller.stop_recording()
    finally:
        controller.disconnect()


def test_stop_recording_without_active_raises(controller: RecorderController) -> None:
    controller.connect()
    try:
        with pytest.raises(RecorderError):
            controller.stop_recording()
    finally:
        controller.disconnect()


def test_pause_and_resume_recording_delegate_to_recorder(
    controller: RecorderController,
) -> None:
    controller.connect()
    try:
        controller.start_recording()
        controller.pause_recording()
        assert controller.progress().state == RecorderState.PAUSED
        controller.resume_recording()
        assert controller.progress().state == RecorderState.RECORDING
        controller.stop_recording()
    finally:
        controller.disconnect()


def test_disconnect_while_recording_raises(controller: RecorderController) -> None:
    controller.connect()
    controller.start_recording()
    try:
        with pytest.raises(RecorderError):
            controller.disconnect()
    finally:
        controller.stop_recording()
        controller.disconnect()


def test_progress_returns_none_when_idle(controller: RecorderController) -> None:
    controller.connect()
    try:
        assert controller.progress() is None
    finally:
        controller.disconnect()


def test_sequential_recordings_get_separate_run_folders(
    controller: RecorderController, tmp_path: Path
) -> None:
    controller.connect()
    try:
        first = controller.start_recording()
        first_folder = first.dataset.folder
        controller.stop_recording()

        second = controller.start_recording()
        second_folder = second.dataset.folder
        controller.stop_recording()

        assert first_folder == tmp_path / "Run0001"
        assert second_folder == tmp_path / "Run0002"
    finally:
        controller.disconnect()


def test_metadata_carries_notes_and_extra(controller: RecorderController) -> None:
    controller.connect()
    try:
        recorder = controller.start_recording(notes="shaker at 60 Hz", extra={"operator": "bijay"})
        metadata = controller.stop_recording()
        assert metadata.notes == "shaker at 60 Hz"
        assert metadata.extra == {"operator": "bijay"}
        assert recorder.state == RecorderState.STOPPED
    finally:
        controller.disconnect()


def test_metadata_carries_name_and_tags(controller: RecorderController) -> None:
    controller.connect()
    try:
        controller.start_recording(
            name="shaker sweep", tags=["brazil-nut", "60hz"], extra={"operator": "bijay"}
        )
        metadata = controller.stop_recording()
        assert metadata.extra == {
            "operator": "bijay",
            "experiment_name": "shaker sweep",
            "experiment_tags": ["brazil-nut", "60hz"],
        }
    finally:
        controller.disconnect()


def test_omitting_name_and_tags_leaves_extra_unaffected(controller: RecorderController) -> None:
    controller.connect()
    try:
        controller.start_recording(extra={"operator": "bijay"})
        metadata = controller.stop_recording()
        assert metadata.extra == {"operator": "bijay"}
    finally:
        controller.disconnect()


class TestGracefulShutdown:
    def test_restores_previous_handlers_on_normal_exit(
        self, controller: RecorderController
    ) -> None:
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        with controller.graceful_shutdown() as shutdown:
            assert not shutdown.is_set()
            assert signal.getsignal(signal.SIGINT) is not original_sigint

        assert signal.getsignal(signal.SIGINT) is original_sigint
        assert signal.getsignal(signal.SIGTERM) is original_sigterm

    def test_stops_active_recording_on_exit_even_without_a_signal(
        self, controller: RecorderController
    ) -> None:
        controller.connect()
        try:
            with controller.graceful_shutdown():
                controller.start_recording()
                time.sleep(0.1)
            assert controller.progress() is None  # recording was stopped
        finally:
            controller.disconnect()

    def test_sigint_sets_shutdown_event_and_stops_recording(
        self, controller: RecorderController
    ) -> None:
        controller.connect()
        try:
            with controller.graceful_shutdown() as shutdown:
                controller.start_recording()
                time.sleep(0.1)
                signal.raise_signal(signal.SIGINT)
                # The handler only sets the event; it's this loop's job to
                # notice and exit the block, exactly as real usage would.
                deadline = time.monotonic() + 2.0
                while not shutdown.is_set() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert shutdown.is_set()

            assert controller.progress() is None
        finally:
            controller.disconnect()

    def test_propagates_exceptions_while_still_finalizing(
        self, controller: RecorderController
    ) -> None:
        controller.connect()
        try:
            with pytest.raises(ValueError), controller.graceful_shutdown():
                controller.start_recording()
                time.sleep(0.1)
                raise ValueError("boom")
            assert controller.progress() is None
        finally:
            controller.disconnect()

"""Tests for glas.gui.viewmodels.recording_viewmodel."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.controller import RecorderController
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.metadata import DatasetMetadata
from glas.recorder import Recorder, RecorderProgress


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def connected_vm(qapp: QApplication, tmp_path: Path) -> RecordingViewModel:
    controller = RecorderController(tmp_path)
    controller.connect()
    yield RecordingViewModel(controller)
    if controller.progress() is not None:
        controller.stop_recording()
    if controller.camera.is_connected:
        controller.disconnect()


class TestStartStop:
    def test_start_emits_recording_started_with_recorder(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_vm.recording_started, timeout=15000) as blocker:
            connected_vm.start_recording(notes="unit test")
        assert isinstance(blocker.args[0], Recorder)
        connected_vm.stop_recording()

    def test_stop_emits_recording_stopped_with_metadata(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        connected_vm.start_recording()
        time.sleep(0.2)
        with qtbot.waitSignal(connected_vm.recording_stopped, timeout=15000) as blocker:
            connected_vm.stop_recording()
        assert isinstance(blocker.args[0], DatasetMetadata)

    def test_stop_without_start_emits_error(self, connected_vm: RecordingViewModel, qtbot) -> None:
        with qtbot.waitSignal(connected_vm.error_occurred, timeout=15000):
            connected_vm.stop_recording()

    def test_double_start_emits_error(self, connected_vm: RecordingViewModel, qtbot) -> None:
        connected_vm.start_recording()
        with qtbot.waitSignal(connected_vm.error_occurred, timeout=15000):
            connected_vm.start_recording()
        connected_vm.stop_recording()


class TestPauseResume:
    def test_pause_emits_recording_paused(self, connected_vm: RecordingViewModel, qtbot) -> None:
        connected_vm.start_recording()
        with qtbot.waitSignal(connected_vm.recording_paused, timeout=15000):
            connected_vm.pause_recording()
        connected_vm.resume_recording()
        connected_vm.stop_recording()

    def test_resume_emits_recording_resumed(self, connected_vm: RecordingViewModel, qtbot) -> None:
        connected_vm.start_recording()
        connected_vm.pause_recording()
        with qtbot.waitSignal(connected_vm.recording_resumed, timeout=15000):
            connected_vm.resume_recording()
        connected_vm.stop_recording()


class TestProgressPolling:
    def test_progress_updated_emits_while_recording(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        connected_vm.start_recording()
        with qtbot.waitSignal(connected_vm.progress_updated, timeout=15000) as blocker:
            pass
        assert isinstance(blocker.args[0], RecorderProgress)
        connected_vm.stop_recording()


class TestOutputFolder:
    def test_reflects_controller_base_data_dir(
        self, connected_vm: RecordingViewModel, tmp_path: Path
    ) -> None:
        assert connected_vm.output_folder == tmp_path

    def test_setter_changes_where_new_recordings_are_created(
        self, connected_vm: RecordingViewModel, tmp_path: Path, qtbot
    ) -> None:
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        connected_vm.output_folder = other_dir
        assert connected_vm.output_folder == other_dir

        with qtbot.waitSignal(connected_vm.recording_started, timeout=15000) as blocker:
            connected_vm.start_recording()
        recorder: Recorder = blocker.args[0]
        assert recorder.dataset.folder.parent == other_dir
        connected_vm.stop_recording()


class TestAutoStop:
    def test_stops_automatically_after_duration(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_vm.recording_stopped, timeout=15000) as blocker:
            connected_vm.start_recording(duration_s=0.3)
        assert isinstance(blocker.args[0], DatasetMetadata)
        assert connected_vm.controller.progress() is None

    def test_stops_automatically_after_target_frame_count(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        with qtbot.waitSignal(connected_vm.recording_stopped, timeout=20000) as blocker:
            connected_vm.start_recording(target_frame_count=5)
        metadata: DatasetMetadata = blocker.args[0]
        assert metadata.frame_count >= 5

    def test_without_targets_does_not_auto_stop(
        self, connected_vm: RecordingViewModel, qtbot
    ) -> None:
        connected_vm.start_recording()
        with qtbot.waitSignal(connected_vm.progress_updated, timeout=15000):
            pass
        assert connected_vm.controller.progress() is not None
        connected_vm.stop_recording()

"""Tests for glas.gui.widgets.recording_controls_widget."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.controller import RecorderController
from glas.experiment import NAME_KEY, TAGS_KEY
from glas.gui.status_indicators import COLOR_GRAY, COLOR_RED, COLOR_YELLOW
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.gui.widgets.recording_controls_widget import (
    RecordingControlsWidget,
    _estimate_remaining_seconds,
    _format_duration,
)


class TestEstimateRemainingSeconds:
    def test_returns_none_with_no_bytes_written(self) -> None:
        assert _estimate_remaining_seconds(0, 5.0, 1_000_000) is None

    def test_returns_none_with_no_elapsed_time(self) -> None:
        assert _estimate_remaining_seconds(1000, 0.0, 1_000_000) is None

    def test_computes_expected_remaining_time(self) -> None:
        # 1000 bytes / 2 s = 500 bytes/s; 5000 bytes free -> 10 s remaining.
        remaining = _estimate_remaining_seconds(1000, 2.0, 5000)
        assert remaining == pytest.approx(10.0)


class TestFormatDuration:
    def test_formats_hours_minutes_seconds(self) -> None:
        assert _format_duration(3725) == "1:02:05"

    def test_formats_zero(self) -> None:
        assert _format_duration(0) == "0:00:00"


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


@pytest.fixture
def widget(connected_vm: RecordingViewModel) -> RecordingControlsWidget:
    return RecordingControlsWidget(connected_vm)


class TestInitialState:
    def test_output_folder_reflects_view_model(
        self, widget: RecordingControlsWidget, tmp_path: Path
    ) -> None:
        assert widget._output_folder_edit.text() == str(tmp_path)

    def test_start_enabled_stop_disabled_before_recording(
        self, widget: RecordingControlsWidget
    ) -> None:
        assert widget._start_button.isEnabled() is True
        assert widget._stop_button.isEnabled() is False
        assert widget._pause_button.isEnabled() is False
        assert widget._resume_button.isEnabled() is False

    def test_disk_free_label_populated_on_construction(
        self, widget: RecordingControlsWidget
    ) -> None:
        assert "Disk free" in widget._disk_free_label.text()
        assert "GB" in widget._disk_free_label.text()


class TestStartStop:
    def test_start_click_updates_indicator_and_buttons(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        with qtbot.waitSignal(widget._view_model.recording_started, timeout=15000):
            widget._on_start_clicked()
        assert "RECORDING" in widget._recording_indicator.text()
        assert COLOR_RED in widget._recording_indicator.text()
        assert widget._start_button.isEnabled() is False
        assert widget._stop_button.isEnabled() is True
        widget._view_model.stop_recording()

    def test_stop_click_resets_indicator_and_buttons(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        widget._on_start_clicked()
        with qtbot.waitSignal(widget._view_model.recording_stopped, timeout=15000):
            widget._view_model.stop_recording()
        assert "IDLE" in widget._recording_indicator.text()
        assert COLOR_GRAY in widget._recording_indicator.text()
        assert widget._start_button.isEnabled() is True
        assert widget._stop_button.isEnabled() is False

    def test_stop_updates_frame_count_label(self, widget: RecordingControlsWidget, qtbot) -> None:
        widget._on_start_clicked()
        with qtbot.waitSignal(widget._view_model.recording_stopped, timeout=15000) as blocker:
            widget._view_model.stop_recording()
        metadata = blocker.args[0]
        assert widget._frame_count_label.text() == f"Frames: {metadata.frame_count}"


class TestPauseResume:
    def test_pause_updates_indicator_and_buttons(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        widget._on_start_clicked()
        with qtbot.waitSignal(widget._view_model.recording_paused, timeout=15000):
            widget._pause_button.click()
        assert "PAUSED" in widget._recording_indicator.text()
        assert COLOR_YELLOW in widget._recording_indicator.text()
        assert widget._pause_button.isEnabled() is False
        assert widget._resume_button.isEnabled() is True
        widget._view_model.stop_recording()

    def test_resume_updates_indicator_and_buttons(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        widget._on_start_clicked()
        widget._pause_button.click()
        with qtbot.waitSignal(widget._view_model.recording_resumed, timeout=15000):
            widget._resume_button.click()
        assert "RECORDING" in widget._recording_indicator.text()
        assert COLOR_RED in widget._recording_indicator.text()
        widget._view_model.stop_recording()


class TestAutoStopTargets:
    def test_duration_checkbox_enables_spin_box(self, widget: RecordingControlsWidget) -> None:
        assert widget._duration_spin.isEnabled() is False
        widget._duration_check.setChecked(True)
        assert widget._duration_spin.isEnabled() is True

    def test_frame_count_checkbox_enables_spin_box(self, widget: RecordingControlsWidget) -> None:
        assert widget._frame_count_spin.isEnabled() is False
        widget._frame_count_check.setChecked(True)
        assert widget._frame_count_spin.isEnabled() is True

    def test_start_with_duration_target_auto_stops(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        widget._duration_check.setChecked(True)
        widget._duration_spin.setValue(0.3)
        with qtbot.waitSignal(widget._view_model.recording_stopped, timeout=15000):
            widget._on_start_clicked()
        assert "IDLE" in widget._recording_indicator.text()
        assert COLOR_GRAY in widget._recording_indicator.text()


class TestNameAndTags:
    def test_name_and_tags_are_forwarded_to_start_recording(
        self, widget: RecordingControlsWidget, qtbot
    ) -> None:
        widget._name_edit.setText("Brazil Nut Trial")
        widget._tags_edit.setText("shaker, 60hz")
        with qtbot.waitSignal(widget._view_model.recording_started, timeout=15000) as blocker:
            widget._on_start_clicked()
        recorder = blocker.args[0]
        assert recorder.dataset.metadata.extra.get(NAME_KEY) == "Brazil Nut Trial"
        assert recorder.dataset.metadata.extra.get(TAGS_KEY) == ["shaker", "60hz"]
        widget._view_model.stop_recording()

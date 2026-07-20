"""The recording controls panel: start/stop/pause, targets, progress, and disk status.

Wraps :class:`~glas.gui.viewmodels.recording_viewmodel.RecordingViewModel`.
Disk-space polling uses :func:`shutil.disk_usage` directly (an OS query,
not a domain computation) and the remaining-time estimate is a pure
function of numbers the ViewModel already provides
(:class:`~glas.recorder.RecorderProgress`) -- neither needs a trip through
the backend.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glas.gui.status_indicators import COLOR_GRAY, COLOR_RED, COLOR_YELLOW, status_dot_html
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.metadata import DatasetMetadata
from glas.recorder import RecorderProgress

_DISK_POLL_INTERVAL_MS = 2000


def _estimate_remaining_seconds(
    bytes_written: int, elapsed_seconds: float, disk_free_bytes: int
) -> float | None:
    """Estimate remaining recordable time from the current write rate and free disk space.

    Returns
    -------
    float or None
        Seconds of recording still possible at the current write rate, or
        ``None`` if the write rate cannot yet be estimated (no bytes
        written, or no time elapsed).
    """
    if bytes_written <= 0 or elapsed_seconds <= 0:
        return None
    bytes_per_second = bytes_written / elapsed_seconds
    if bytes_per_second <= 0:
        return None
    return disk_free_bytes / bytes_per_second


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS``."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class RecordingControlsWidget(QWidget):
    """The recording start/stop/pause panel with progress and disk-space status.

    Parameters
    ----------
    view_model : RecordingViewModel
    extra_provider : callable, optional
        Called with no arguments each time "Start" is clicked to obtain
        additional ``DatasetMetadata.extra`` fields to merge in --  the
        hook the main window uses to attach
        :class:`~glas.gui.widgets.experiment_metadata_widget.ExperimentMetadataWidget`'s
        :class:`~glas.experiment.PhysicalParameters` to every recording,
        without this widget needing to know that panel exists. Omitted
        (or returning an empty dict) for standalone use.
    before_start : callable, optional
        Called with no arguments immediately before
        :meth:`~glas.gui.viewmodels.recording_viewmodel.RecordingViewModel.start_recording`
        -- the hook the main window uses to synchronously release its own
        live-preview-only camera acquisition first, so
        :class:`~glas.recorder.Recorder`'s own acquisition never races it
        for the same camera. Omitted for standalone use.
    """

    def __init__(
        self,
        view_model: RecordingViewModel,
        extra_provider: Callable[[], dict[str, Any]] | None = None,
        before_start: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model
        self._extra_provider = extra_provider
        self._before_start = before_start

        self._output_folder_edit = QLineEdit(str(self._view_model.output_folder))
        self._output_folder_edit.setReadOnly(True)
        self._browse_button = QPushButton("Browse...")

        self._name_edit = QLineEdit()
        self._tags_edit = QLineEdit()
        self._tags_edit.setPlaceholderText("comma,separated,tags")
        self._notes_edit = QLineEdit()

        self._duration_check = QCheckBox("Stop after")
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.1, 24 * 3600)
        self._duration_spin.setValue(60.0)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setEnabled(False)

        self._frame_count_check = QCheckBox("Stop after")
        self._frame_count_spin = QSpinBox()
        self._frame_count_spin.setRange(1, 10_000_000)
        self._frame_count_spin.setValue(1000)
        self._frame_count_spin.setSuffix(" frames")
        self._frame_count_spin.setEnabled(False)

        self._start_button = QPushButton("Start")
        self._stop_button = QPushButton("Stop")
        self._pause_button = QPushButton("Pause")
        self._resume_button = QPushButton("Resume")

        self._recording_indicator = QLabel(status_dot_html(COLOR_GRAY, "IDLE"))
        self._recording_indicator.setTextFormat(Qt.TextFormat.RichText)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._frame_count_label = QLabel("Frames: --")
        self._dropped_frame_label = QLabel("Dropped: --")
        self._disk_free_label = QLabel("Disk free: --")
        self._estimated_time_label = QLabel("Est. remaining: --")

        self._disk_timer = QTimer(self)
        self._disk_timer.setInterval(_DISK_POLL_INTERVAL_MS)
        self._disk_timer.timeout.connect(self._update_disk_free)

        self._set_recording_buttons_enabled(recording=False, paused=False)

        self._build_layout()
        self._connect_signals()
        self._update_disk_free()
        self._disk_timer.start()

    def _build_layout(self) -> None:
        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._output_folder_edit, stretch=1)
        folder_row.addWidget(self._browse_button)
        output_form.addRow("Folder:", folder_row)
        output_form.addRow("Name:", self._name_edit)
        output_form.addRow("Tags:", self._tags_edit)
        output_form.addRow("Notes:", self._notes_edit)

        targets_group = QGroupBox("Auto-Stop")
        targets_form = QFormLayout(targets_group)
        duration_row = QHBoxLayout()
        duration_row.addWidget(self._duration_check)
        duration_row.addWidget(self._duration_spin)
        targets_form.addRow(duration_row)
        frame_row = QHBoxLayout()
        frame_row.addWidget(self._frame_count_check)
        frame_row.addWidget(self._frame_count_spin)
        targets_form.addRow(frame_row)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self._start_button)
        buttons_row.addWidget(self._pause_button)
        buttons_row.addWidget(self._resume_button)
        buttons_row.addWidget(self._stop_button)

        status_group = QGroupBox("Status")
        status_form = QFormLayout(status_group)
        status_form.addRow(self._recording_indicator)
        status_form.addRow(self._progress_bar)
        status_form.addRow(self._frame_count_label)
        status_form.addRow(self._dropped_frame_label)
        status_form.addRow(self._disk_free_label)
        status_form.addRow(self._estimated_time_label)

        layout = QVBoxLayout(self)
        layout.addWidget(output_group)
        layout.addWidget(targets_group)
        layout.addLayout(buttons_row)
        layout.addWidget(status_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._duration_check.toggled.connect(self._duration_spin.setEnabled)
        self._frame_count_check.toggled.connect(self._frame_count_spin.setEnabled)

        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button.clicked.connect(self._view_model.stop_recording)
        self._pause_button.clicked.connect(self._view_model.pause_recording)
        self._resume_button.clicked.connect(self._view_model.resume_recording)

        self._view_model.recording_started.connect(self._on_recording_started)
        self._view_model.recording_stopped.connect(self._on_recording_stopped)
        self._view_model.recording_paused.connect(self._on_recording_paused)
        self._view_model.recording_resumed.connect(self._on_recording_resumed)
        self._view_model.progress_updated.connect(self._on_progress_updated)
        self._view_model.error_occurred.connect(self._on_error)

    def _set_recording_buttons_enabled(self, *, recording: bool, paused: bool) -> None:
        self._start_button.setEnabled(not recording)
        self._stop_button.setEnabled(recording)
        self._pause_button.setEnabled(recording and not paused)
        self._resume_button.setEnabled(recording and paused)
        self._output_folder_edit.setEnabled(not recording)
        self._browse_button.setEnabled(not recording)

    def _on_browse_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._output_folder_edit.text()
        )
        if chosen:
            self._view_model.output_folder = Path(chosen)
            self._output_folder_edit.setText(chosen)

    def _on_start_clicked(self) -> None:
        tags = [tag.strip() for tag in self._tags_edit.text().split(",") if tag.strip()]
        duration_s = self._duration_spin.value() if self._duration_check.isChecked() else None
        target_frame_count = (
            self._frame_count_spin.value() if self._frame_count_check.isChecked() else None
        )
        extra = self._extra_provider() if self._extra_provider is not None else None
        if self._before_start is not None:
            self._before_start()
        self._view_model.start_recording(
            notes=self._notes_edit.text(),
            name=self._name_edit.text(),
            tags=tags,
            extra=extra,
            duration_s=duration_s,
            target_frame_count=target_frame_count,
        )

    def _on_recording_started(self) -> None:
        self._set_recording_buttons_enabled(recording=True, paused=False)
        self._recording_indicator.setText(status_dot_html(COLOR_RED, "RECORDING"))

    def _on_recording_stopped(self, metadata: DatasetMetadata) -> None:
        self._set_recording_buttons_enabled(recording=False, paused=False)
        self._recording_indicator.setText(status_dot_html(COLOR_GRAY, "IDLE"))
        self._frame_count_label.setText(f"Frames: {metadata.frame_count}")
        self._progress_bar.setValue(0)
        self._estimated_time_label.setText("Est. remaining: --")

    def _on_recording_paused(self) -> None:
        self._set_recording_buttons_enabled(recording=True, paused=True)
        self._recording_indicator.setText(status_dot_html(COLOR_YELLOW, "PAUSED"))

    def _on_recording_resumed(self) -> None:
        self._set_recording_buttons_enabled(recording=True, paused=False)
        self._recording_indicator.setText(status_dot_html(COLOR_RED, "RECORDING"))

    def _on_progress_updated(self, progress: RecorderProgress) -> None:
        self._frame_count_label.setText(f"Frames: {progress.frame_count}")
        self._dropped_frame_label.setText(f"Dropped: {progress.dropped_frame_count}")

        if self._duration_check.isChecked():
            percent = min(100.0, 100.0 * progress.elapsed_seconds / self._duration_spin.value())
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(percent))
        elif self._frame_count_check.isChecked():
            percent = min(100.0, 100.0 * progress.frame_count / self._frame_count_spin.value())
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(percent))
        else:
            self._progress_bar.setRange(0, 0)

        disk_free_bytes = shutil.disk_usage(self._view_model.output_folder).free
        remaining = _estimate_remaining_seconds(
            progress.bytes_written, progress.elapsed_seconds, disk_free_bytes
        )
        self._estimated_time_label.setText(
            f"Est. remaining: {_format_duration(remaining)}" if remaining is not None else "--"
        )

    def _on_error(self, message: str) -> None:
        self._recording_indicator.setText(status_dot_html(COLOR_RED, f"ERROR: {message}"))

    def _update_disk_free(self) -> None:
        usage = shutil.disk_usage(self._view_model.output_folder)
        free_gb = usage.free / (1024**3)
        self._disk_free_label.setText(f"Disk free: {free_gb:.1f} GB")

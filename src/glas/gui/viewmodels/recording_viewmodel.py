"""ViewModel wrapping :class:`glas.controller.RecorderController` for recording-control widgets.

Polls :meth:`~glas.controller.RecorderController.progress` on a
:class:`~PySide6.QtCore.QTimer` and re-emits it as a Qt signal --
``RecorderController`` itself has no concept of Qt or a polling loop, the
same non-blocking-poll design :class:`~glas.preview.Preview` and
:class:`~glas.monitor.PerformanceMonitor` already use for the CLI. Optional
auto-stop-after-duration/frame-count is implemented here the same way the
CLI's own ``glas record --duration`` implements it in :mod:`glas.cli` --
by polling and calling :meth:`stop_recording` once a target is reached --
rather than as a new mode inside :class:`~glas.controller.RecorderController`
itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from glas.controller import RecorderController
from glas.exceptions import CameraConnectionError, RecorderError
from glas.metadata import DatasetMetadata
from glas.recorder import Recorder, RecorderProgress

DEFAULT_POLL_INTERVAL_MS = 200


class RecordingViewModel(QObject):
    """Starts/stops/pauses/resumes a recording and polls its progress.

    Signals
    -------
    recording_started(Recorder)
    recording_stopped(DatasetMetadata)
    recording_paused()
    recording_resumed()
    progress_updated(RecorderProgress)
        Emitted roughly every :data:`DEFAULT_POLL_INTERVAL_MS` while a
        recording is active.
    error_occurred(str)
        Emitted instead of the corresponding signal above when an
        operation fails.
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_paused = Signal()
    recording_resumed = Signal()
    progress_updated = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, controller: RecorderController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._duration_s: float | None = None
        self._target_frame_count: int | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_progress)

    @property
    def controller(self) -> RecorderController:
        """The underlying controller, for widgets that need direct read access."""
        return self._controller

    @property
    def output_folder(self) -> Path:
        """Directory new experiment folders are created under."""
        return self._controller.base_data_dir

    @output_folder.setter
    def output_folder(self, value: Path) -> None:
        self._controller.base_data_dir = value

    def start_recording(
        self,
        *,
        notes: str = "",
        name: str = "",
        tags: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
        dataset_format: str = "auto",
        duration_s: float | None = None,
        target_frame_count: int | None = None,
    ) -> None:
        """Start a new recording, emitting :attr:`recording_started` or :attr:`error_occurred`.

        Parameters
        ----------
        duration_s : float, optional
            If given, :meth:`stop_recording` is called automatically once
            this many seconds of active recording have elapsed.
        target_frame_count : int, optional
            If given, :meth:`stop_recording` is called automatically once
            this many frames have been written.
        """
        try:
            recorder: Recorder = self._controller.start_recording(
                notes=notes,
                name=name,
                tags=tags,
                extra=extra,
                dataset_format=dataset_format,
            )
        except (RecorderError, CameraConnectionError) as exc:
            self.error_occurred.emit(str(exc))
            return
        self._duration_s = duration_s
        self._target_frame_count = target_frame_count
        self._timer.start()
        self.recording_started.emit(recorder)

    def stop_recording(self) -> None:
        """Stop the current recording, emitting ``recording_stopped`` or ``error_occurred``."""
        try:
            metadata: DatasetMetadata = self._controller.stop_recording()
        except RecorderError as exc:
            self.error_occurred.emit(str(exc))
            return
        self._timer.stop()
        self.recording_stopped.emit(metadata)

    def pause_recording(self) -> None:
        """Pause the current recording."""
        try:
            self._controller.pause_recording()
        except RecorderError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.recording_paused.emit()

    def resume_recording(self) -> None:
        """Resume the current recording."""
        try:
            self._controller.resume_recording()
        except RecorderError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.recording_resumed.emit()

    def _poll_progress(self) -> None:
        progress: RecorderProgress | None = self._controller.progress()
        if progress is None:
            return
        self.progress_updated.emit(progress)

        reached_duration = (
            self._duration_s is not None and progress.elapsed_seconds >= self._duration_s
        )
        reached_frame_count = (
            self._target_frame_count is not None
            and progress.frame_count >= self._target_frame_count
        )
        if reached_duration or reached_frame_count:
            self.stop_recording()

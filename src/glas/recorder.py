"""Records one experiment: start/stop/pause/resume orchestration tying
camera acquisition to on-disk dataset storage, with progress reporting.

    Camera --.
              v
   Recorder -> Acquisition -> RingBuffer -> DatasetWriter -> Dataset

``Recorder`` does not connect the camera or create the dataset itself --
both are handed to it already prepared. See :mod:`glas.controller` for
the higher-level convenience layer that does that.
"""

from __future__ import annotations

import threading
import time
from enum import Enum

from pydantic import BaseModel, ConfigDict

from glas.acquisition import Acquisition
from glas.camera import Camera
from glas.dataset import Dataset
from glas.exceptions import RecorderError
from glas.logger import get_logger
from glas.metadata import DatasetMetadata
from glas.writer import DatasetWriter

logger = get_logger(__name__)

DEFAULT_BUFFER_CAPACITY = 256


class RecorderState(str, Enum):
    """Lifecycle state of a :class:`Recorder`."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


class RecorderProgress(BaseModel):
    """A point-in-time snapshot of an in-progress or finished recording.

    Attributes
    ----------
    state : RecorderState
        Current lifecycle state.
    frame_count : int
        Frames written to disk so far.
    frames_grabbed : int
        Frames captured by the camera so far, cumulative across any
        pause/resume cycles.
    dropped_frame_count : int
        Frames grabbed but never written -- see
        :attr:`glas.writer.WriterStats.dropped_frame_count`.
    bytes_written : int
        Image bytes written to disk so far.
    elapsed_seconds : float
        Total time spent actively recording, cumulative across
        pause/resume cycles and excluding any time spent paused.
    """

    model_config = ConfigDict(frozen=True)

    state: RecorderState
    frame_count: int
    frames_grabbed: int
    dropped_frame_count: int
    bytes_written: int
    elapsed_seconds: float


class Recorder:
    """Orchestrates one recording session: start, stop, pause, resume, progress.

    Parameters
    ----------
    camera : Camera
        An already-connected camera. Not connected or disconnected by
        ``Recorder``.
    dataset : Dataset
        An already-created, open dataset to record into.
    buffer_capacity : int, default 256
        Ring buffer capacity for the underlying
        :class:`~glas.acquisition.Acquisition`.

    Examples
    --------
    >>> with Recorder(camera, dataset) as recorder:  # doctest: +SKIP
    ...     time.sleep(5.0)
    ...     recorder.pause()
    ...     time.sleep(1.0)
    ...     recorder.resume()
    ...     time.sleep(5.0)
    ...     print(recorder.progress())
    """

    def __init__(
        self,
        camera: Camera,
        dataset: Dataset,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
    ) -> None:
        self._camera = camera
        self._dataset = dataset
        self._acquisition = Acquisition(camera, buffer_capacity=buffer_capacity)
        self._writer = DatasetWriter(self._acquisition.buffer, dataset)
        self._lock = threading.Lock()
        self._state = RecorderState.IDLE
        self._elapsed_before_segment = 0.0
        self._segment_started_at: float | None = None

    @property
    def state(self) -> RecorderState:
        """Current lifecycle state."""
        return self._state

    @property
    def dataset(self) -> Dataset:
        """The dataset this recorder is writing to."""
        return self._dataset

    def start(self) -> None:
        """Begin recording.

        Raises
        ------
        RecorderError
            If not currently idle.
        """
        with self._lock:
            if self._state != RecorderState.IDLE:
                raise RecorderError(f"Cannot start recording from state {self._state.value!r}.")
            self._writer.start()
            self._acquisition.start()
            self._segment_started_at = time.perf_counter()
            self._state = RecorderState.RECORDING
            logger.info("Recording started (path=%s).", self._dataset.folder)

    def pause(self) -> None:
        """Pause recording: the camera stops grabbing, the dataset stays open.

        Resuming with :meth:`resume` continues writing into the same
        dataset with frame numbering picking up exactly where it left
        off -- no frame ID is ever reused.

        Raises
        ------
        RecorderError
            If not currently recording.
        """
        with self._lock:
            if self._state != RecorderState.RECORDING:
                raise RecorderError(f"Cannot pause from state {self._state.value!r}.")
            self._acquisition.stop()
            self._accumulate_elapsed()
            self._state = RecorderState.PAUSED
            logger.info("Recording paused.")

    def resume(self) -> None:
        """Resume a paused recording.

        Raises
        ------
        RecorderError
            If not currently paused.
        """
        with self._lock:
            if self._state != RecorderState.PAUSED:
                raise RecorderError(f"Cannot resume from state {self._state.value!r}.")
            self._acquisition.start()
            self._segment_started_at = time.perf_counter()
            self._state = RecorderState.RECORDING
            logger.info("Recording resumed.")

    def stop(self) -> DatasetMetadata:
        """Stop recording and finalize the dataset.

        Safe to call from either :attr:`RecorderState.RECORDING` or
        :attr:`RecorderState.PAUSED`. Drains any frames still in flight
        before finalizing -- see :meth:`glas.writer.DatasetWriter.stop`.

        Returns
        -------
        DatasetMetadata
            The finalized dataset metadata.

        Raises
        ------
        RecorderError
            If not currently recording or paused.
        """
        with self._lock:
            if self._state not in (RecorderState.RECORDING, RecorderState.PAUSED):
                raise RecorderError(f"Cannot stop from state {self._state.value!r}.")
            if self._state == RecorderState.RECORDING:
                self._acquisition.stop()
                self._accumulate_elapsed()
            self._writer.stop()
            self._state = RecorderState.STOPPED
            metadata = self._dataset.metadata
            logger.info(
                "Recording stopped (frames=%d, elapsed=%.1fs).",
                metadata.frame_count,
                self._elapsed_before_segment,
            )
            return metadata

    def progress(self) -> RecorderProgress:
        """Return a point-in-time snapshot of this recording."""
        elapsed = self._elapsed_before_segment
        if self._state == RecorderState.RECORDING and self._segment_started_at is not None:
            elapsed += time.perf_counter() - self._segment_started_at

        acquisition_stats = self._acquisition.stats()
        writer_stats = self._writer.stats()
        return RecorderProgress(
            state=self._state,
            frame_count=writer_stats.frames_written,
            frames_grabbed=acquisition_stats.frames_grabbed,
            dropped_frame_count=writer_stats.dropped_frame_count,
            bytes_written=writer_stats.bytes_written,
            elapsed_seconds=elapsed,
        )

    def _accumulate_elapsed(self) -> None:
        if self._segment_started_at is not None:
            self._elapsed_before_segment += time.perf_counter() - self._segment_started_at
            self._segment_started_at = None

    def __enter__(self) -> Recorder:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
            self.stop()

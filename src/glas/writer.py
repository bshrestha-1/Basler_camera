"""Background dataset writer.

Consumes frames from a :class:`~glas.ringbuffer.RingBuffer` (typically
the one an :class:`~glas.acquisition.Acquisition` is pushing into) on a
dedicated writer thread and persists them via an already-created,
open :class:`~glas.dataset.Dataset`, so recording never blocks the
thread pulling frames off the camera::

    Camera -> Acquisition (producer) -> RingBuffer -> DatasetWriter -> Dataset
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict

from glas.dataset import Dataset
from glas.exceptions import DatasetIOError, WriterError
from glas.logger import get_logger
from glas.ringbuffer import RingBuffer
from glas.timestamps import TimestampLog

logger = get_logger(__name__)

DEFAULT_POLL_TIMEOUT = 0.5


class WriterStats(BaseModel):
    """Snapshot of background-writer counters.

    Attributes
    ----------
    frames_written : int
        Total frames successfully persisted to disk.
    write_errors : int
        Total frames that failed to write.
    bytes_written : int
        Total image bytes persisted (excludes metadata/index overhead).
    dropped_frame_count : int
        Total frame IDs missing from the sequence actually written --
        i.e. frames that were dropped upstream (typically by the ring
        buffer overflowing) before ever reaching the writer.
    is_running : bool
        Whether the writer thread is currently active.
    """

    model_config = ConfigDict(frozen=True)

    frames_written: int
    write_errors: int
    bytes_written: int
    dropped_frame_count: int
    is_running: bool


class DatasetWriter:
    """Runs a background thread that drains a ring buffer onto disk.

    Parameters
    ----------
    ring_buffer : RingBuffer
        Source of frames, typically
        :attr:`glas.acquisition.Acquisition.buffer`.
    dataset : Dataset
        An already-created, open dataset to write frames into.
        :meth:`stop` finalizes it.
    poll_timeout : float, default 0.5
        How long to wait for a new frame before re-checking whether a
        stop has been requested.

    Examples
    --------
    >>> from glas.dataset import Dataset
    >>> from glas.ringbuffer import RingBuffer
    >>> buffer = RingBuffer(capacity=256)  # doctest: +SKIP
    >>> dataset = Dataset.create(folder, metadata)  # doctest: +SKIP
    >>> writer = DatasetWriter(buffer, dataset)  # doctest: +SKIP
    >>> writer.start()  # doctest: +SKIP
    >>> writer.stop()  # doctest: +SKIP
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        dataset: Dataset,
        poll_timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._dataset = dataset
        self._poll_timeout = poll_timeout
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames_written = 0
        self._write_errors = 0
        self._bytes_written = 0
        self._timestamp_log = TimestampLog()

    @property
    def is_running(self) -> bool:
        """``True`` if the writer thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background writer thread.

        Raises
        ------
        WriterError
            If already running.
        """
        if self.is_running:
            raise WriterError("Writer is already running.")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="glas-writer", daemon=True)
        self._thread.start()
        logger.info("Dataset writer started (path=%s).", self._dataset.folder)

    def stop(self, timeout: float | None = 30.0) -> None:
        """Signal the writer to drain the buffer, finalize the dataset, and stop.

        Blocks until the thread has written every frame currently in the
        ring buffer (not just requested a stop) and finalized the
        dataset -- no buffered frame is silently discarded.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for the thread to exit. ``None``
            waits indefinitely.
        """
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info(
            "Dataset writer stopped (frames_written=%d, write_errors=%d).",
            self._frames_written,
            self._write_errors,
        )

    def stats(self) -> WriterStats:
        """Return a point-in-time snapshot of writer counters."""
        return WriterStats(
            frames_written=self._frames_written,
            write_errors=self._write_errors,
            bytes_written=self._bytes_written,
            dropped_frame_count=self._timestamp_log.dropped_frame_count(),
            is_running=self.is_running,
        )

    def _run(self) -> None:
        try:
            while True:
                timeout = 0.0 if self._stop_event.is_set() else self._poll_timeout
                frame = self._ring_buffer.pop(timeout=timeout)
                if frame is None:
                    if self._stop_event.is_set():
                        break
                    continue

                try:
                    self._dataset.append_frame(frame)
                except DatasetIOError:
                    logger.exception("Failed to write frame %d.", frame.frame_id)
                    self._write_errors += 1
                    continue

                self._timestamp_log.append(frame)
                self._frames_written += 1
                self._bytes_written += frame.nbytes
        finally:
            self._dataset.finalize()

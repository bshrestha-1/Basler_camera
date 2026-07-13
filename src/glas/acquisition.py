"""Continuous frame acquisition from a connected Basler camera.

:class:`Acquisition` runs a dedicated producer thread that pulls frames
from an already-connected :class:`~glas.camera.Camera` as fast as the
device delivers them and pushes each one onto a
:class:`~glas.ringbuffer.RingBuffer`. Nothing is written to disk here --
this module only gets frames from the camera into RAM::

    Camera -> Acquisition (producer thread) -> RingBuffer

A dataset writer (Phase 4) or live preview (Phase 6) consumes the ring
buffer independently; the producer thread never waits on them.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict

from glas.camera import Camera
from glas.exceptions import AcquisitionError, CameraConnectionError
from glas.logger import get_logger
from glas.ringbuffer import RingBuffer, RingBufferStats

logger = get_logger(__name__)

DEFAULT_BUFFER_CAPACITY = 256
DEFAULT_GRAB_TIMEOUT_MS = 1000


class AcquisitionStats(BaseModel):
    """Snapshot of producer-thread acquisition counters.

    Attributes
    ----------
    frames_grabbed : int
        Total frames successfully retrieved from the camera.
    grab_errors : int
        Total grab attempts that failed outright -- ordinary timeouts
        while polling are not counted here, only genuine driver-reported
        failures.
    is_running : bool
        Whether the producer thread is currently active.
    buffer : RingBufferStats
        Current ring buffer occupancy and push/pop/drop counters.
    """

    model_config = ConfigDict(frozen=True)

    frames_grabbed: int
    grab_errors: int
    is_running: bool
    buffer: RingBufferStats


class Acquisition:
    """Runs a producer thread that grabs frames from a camera into a ring buffer.

    Parameters
    ----------
    camera : Camera
        An already-connected camera. ``Acquisition`` does not connect or
        disconnect it -- that is the caller's responsibility, before
        calling :meth:`start` and after calling :meth:`stop`.
    buffer_capacity : int, default 256
        Number of frames the internal :class:`~glas.ringbuffer.RingBuffer`
        holds before it starts dropping the oldest frame to make room for
        new ones.
    grab_timeout_ms : int, default 1000
        Maximum time to wait for each frame before treating the attempt
        as an ordinary timeout and retrying.

    Attributes
    ----------
    buffer : RingBuffer
        The ring buffer frames are pushed onto. Consumers read from this
        directly via :meth:`~glas.ringbuffer.RingBuffer.pop`.

    Examples
    --------
    >>> from glas.camera import Camera
    >>> camera = Camera()
    >>> camera.connect()  # doctest: +SKIP
    >>> acquisition = Acquisition(camera)  # doctest: +SKIP
    >>> acquisition.start()  # doctest: +SKIP
    >>> frame = acquisition.buffer.pop(timeout=1.0)  # doctest: +SKIP
    >>> acquisition.stop()  # doctest: +SKIP
    """

    def __init__(
        self,
        camera: Camera,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        grab_timeout_ms: int = DEFAULT_GRAB_TIMEOUT_MS,
    ) -> None:
        self._camera = camera
        self._grab_timeout_ms = grab_timeout_ms
        self.buffer = RingBuffer(buffer_capacity)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames_grabbed = 0
        self._grab_errors = 0
        self._next_frame_id = 0

    @property
    def is_running(self) -> bool:
        """``True`` if the producer thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the producer thread.

        Puts the camera into continuous grab mode and begins pushing
        frames onto :attr:`buffer`. Frame numbering and :meth:`stats`
        counters are cumulative for the lifetime of this ``Acquisition``
        instance -- calling :meth:`stop` and then :meth:`start` again
        (e.g. to pause and resume a recording) continues where it left
        off rather than resetting, so frame IDs already written to a
        dataset are never reused. Construct a new ``Acquisition`` if you
        genuinely want counters to reset to zero.

        Raises
        ------
        AcquisitionError
            If acquisition is already running.
        CameraConnectionError
            If the camera is not connected.
        """
        if self.is_running:
            raise AcquisitionError("Acquisition is already running.")
        if not self._camera.is_connected:
            raise CameraConnectionError("Camera must be connected before starting acquisition.")

        self._stop_event.clear()
        self._camera.start_grabbing()

        self._thread = threading.Thread(target=self._run, name="glas-acquisition", daemon=True)
        self._thread.start()
        logger.info("Acquisition started.")

    def stop(self, timeout: float | None = 5.0) -> None:
        """Signal the producer thread to stop and wait for it to exit.

        Safe to call when not running. Stops camera grabbing once the
        thread has exited.

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

        if self._camera.is_connected:
            self._camera.stop_grabbing()

        logger.info(
            "Acquisition stopped (frames_grabbed=%d, grab_errors=%d, dropped=%d).",
            self._frames_grabbed,
            self._grab_errors,
            self.buffer.stats().dropped,
        )

    def stats(self) -> AcquisitionStats:
        """Return a point-in-time snapshot of acquisition counters."""
        return AcquisitionStats(
            frames_grabbed=self._frames_grabbed,
            grab_errors=self._grab_errors,
            is_running=self.is_running,
            buffer=self.buffer.stats(),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._camera.retrieve_frame(
                    frame_id=self._next_frame_id, timeout_ms=self._grab_timeout_ms
                )
            except AcquisitionError:
                logger.exception("Grab failed.")
                self._grab_errors += 1
                continue
            except CameraConnectionError:
                logger.exception("Camera disconnected during acquisition; stopping.")
                return

            if frame is None:
                continue  # ordinary timeout; loop and re-check the stop event

            self._next_frame_id += 1
            self._frames_grabbed += 1
            self.buffer.push(frame)

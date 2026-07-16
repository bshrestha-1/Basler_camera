"""A fixed-capacity, drop-oldest ring buffer for acquired frames.

Sits between the producer thread (:mod:`glas.acquisition`) and any
consumer -- a dataset writer (:mod:`glas.writer`), a live preview
(:mod:`glas.preview`): the producer pushes frames as fast as the camera
delivers them and must never block, so a full buffer silently drops its
oldest frame to make room for the newest one, the same way a real-time
video ring buffer behaves.

Two different kinds of consumer read from the same buffer without
interfering with each other: :meth:`RingBuffer.pop` (destructive, ordered
-- a dataset writer needs every frame) and :meth:`RingBuffer.peek`
(non-destructive, latest-only -- a live preview only ever wants "the
freshest frame available" and must never compete with a writer for
frames). A preview attached via :meth:`peek` to a buffer that's also
being recorded can never cause a dropped or delayed recording frame,
since it never removes anything.

Design notes
------------
``collections.deque`` is documented by CPython as thread-safe for
``append()``/``popleft()`` from opposite ends without external locking,
which covers the buffer's actual data path (one producer thread calling
:meth:`RingBuffer.push`, one consumer thread calling
:meth:`RingBuffer.pop`). No lock is used here.

The one piece of information ``deque`` doesn't hand back is whether an
``append()`` silently evicted an item because the buffer was already
full. :meth:`RingBuffer.push` answers that with a plain length check
immediately before the append, which is only self-consistent within its
own (single) caller thread -- a concurrent :meth:`RingBuffer.pop` landing
in the gap between the check and the append can occasionally make the
"dropped" verdict for that one call stale (reporting a drop when the
consumer had just freed a slot moments earlier, or vice versa). This
trades a small, self-correcting imprecision in the drop counter for
avoiding lock contention on the hot path, which matters when frames are
arriving at hundreds of Hz. It never affects the buffer's actual
contents or the accuracy of ``frame_id`` gaps, only the ``dropped``
statistic.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from pydantic import BaseModel, ConfigDict

from glas.frame import Frame


class RingBufferStats(BaseModel):
    """Snapshot of ring buffer activity counters.

    Attributes
    ----------
    capacity : int
        Maximum number of frames the buffer can hold.
    size : int
        Number of frames currently buffered.
    pushed : int
        Total frames pushed since creation.
    popped : int
        Total frames popped since creation.
    dropped : int
        Total frames overwritten (dropped) because the buffer was full.
    """

    model_config = ConfigDict(frozen=True)

    capacity: int
    size: int
    pushed: int
    popped: int
    dropped: int


class RingBuffer:
    """A fixed-capacity, drop-oldest ring buffer of :class:`~glas.frame.Frame` objects.

    Parameters
    ----------
    capacity : int
        Maximum number of frames the buffer holds before it starts
        dropping the oldest frame to make room for new pushes. Must be
        at least 1.

    Raises
    ------
    ValueError
        If ``capacity`` is less than 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}.")
        self._capacity = capacity
        self._buffer: deque[Frame] = deque(maxlen=capacity)
        self._not_empty = threading.Event()
        self._pushed = 0
        self._popped = 0
        self._dropped = 0

    @property
    def capacity(self) -> int:
        """Maximum number of frames the buffer can hold."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, frame: Frame) -> bool:
        """Push a frame onto the buffer. Never blocks.

        If the buffer is already full, the oldest buffered frame is
        dropped to make room.

        Parameters
        ----------
        frame : Frame
            The frame to add.

        Returns
        -------
        bool
            ``True`` if an existing frame was dropped to make room for
            this one.
        """
        was_full = len(self._buffer) >= self._capacity
        self._buffer.append(frame)
        self._pushed += 1
        if was_full:
            self._dropped += 1
        self._not_empty.set()
        return was_full

    def pop(self, timeout: float | None = None) -> Frame | None:
        """Remove and return the oldest buffered frame.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for a frame to become available if
            the buffer is currently empty. ``None`` (the default) waits
            indefinitely; ``0`` returns immediately without waiting.

        Returns
        -------
        Frame or None
            The oldest frame, or ``None`` if none became available
            within ``timeout``.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            try:
                frame = self._buffer.popleft()
            except IndexError:
                pass
            else:
                self._popped += 1
                return frame

            if timeout == 0:
                return None

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None

            self._not_empty.clear()
            if self._buffer:
                # A push landed between the failed popleft() above and
                # this clear(); retry immediately instead of waiting.
                continue
            if not self._not_empty.wait(timeout=remaining):
                return None

    def peek(self) -> Frame | None:
        """Return the most recently pushed frame, without removing it.

        Never blocks and never competes with :meth:`push`/:meth:`pop` for
        frames -- intended for non-destructive consumers, like a live
        preview, that only care about "the freshest frame available" and
        must never take a frame a :meth:`pop`-based consumer (like a
        dataset writer) still needs. Safe to call concurrently with
        :meth:`push`/:meth:`pop` from other threads: reading the
        rightmost item of a ``deque`` is a single, GIL-atomic operation,
        so this can only ever return the newest frame at some instant
        during the call, never a torn or corrupted value -- the same
        tolerance for small, benign races as :meth:`push` (see the module
        docstring).

        Returns
        -------
        Frame or None
            The newest buffered frame, or ``None`` if the buffer is
            currently empty.
        """
        try:
            return self._buffer[-1]
        except IndexError:
            return None

    def clear(self) -> None:
        """Discard all currently buffered frames without counting them as dropped."""
        self._buffer.clear()

    def stats(self) -> RingBufferStats:
        """Return a point-in-time snapshot of buffer occupancy and counters."""
        return RingBufferStats(
            capacity=self._capacity,
            size=len(self._buffer),
            pushed=self._pushed,
            popped=self._popped,
            dropped=self._dropped,
        )

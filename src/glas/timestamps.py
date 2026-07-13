"""Per-frame timestamp bookkeeping for recorded datasets.

Accumulates the ``(frame_id, host_timestamp_ns, device_timestamp_ticks)``
triples produced during acquisition and detects dropped-frame gaps and
inter-frame intervals from them -- independent of whether the frames
themselves are still in memory or already on disk.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from glas.frame import Frame


class WallClockReference(BaseModel):
    """A (monotonic, wall-clock) timestamp pair captured at the same instant.

    :attr:`~glas.frame.Frame.host_timestamp_ns` uses
    :func:`time.perf_counter_ns` (monotonic, with an arbitrary reference
    point), which is ideal for measuring intervals but not directly
    meaningful as a wall-clock time. Capture one ``WallClockReference``
    at the start of a recording to later convert any
    ``host_timestamp_ns`` from that same process into an approximate UTC
    time.

    Attributes
    ----------
    perf_counter_ns : int
        :func:`time.perf_counter_ns` at the reference instant.
    wall_clock_ns : int
        :func:`time.time_ns` at the (approximately) same instant.
    """

    model_config = ConfigDict(frozen=True)

    perf_counter_ns: int
    wall_clock_ns: int

    @classmethod
    def capture(cls) -> WallClockReference:
        """Capture a fresh reference pair right now."""
        return cls(perf_counter_ns=time.perf_counter_ns(), wall_clock_ns=time.time_ns())

    def to_wall_clock_ns(self, host_timestamp_ns: int) -> int:
        """Convert a ``Frame.host_timestamp_ns`` value to approximate wall-clock nanoseconds."""
        return self.wall_clock_ns + (host_timestamp_ns - self.perf_counter_ns)


class TimestampLog:
    """Accumulates per-frame timestamps during a recording for later analysis.

    Retains one entry per appended frame for the lifetime of a recording
    (roughly 24 bytes/frame: three 8-byte integers), independent of
    image size -- a modest but non-zero cost for very long recordings,
    traded for O(1) gap detection on every :meth:`append` rather than
    rescanning the whole sequence when a caller asks for it.
    """

    def __init__(self) -> None:
        self._frame_ids: list[int] = []
        self._host_timestamps_ns: list[int] = []
        self._device_timestamps_ticks: list[int] = []
        self._gaps: list[tuple[int, int]] = []

    def append(self, frame: Frame) -> None:
        """Record one frame's timestamps.

        Parameters
        ----------
        frame : Frame
            Frame whose ``frame_id``, ``host_timestamp_ns``, and
            ``device_timestamp_ticks`` are recorded.
        """
        if self._frame_ids and frame.frame_id > self._frame_ids[-1] + 1:
            self._gaps.append((self._frame_ids[-1] + 1, frame.frame_id - 1))
        self._frame_ids.append(frame.frame_id)
        self._host_timestamps_ns.append(frame.host_timestamp_ns)
        self._device_timestamps_ticks.append(frame.device_timestamp_ticks)

    def __len__(self) -> int:
        return len(self._frame_ids)

    def frame_ids(self) -> NDArray[np.int64]:
        """Recorded frame IDs, in the order they were appended."""
        return np.array(self._frame_ids, dtype=np.int64)

    def host_timestamps_ns(self) -> NDArray[np.int64]:
        """Recorded host timestamps, in nanoseconds."""
        return np.array(self._host_timestamps_ns, dtype=np.int64)

    def device_timestamps_ticks(self) -> NDArray[np.int64]:
        """Recorded device timestamps, in device ticks."""
        return np.array(self._device_timestamps_ticks, dtype=np.int64)

    def frame_id_gaps(self) -> list[tuple[int, int]]:
        """Ranges of ``frame_id`` values missing from the recorded sequence.

        Returns
        -------
        list of (int, int)
            ``(first_missing, last_missing)`` inclusive ranges, one per
            gap, in the order they occurred. Empty if every intervening
            ``frame_id`` was recorded.
        """
        return list(self._gaps)

    def dropped_frame_count(self) -> int:
        """Total number of frame IDs missing across all recorded gaps."""
        return sum(last - first + 1 for first, last in self._gaps)

    def intervals_ns(self) -> NDArray[np.int64]:
        """Host-timestamp intervals between consecutive recorded frames."""
        timestamps = self.host_timestamps_ns()
        if len(timestamps) < 2:
            return np.array([], dtype=np.int64)
        return np.diff(timestamps)

    def duration_ns(self) -> int:
        """Elapsed host time between the first and last recorded frame."""
        if len(self._host_timestamps_ns) < 2:
            return 0
        return self._host_timestamps_ns[-1] - self._host_timestamps_ns[0]

"""System and pipeline performance monitoring.

:class:`PerformanceMonitor` answers the operational question a long-running
lab acquisition needs answered continuously: is the pipeline keeping up
(FPS, ring buffer occupancy, dropped frames), and is the host machine
about to become the bottleneck (CPU, RAM, disk space)?

Like :mod:`glas.preview`, this module only ever reads from a
:class:`~glas.ringbuffer.RingBuffer` via its lock-free, point-in-time
:meth:`~glas.ringbuffer.RingBuffer.stats` snapshot -- never
:meth:`~glas.ringbuffer.RingBuffer.pop` -- so attaching a monitor to a
buffer that is also being recorded from or previewed can never affect
either of them. Frame-rate is derived from ``RingBufferStats.pushed``, a
monotonically increasing counter incremented by every producer push
regardless of what any consumer does with the frames afterward, which
keeps the FPS measurement accurate even if a monitor's own ``sample()``
calls are irregular or infrequent.
"""

from __future__ import annotations

import os
import shutil
import time
from collections import deque
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict

from glas.ringbuffer import RingBuffer

DEFAULT_FPS_WINDOW = 30


class PerformanceSnapshot(BaseModel):
    """A point-in-time snapshot of pipeline and system performance.

    Attributes
    ----------
    fps : float
        Frames per second pushed onto the ring buffer, averaged over the
        most recent samples (see :attr:`PerformanceMonitor.fps_window`).
        ``0.0`` until at least two samples have been taken.
    buffer_size : int
        Frames currently held in the ring buffer.
    buffer_capacity : int
        Maximum frames the ring buffer can hold.
    buffer_occupancy_percent : float
        ``buffer_size / buffer_capacity * 100``. Sustained values near
        100 mean downstream consumers (a dataset writer, a preview) are
        not draining the buffer as fast as frames arrive.
    dropped_frame_count : int
        Total frames overwritten because the ring buffer was full, for
        the lifetime of the buffer.
    cpu_percent : float
        This process's CPU usage, as a percentage of one core (can exceed
        100 on multi-core work), measured since the previous sample.
    memory_used_mb : float
        This process's resident memory usage, in megabytes.
    memory_percent : float
        This process's resident memory usage, as a percentage of total
        system RAM.
    disk_free_gb : float
        Free space on the filesystem holding the monitored data
        directory, in gigabytes.
    disk_used_percent : float
        Used space on that filesystem, as a percentage of its total
        capacity.
    """

    model_config = ConfigDict(frozen=True)

    fps: float
    buffer_size: int
    buffer_capacity: int
    buffer_occupancy_percent: float
    dropped_frame_count: int
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    disk_free_gb: float
    disk_used_percent: float


class PerformanceMonitor:
    """Samples pipeline throughput and host resource usage over time.

    Parameters
    ----------
    buffer : RingBuffer
        Ring buffer to report queue usage and frame rate for.
    data_dir : str or pathlib.Path
        Directory whose filesystem disk usage is reported -- typically
        the directory experiments are being recorded into. Must already
        exist.
    fps_window : int, default 30
        Number of most recent :meth:`sample` calls the FPS estimate is
        averaged over.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.

    Notes
    -----
    Not thread-safe against concurrent :meth:`sample` calls from multiple
    threads -- intended for a single monitoring loop per instance, the
    same expectation :class:`~glas.preview.Preview` has for its own
    per-call FPS tracking.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        data_dir: str | Path,
        fps_window: int = DEFAULT_FPS_WINDOW,
    ) -> None:
        self._buffer = buffer
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_dir}")

        self._process = psutil.Process(os.getpid())
        # Primes psutil's internal CPU-time baseline; the first real
        # reading is taken on the first sample() call, comparing against
        # this baseline rather than process start (per psutil's own
        # documented usage pattern for non-blocking cpu_percent()).
        self._process.cpu_percent(interval=None)

        self._recent_samples: deque[tuple[float, int]] = deque(maxlen=fps_window)

    def sample(self) -> PerformanceSnapshot:
        """Take a new performance snapshot.

        Returns
        -------
        PerformanceSnapshot
            Current pipeline and system performance.
        """
        buffer_stats = self._buffer.stats()
        self._recent_samples.append((time.monotonic(), buffer_stats.pushed))

        occupancy_percent = (
            buffer_stats.size / buffer_stats.capacity * 100 if buffer_stats.capacity else 0.0
        )

        memory_info = self._process.memory_info()
        disk_usage = shutil.disk_usage(self._data_dir)
        disk_used_percent = disk_usage.used / disk_usage.total * 100 if disk_usage.total else 0.0

        return PerformanceSnapshot(
            fps=self._fps(),
            buffer_size=buffer_stats.size,
            buffer_capacity=buffer_stats.capacity,
            buffer_occupancy_percent=occupancy_percent,
            dropped_frame_count=buffer_stats.dropped,
            cpu_percent=self._process.cpu_percent(interval=None),
            memory_used_mb=memory_info.rss / 1e6,
            memory_percent=self._process.memory_percent(),
            disk_free_gb=disk_usage.free / 1e9,
            disk_used_percent=disk_used_percent,
        )

    def _fps(self) -> float:
        if len(self._recent_samples) < 2:
            return 0.0
        first_time, first_count = self._recent_samples[0]
        last_time, last_count = self._recent_samples[-1]
        elapsed = last_time - first_time
        if elapsed <= 0:
            return 0.0
        return (last_count - first_count) / elapsed

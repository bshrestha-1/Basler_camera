"""Tests for glas.monitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glas.frame import Frame
from glas.monitor import PerformanceMonitor, PerformanceSnapshot
from glas.ringbuffer import RingBuffer


def _dummy_push() -> Frame:
    return Frame(
        frame_id=0,
        image=np.zeros((2, 2), dtype=np.uint8),
        pixel_format="Mono8",
        host_timestamp_ns=0,
        device_timestamp_ticks=0,
    )


class TestConstruction:
    def test_raises_if_data_dir_does_not_exist(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        with pytest.raises(FileNotFoundError):
            PerformanceMonitor(buffer, tmp_path / "does_not_exist")

    def test_raises_if_data_dir_is_a_file(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        file_path = tmp_path / "a_file"
        file_path.write_text("x")
        with pytest.raises(FileNotFoundError):
            PerformanceMonitor(buffer, file_path)

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        PerformanceMonitor(buffer, str(tmp_path))  # must not raise


class TestBufferMetrics:
    def test_reports_size_and_capacity(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=10)
        buffer.push(_dummy_push())
        buffer.push(_dummy_push())
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.buffer_size == 2
        assert snapshot.buffer_capacity == 10

    def test_reports_occupancy_percent(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        buffer.push(_dummy_push())
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.buffer_occupancy_percent == pytest.approx(25.0)

    def test_reports_zero_occupancy_for_empty_buffer(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.buffer_occupancy_percent == 0.0

    def test_reports_dropped_frame_count(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=2)
        for _ in range(5):
            buffer.push(_dummy_push())
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.dropped_frame_count == 3


class TestFps:
    def test_fps_is_zero_before_two_samples(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=100)
        monitor = PerformanceMonitor(buffer, tmp_path)

        assert monitor.sample().fps == 0.0

    def test_fps_computed_from_pushed_count_over_elapsed_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buffer = RingBuffer(capacity=1000)
        monitor = PerformanceMonitor(buffer, tmp_path)

        clock = iter([0.0, 1.0, 2.0])
        monkeypatch.setattr("glas.monitor.time.monotonic", lambda: next(clock))

        monitor.sample()  # t=0, pushed=0
        for _ in range(50):
            buffer.push(_dummy_push())
        monitor.sample()  # t=1, pushed=50
        for _ in range(50):
            buffer.push(_dummy_push())
        snapshot = monitor.sample()  # t=2, pushed=100

        # fps is averaged over the whole window (t=0 to t=2): 100 pushes / 2s.
        assert snapshot.fps == pytest.approx(50.0)

    def test_fps_window_bounds_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buffer = RingBuffer(capacity=1000)
        monitor = PerformanceMonitor(buffer, tmp_path, fps_window=2)

        clock = iter([0.0, 100.0, 101.0])
        monkeypatch.setattr("glas.monitor.time.monotonic", lambda: next(clock))

        monitor.sample()  # t=0, pushed=0 (falls out of the window=2 history later)
        for _ in range(10):
            buffer.push(_dummy_push())
        monitor.sample()  # t=100, pushed=10
        for _ in range(10):
            buffer.push(_dummy_push())
        snapshot = monitor.sample()  # t=101, pushed=20; window now holds only [t=100, t=101]

        assert snapshot.fps == pytest.approx(10.0)


class TestSystemMetrics:
    def test_cpu_percent_is_a_non_negative_float(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert isinstance(snapshot.cpu_percent, float)
        assert snapshot.cpu_percent >= 0.0

    def test_memory_used_and_percent_are_positive(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.memory_used_mb > 0.0
        assert snapshot.memory_percent > 0.0

    def test_disk_fields_are_within_sane_bounds(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)

        snapshot = monitor.sample()
        assert snapshot.disk_free_gb >= 0.0
        assert 0.0 <= snapshot.disk_used_percent <= 100.0


class TestPerformanceSnapshot:
    def test_is_frozen(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)
        snapshot = monitor.sample()

        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            snapshot.fps = 1.0  # type: ignore[misc]

    def test_returns_a_performance_snapshot(self, tmp_path: Path) -> None:
        buffer = RingBuffer(capacity=4)
        monitor = PerformanceMonitor(buffer, tmp_path)
        assert isinstance(monitor.sample(), PerformanceSnapshot)

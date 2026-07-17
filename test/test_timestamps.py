"""Tests for glas.timestamps."""

from __future__ import annotations

import numpy as np

from glas.frame import Frame
from glas.timestamps import TimestampLog, WallClockReference


def _make_frame(frame_id: int, host_ts: int, device_ts: int = 0) -> Frame:
    return Frame(
        frame_id=frame_id,
        image=np.zeros((2, 2), dtype=np.uint8),
        pixel_format="Mono8",
        host_timestamp_ns=host_ts,
        device_timestamp_ticks=device_ts,
    )


def test_wall_clock_reference_capture_returns_recent_values() -> None:
    import time

    before_perf = time.perf_counter_ns()
    before_wall = time.time_ns()
    ref = WallClockReference.capture()
    after_perf = time.perf_counter_ns()

    assert before_perf <= ref.perf_counter_ns <= after_perf
    assert ref.wall_clock_ns >= before_wall


def test_wall_clock_reference_converts_host_timestamp() -> None:
    ref = WallClockReference(perf_counter_ns=1_000_000, wall_clock_ns=2_000_000_000)
    # A frame 500ns after the reference perf_counter reading should map to
    # 500ns after the reference wall-clock reading.
    assert ref.to_wall_clock_ns(1_000_500) == 2_000_000_500


def test_empty_log_has_zero_length_and_no_gaps() -> None:
    log = TimestampLog()
    assert len(log) == 0
    assert log.frame_id_gaps() == []
    assert log.dropped_frame_count() == 0
    assert log.duration_ns() == 0


def test_append_accumulates_entries() -> None:
    log = TimestampLog()
    log.append(_make_frame(0, host_ts=100, device_ts=10))
    log.append(_make_frame(1, host_ts=200, device_ts=20))

    assert len(log) == 2
    np.testing.assert_array_equal(log.frame_ids(), [0, 1])
    np.testing.assert_array_equal(log.host_timestamps_ns(), [100, 200])
    np.testing.assert_array_equal(log.device_timestamps_ticks(), [10, 20])


def test_no_gaps_for_consecutive_frame_ids() -> None:
    log = TimestampLog()
    for i in range(5):
        log.append(_make_frame(i, host_ts=i * 100))
    assert log.frame_id_gaps() == []
    assert log.dropped_frame_count() == 0


def test_detects_single_gap() -> None:
    log = TimestampLog()
    log.append(_make_frame(0, host_ts=0))
    log.append(_make_frame(1, host_ts=100))
    log.append(_make_frame(5, host_ts=500))  # frames 2, 3, 4 missing
    log.append(_make_frame(6, host_ts=600))

    assert log.frame_id_gaps() == [(2, 4)]
    assert log.dropped_frame_count() == 3


def test_detects_multiple_gaps() -> None:
    log = TimestampLog()
    for frame_id in (0, 2, 3, 7):
        log.append(_make_frame(frame_id, host_ts=frame_id * 10))

    assert log.frame_id_gaps() == [(1, 1), (4, 6)]
    assert log.dropped_frame_count() == 1 + 3


def test_intervals_ns_computes_consecutive_differences() -> None:
    log = TimestampLog()
    for host_ts in (0, 100, 350, 400):
        log.append(_make_frame(len(log), host_ts=host_ts))

    np.testing.assert_array_equal(log.intervals_ns(), [100, 250, 50])


def test_intervals_ns_empty_for_fewer_than_two_frames() -> None:
    log = TimestampLog()
    assert log.intervals_ns().size == 0
    log.append(_make_frame(0, host_ts=0))
    assert log.intervals_ns().size == 0


def test_duration_ns_is_span_between_first_and_last() -> None:
    log = TimestampLog()
    log.append(_make_frame(0, host_ts=1_000))
    log.append(_make_frame(1, host_ts=1_500))
    log.append(_make_frame(2, host_ts=3_000))

    assert log.duration_ns() == 2_000

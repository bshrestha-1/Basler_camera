"""Tests for glas.ringbuffer."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from glas.frame import Frame
from glas.ringbuffer import RingBuffer


def _make_frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        image=np.zeros((2, 2), dtype=np.uint8),
        pixel_format="Mono8",
        host_timestamp_ns=frame_id,
        device_timestamp_ticks=frame_id,
    )


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)
    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_push_and_pop_preserve_order() -> None:
    buffer = RingBuffer(capacity=4)
    for i in range(3):
        buffer.push(_make_frame(i))

    assert len(buffer) == 3
    assert [buffer.pop().frame_id for _ in range(3)] == [0, 1, 2]


def test_push_reports_no_drop_when_not_full() -> None:
    buffer = RingBuffer(capacity=4)
    assert buffer.push(_make_frame(0)) is False


def test_push_drops_oldest_when_full() -> None:
    buffer = RingBuffer(capacity=2)
    buffer.push(_make_frame(0))
    buffer.push(_make_frame(1))
    dropped = buffer.push(_make_frame(2))

    assert dropped is True
    assert len(buffer) == 2
    remaining_ids = [buffer.pop().frame_id, buffer.pop().frame_id]
    assert remaining_ids == [1, 2]


def test_peek_on_empty_buffer_returns_none() -> None:
    buffer = RingBuffer(capacity=4)
    assert buffer.peek() is None


def test_peek_returns_newest_without_removing_it() -> None:
    buffer = RingBuffer(capacity=4)
    buffer.push(_make_frame(0))
    buffer.push(_make_frame(1))

    assert buffer.peek().frame_id == 1
    assert buffer.peek().frame_id == 1  # calling it again doesn't consume anything
    assert len(buffer) == 2


def test_peek_never_competes_with_pop_for_frames() -> None:
    """The whole point of peek(): a consumer using only peek() must never
    cause pop()-based consumers (like a dataset writer) to lose a frame."""
    buffer = RingBuffer(capacity=4)
    for i in range(4):
        buffer.push(_make_frame(i))

    peeked_ids = [buffer.peek().frame_id for _ in range(10)]
    assert all(frame_id == 3 for frame_id in peeked_ids)

    popped_ids = [buffer.pop().frame_id for _ in range(4)]
    assert popped_ids == [0, 1, 2, 3]


def test_pop_on_empty_buffer_without_timeout_returns_none_immediately() -> None:
    buffer = RingBuffer(capacity=4)
    start = time.monotonic()
    result = buffer.pop(timeout=0)
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed < 0.1


def test_pop_respects_timeout_when_buffer_stays_empty() -> None:
    buffer = RingBuffer(capacity=4)
    start = time.monotonic()
    result = buffer.pop(timeout=0.1)
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed >= 0.1


def test_pop_wakes_up_when_a_frame_is_pushed_concurrently() -> None:
    buffer = RingBuffer(capacity=4)
    result: list[Frame | None] = []

    def consumer() -> None:
        result.append(buffer.pop(timeout=2.0))

    thread = threading.Thread(target=consumer)
    thread.start()
    time.sleep(0.05)  # give the consumer time to start waiting
    buffer.push(_make_frame(42))
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result and result[0] is not None
    assert result[0].frame_id == 42


def test_clear_empties_buffer_without_counting_as_dropped() -> None:
    buffer = RingBuffer(capacity=4)
    buffer.push(_make_frame(0))
    buffer.push(_make_frame(1))
    buffer.clear()

    assert len(buffer) == 0
    assert buffer.stats().dropped == 0


def test_stats_reports_capacity_size_and_counters() -> None:
    buffer = RingBuffer(capacity=2)
    buffer.push(_make_frame(0))
    buffer.push(_make_frame(1))
    buffer.push(_make_frame(2))  # drops frame 0
    buffer.pop()

    stats = buffer.stats()
    assert stats.capacity == 2
    assert stats.size == 1
    assert stats.pushed == 3
    assert stats.dropped == 1
    assert stats.popped == 1


def test_concurrent_producer_and_consumer_preserve_total_frame_count() -> None:
    """A capacity large enough that nothing is dropped: every pushed frame
    must eventually be popped exactly once, and in order, even with the
    producer and consumer running on separate threads simultaneously."""
    buffer = RingBuffer(capacity=1000)
    frame_count = 500
    consumed: list[int] = []

    def produce() -> None:
        for i in range(frame_count):
            buffer.push(_make_frame(i))

    def consume() -> None:
        while len(consumed) < frame_count:
            frame = buffer.pop(timeout=2.0)
            if frame is None:
                break
            consumed.append(frame.frame_id)

    producer_thread = threading.Thread(target=produce)
    consumer_thread = threading.Thread(target=consume)
    consumer_thread.start()
    producer_thread.start()
    producer_thread.join(timeout=5.0)
    consumer_thread.join(timeout=5.0)

    assert consumed == list(range(frame_count))
    stats = buffer.stats()
    assert stats.pushed == frame_count
    assert stats.popped == frame_count
    assert stats.dropped == 0

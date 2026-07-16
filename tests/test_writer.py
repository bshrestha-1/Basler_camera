"""Tests for glas.writer."""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from glas.dataset import Dataset, validate_dataset
from glas.exceptions import WriterError
from glas.frame import Frame
from glas.metadata import DatasetMetadata
from glas.ringbuffer import RingBuffer
from glas.writer import DatasetWriter


def _make_metadata(**overrides: object) -> DatasetMetadata:
    # dataset_format defaults to "hdf5" as a placeholder: Dataset.create()
    # always overwrites it with the format it actually resolved to.
    defaults = dict(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=8,
        height=4,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)  # type: ignore[arg-type]


def _make_frame(frame_id: int, width: int = 8, height: int = 4) -> Frame:
    return Frame(
        frame_id=frame_id,
        image=np.full((height, width), frame_id % 256, dtype=np.uint8),
        pixel_format="Mono8",
        host_timestamp_ns=frame_id * 1000,
        device_timestamp_ticks=frame_id,
    )


def test_writer_persists_frames_pushed_to_the_buffer(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=64)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset, poll_timeout=0.05)

    writer.start()
    for i in range(10):
        buffer.push(_make_frame(i))
    time.sleep(0.3)
    writer.stop()

    stats = writer.stats()
    assert stats.frames_written == 10
    assert stats.write_errors == 0
    assert not stats.is_running

    result = validate_dataset(tmp_path)
    assert result.valid, result.errors
    assert result.metadata is not None
    assert result.metadata.frame_count == 10


def test_stop_drains_frames_pushed_just_before_stop(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=64)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset, poll_timeout=0.05)

    writer.start()
    for i in range(20):
        buffer.push(_make_frame(i))
    writer.stop()  # must drain all 20 frames before returning

    assert writer.stats().frames_written == 20
    with h5py.File(tmp_path / "frames.h5", "r") as handle:
        assert len(handle["frame_ids"]) == 20


def test_start_twice_raises(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=8)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset)

    writer.start()
    try:
        with pytest.raises(WriterError):
            writer.start()
    finally:
        writer.stop()


def test_stop_before_start_is_a_no_op(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=8)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset)

    writer.stop()  # must not raise
    assert not writer.is_running


def test_stopping_finalizes_the_dataset_even_with_no_frames(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=8)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset, poll_timeout=0.05)

    writer.start()
    time.sleep(0.1)
    writer.stop()

    assert (tmp_path / "metadata.json").is_file()
    assert writer.stats().frames_written == 0


def test_dropped_frame_count_reflects_ring_buffer_gaps(tmp_path: Path) -> None:
    buffer = RingBuffer(capacity=2)
    dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
    writer = DatasetWriter(buffer, dataset, poll_timeout=0.05)
    writer.start()

    def _wait_until_written(count: int) -> None:
        deadline = time.monotonic() + 2.0
        while writer.stats().frames_written < count and time.monotonic() < deadline:
            time.sleep(0.005)
        assert writer.stats().frames_written == count

    # Establish a clean baseline: frames 0 and 1 are confirmed written
    # before the flood below, so whatever gap the flood produces has a
    # "previous" entry in the writer's timestamp log to be detected
    # against (a gap with nothing recorded before it is invisible to gap
    # detection by construction -- see glas.timestamps).
    buffer.push(_make_frame(0))
    _wait_until_written(1)
    buffer.push(_make_frame(1))
    _wait_until_written(2)

    # Flood far more frames than the buffer can hold, all at once, so
    # the ring buffer is virtually certain to drop some before the
    # writer thread -- doing real disk I/O per frame -- can keep up.
    flood_size = 200
    for i in range(2, 2 + flood_size):
        buffer.push(_make_frame(i))

    writer.stop()

    stats = writer.stats()
    total_pushed = 2 + flood_size
    assert stats.dropped_frame_count > 0
    # Drop-oldest semantics guarantee every pushed frame_id is either
    # written or accounted for as a gap: the most recent flood frame is
    # never evicted (nothing newer ever arrives after it), so it's
    # always eventually written, giving gap detection a valid endpoint
    # on both sides of whatever was dropped in between.
    assert stats.frames_written + stats.dropped_frame_count == total_pushed

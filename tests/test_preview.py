"""Tests for glas.preview."""

from __future__ import annotations

import numpy as np
import pytest

from glas.frame import Frame
from glas.preview import Preview, ZoomRegion, apply_zoom
from glas.ringbuffer import RingBuffer


def _make_frame(
    frame_id: int, host_timestamp_ns: int = 0, size: tuple[int, int] = (10, 20)
) -> Frame:
    height, width = size
    image = np.full((height, width), frame_id % 256, dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        image=image,
        pixel_format="Mono8",
        host_timestamp_ns=host_timestamp_ns,
        device_timestamp_ticks=frame_id,
    )


class TestZoomRegion:
    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError):
            ZoomRegion(x=0, y=0, width=0, height=10)
        with pytest.raises(ValueError):
            ZoomRegion(x=0, y=0, width=10, height=-1)

    def test_rejects_negative_offset(self) -> None:
        with pytest.raises(ValueError):
            ZoomRegion(x=-1, y=0, width=10, height=10)

    def test_clamped_to_leaves_in_bounds_region_unchanged(self) -> None:
        region = ZoomRegion(x=5, y=5, width=10, height=10)
        clamped = region.clamped_to(image_width=100, image_height=100)
        assert clamped == region

    def test_clamped_to_shrinks_region_that_overflows(self) -> None:
        region = ZoomRegion(x=90, y=90, width=50, height=50)
        clamped = region.clamped_to(image_width=100, image_height=100)
        assert clamped.x == 90
        assert clamped.y == 90
        assert clamped.width == 10
        assert clamped.height == 10

    def test_clamped_to_moves_corner_that_is_entirely_outside(self) -> None:
        region = ZoomRegion(x=200, y=200, width=10, height=10)
        clamped = region.clamped_to(image_width=100, image_height=100)
        assert clamped.x == 99
        assert clamped.y == 99
        assert clamped.width == 1
        assert clamped.height == 1

    def test_is_frozen(self) -> None:
        region = ZoomRegion(x=0, y=0, width=10, height=10)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            region.x = 5  # type: ignore[misc]


class TestApplyZoom:
    def test_none_region_returns_image_unchanged(self) -> None:
        image = np.arange(20).reshape(4, 5).astype(np.uint8)
        result = apply_zoom(image, None)
        assert result is image

    def test_crops_to_region(self) -> None:
        image = np.arange(100).reshape(10, 10).astype(np.uint8)
        region = ZoomRegion(x=2, y=3, width=4, height=5)
        result = apply_zoom(image, region)
        assert result.shape == (5, 4)
        assert np.array_equal(result, image[3:8, 2:6])

    def test_clamps_region_larger_than_image(self) -> None:
        image = np.arange(100).reshape(10, 10).astype(np.uint8)
        region = ZoomRegion(x=5, y=5, width=50, height=50)
        result = apply_zoom(image, region)
        assert result.shape == (5, 5)


class TestPreviewUpdate:
    def test_update_on_empty_buffer_returns_none(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        assert preview.update() is None

    def test_update_returns_newest_frame(self) -> None:
        buffer = RingBuffer(capacity=4)
        buffer.push(_make_frame(0))
        buffer.push(_make_frame(1))
        preview = Preview(buffer)

        frame = preview.update()
        assert frame is not None
        assert frame.frame_id == 1

    def test_update_never_pops_from_the_buffer(self) -> None:
        buffer = RingBuffer(capacity=4)
        buffer.push(_make_frame(0))
        preview = Preview(buffer)

        preview.update()
        preview.update()
        preview.update()

        assert len(buffer) == 1
        assert buffer.pop(timeout=0).frame_id == 0


class TestPreviewFps:
    def test_fps_is_zero_before_any_update(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        assert preview.fps() == 0.0

    def test_fps_is_zero_after_a_single_distinct_frame(self) -> None:
        buffer = RingBuffer(capacity=4)
        buffer.push(_make_frame(0, host_timestamp_ns=0))
        preview = Preview(buffer)
        preview.update()
        assert preview.fps() == 0.0

    def test_fps_computed_from_distinct_frame_intervals(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)

        # 10 ms apart -> 100 fps.
        for i in range(5):
            buffer.push(_make_frame(i, host_timestamp_ns=i * 10_000_000))
            preview.update()

        assert preview.fps() == pytest.approx(100.0, rel=1e-6)

    def test_repeated_updates_on_the_same_frame_do_not_affect_fps(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        buffer.push(_make_frame(0, host_timestamp_ns=0))
        buffer.push(_make_frame(1, host_timestamp_ns=10_000_000))
        preview.update()
        first_fps = preview.fps()

        for _ in range(20):
            preview.update()

        assert preview.fps() == first_fps

    def test_fps_window_bounds_history(self) -> None:
        buffer = RingBuffer(capacity=100)
        preview = Preview(buffer, fps_window=3)

        # Uneven early spacing, then a steady 10 ms cadence for the last
        # few frames -- fps() should reflect only the windowed frames.
        timestamps_ns = [0, 5_000_000, 200_000_000, 210_000_000, 220_000_000]
        for i, ts in enumerate(timestamps_ns):
            buffer.push(_make_frame(i, host_timestamp_ns=ts))
            preview.update()

        assert preview.fps() == pytest.approx(100.0, rel=1e-6)


class TestPreviewZoomTo:
    def test_zoom_to_rejects_non_positive_factor(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        with pytest.raises(ValueError):
            preview.zoom_to(0, center=(50, 50), source_width=100, source_height=100)

    def test_zoom_to_centers_region_at_requested_factor(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        preview.zoom_to(2.0, center=(50, 50), source_width=100, source_height=100)

        assert preview.zoom is not None
        assert preview.zoom.width == 50
        assert preview.zoom.height == 50
        assert preview.zoom.x == 25
        assert preview.zoom.y == 25

    def test_zoom_to_clamps_near_the_image_edge(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        preview.zoom_to(4.0, center=(0, 0), source_width=100, source_height=100)

        assert preview.zoom is not None
        assert preview.zoom.x == 0
        assert preview.zoom.y == 0

    def test_reset_zoom_clears_zoom(self) -> None:
        buffer = RingBuffer(capacity=4)
        preview = Preview(buffer)
        preview.zoom_to(2.0, center=(50, 50), source_width=100, source_height=100)
        preview.reset_zoom()
        assert preview.zoom is None


class TestPreviewHistogram:
    def test_histogram_counts_all_pixels(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        frame = Frame(
            frame_id=0,
            image=image,
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        counts = Preview.histogram(frame, bins=256)
        assert counts.sum() == image.size
        assert counts[0] == image.size

    def test_histogram_default_bin_count(self) -> None:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        frame = Frame(
            frame_id=0,
            image=image,
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        counts = Preview.histogram(frame)
        assert counts.shape == (256,)
        assert counts.dtype == np.int64

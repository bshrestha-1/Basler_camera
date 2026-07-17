"""Tests for glas.display.

The rendering functions (render_frame, render_histogram) are pure and
fully exercised here without a display. PreviewWindow's actual
cv2.imshow/waitKey calls are not exercised (they require a display this
test environment does not have) -- what *is* tested, deterministically,
is that PreviewWindow raises DisplayError immediately rather than ever
attempting (and hanging on) those calls, since DISPLAY/WAYLAND_DISPLAY is
guaranteed unset in this sandbox.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from glas.camera_validator import ROI
from glas.display import (
    PreviewWindow,
    _display_available,
    _to_bgr_uint8,
    render_frame,
    render_histogram,
)
from glas.exceptions import DisplayError
from glas.frame import Frame
from glas.preview import Preview, ZoomRegion
from glas.ringbuffer import RingBuffer


def _make_frame(size: tuple[int, int] = (10, 20), dtype: type = np.uint8) -> Frame:
    height, width = size
    image = np.arange(height * width, dtype=dtype).reshape(height, width)
    return Frame(
        frame_id=0,
        image=image,
        pixel_format="Mono8",
        host_timestamp_ns=0,
        device_timestamp_ticks=0,
    )


class TestDisplayAvailable:
    def test_returns_false_when_no_display_env_vars_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _display_available() is False

    def test_returns_true_when_display_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        assert _display_available() is True

    def test_returns_true_when_wayland_display_env_var_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert _display_available() is True

    def test_always_true_on_non_linux_platforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _display_available() is True

    def test_no_display_in_this_sandbox_by_default(self) -> None:
        """Sanity check the assumption every DisplayError test below relies on."""
        assert not os.environ.get("DISPLAY")
        assert not os.environ.get("WAYLAND_DISPLAY")


class TestToBgrUint8:
    def test_uint8_mono_converts_to_three_channel(self) -> None:
        image = np.zeros((4, 5), dtype=np.uint8)
        result = _to_bgr_uint8(image)
        assert result.shape == (4, 5, 3)
        assert result.dtype == np.uint8

    def test_non_uint8_is_scaled_into_range(self) -> None:
        image = np.full((2, 2), np.iinfo(np.uint16).max, dtype=np.uint16)
        result = _to_bgr_uint8(image)
        assert result.dtype == np.uint8
        assert result.max() == 255

    def test_already_color_image_passes_through_shape(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        result = _to_bgr_uint8(image)
        assert result.shape == (4, 5, 3)


class TestRenderFrame:
    def test_output_shape_matches_frame_when_unzoomed(self) -> None:
        frame = _make_frame(size=(10, 20))
        image = render_frame(frame)
        assert image.shape == (10, 20, 3)
        assert image.dtype == np.uint8

    def test_output_shape_reflects_zoom_crop(self) -> None:
        frame = _make_frame(size=(10, 20))
        zoom = ZoomRegion(x=2, y=2, width=5, height=4)
        image = render_frame(frame, zoom=zoom)
        assert image.shape == (4, 5, 3)

    def test_does_not_mutate_the_source_frame(self) -> None:
        frame = _make_frame(size=(10, 20))
        original = frame.image.copy()
        render_frame(frame, crosshair=True, crosshair_position=(5, 5))
        assert np.array_equal(frame.image, original)

    def test_crosshair_draws_green_pixels(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((20, 20), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, crosshair=True, crosshair_position=(10, 10))
        assert tuple(image[10, 10]) == (0, 255, 0)
        assert tuple(image[0, 10]) == (0, 255, 0)

    def test_crosshair_without_position_draws_nothing_extra(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((20, 20), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, crosshair=True, crosshair_position=None)
        assert image.sum() == 0

    def test_roi_draws_red_rectangle_border(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((20, 20), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        roi = ROI(width=10, height=10, offset_x=5, offset_y=5)
        image = render_frame(frame, roi=roi)
        assert tuple(image[5, 5]) == (0, 0, 255)

    def test_fps_text_draws_non_background_pixels(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((30, 100), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, fps=30.0)
        assert image.sum() > 0

    def test_no_overlays_by_default(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((20, 20), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame)
        assert image.sum() == 0

    def test_overlay_grid_draws_lines(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((120, 120), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, overlay_grid=True)
        assert image.sum() > 0
        assert tuple(image[0, 0]) == (80, 80, 80)

    def test_overlay_grid_off_by_default(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((120, 120), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame)
        assert image.sum() == 0

    def test_timestamp_text_draws_non_background_pixels(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((30, 150), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, timestamp_text="12:34:56.789")
        assert image.sum() > 0

    def test_timestamp_text_none_draws_nothing(self) -> None:
        frame = Frame(
            frame_id=0,
            image=np.zeros((30, 150), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        image = render_frame(frame, timestamp_text=None)
        assert image.sum() == 0


class TestRenderHistogram:
    def test_output_shape_matches_requested_dimensions(self) -> None:
        counts = np.ones(256, dtype=np.int64)
        image = render_histogram(counts, width=256, height=100)
        assert image.shape == (100, 256)
        assert image.dtype == np.uint8

    def test_tallest_bar_reaches_the_top_row(self) -> None:
        counts = np.zeros(10, dtype=np.int64)
        counts[5] = 100
        image = render_histogram(counts, width=10, height=50)
        assert image[0, 5] == 255

    def test_empty_bin_column_stays_background(self) -> None:
        counts = np.zeros(10, dtype=np.int64)
        counts[5] = 100
        image = render_histogram(counts, width=10, height=50)
        assert image[0, 0] == 0

    def test_all_zero_counts_produce_blank_image(self) -> None:
        counts = np.zeros(256, dtype=np.int64)
        image = render_histogram(counts)
        assert image.sum() == 0

    def test_resamples_when_bin_count_differs_from_width(self) -> None:
        counts = np.ones(16, dtype=np.int64)
        image = render_histogram(counts, width=256, height=100)
        assert image.shape == (100, 256)


class TestPreviewWindowRaisesWithoutDisplay:
    def _make_window(self) -> PreviewWindow:
        buffer = RingBuffer(capacity=4)
        buffer.push(
            Frame(
                frame_id=0,
                image=np.zeros((10, 10), dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=0,
                device_timestamp_ticks=0,
            )
        )
        preview = Preview(buffer)
        return PreviewWindow(preview)

    def test_show_once_raises_display_error(self) -> None:
        window = self._make_window()
        with pytest.raises(DisplayError):
            window.show_once()

    def test_wait_key_raises_display_error(self) -> None:
        window = self._make_window()
        with pytest.raises(DisplayError):
            window.wait_key()

    def test_run_raises_display_error_and_does_not_hang(self) -> None:
        window = self._make_window()
        with pytest.raises(DisplayError):
            window.run()

    def test_close_without_ever_opening_is_a_no_op(self) -> None:
        window = self._make_window()
        window.close()  # must not raise

    def test_context_manager_closes_on_exit_without_raising(self) -> None:
        with self._make_window() as window:
            assert isinstance(window, PreviewWindow)

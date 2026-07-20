"""Tests for glas.gui.widgets.live_preview_widget."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.frame import Frame
from glas.gui.viewmodels.live_feed_viewmodel import LiveFeedViewModel
from glas.gui.widgets.live_preview_widget import (
    _EMPTY_STATE_PAGE,
    _LIVE_VIEW_PAGE,
    LivePreviewWidget,
    _format_wall_clock_text,
)
from glas.ringbuffer import RingBuffer


def _make_frame(frame_id: int, host_timestamp_ns: int = 0) -> Frame:
    image = np.random.randint(0, 255, (60, 80), dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        image=image,
        pixel_format="Mono8",
        host_timestamp_ns=host_timestamp_ns,
        device_timestamp_ticks=frame_id,
    )


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def widget(qapp: QApplication) -> LivePreviewWidget:
    vm = LiveFeedViewModel()
    return LivePreviewWidget(vm)


@pytest.fixture
def attached_widget(widget: LivePreviewWidget) -> LivePreviewWidget:
    buffer = RingBuffer(capacity=4)
    widget._view_model.attach(buffer)
    widget._buffer = buffer  # type: ignore[attr-defined]
    yield widget
    widget._view_model.detach()


class TestFormatWallClockText:
    def test_returns_hh_mm_ss_millis_format(self) -> None:
        text = _format_wall_clock_text(1_700_000_000_000_000_000)
        assert len(text.split(":")) == 3
        assert "." in text


class TestFrameRendering:
    def test_frame_ready_populates_pixmap_and_labels(
        self, attached_widget: LivePreviewWidget
    ) -> None:
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0))
        attached_widget._view_model._poll()

        assert attached_widget._pixmap_item.pixmap().isNull() is False
        assert attached_widget._frame_label.text() == "Frame: 0"
        assert attached_widget._pixmap_item.pixmap().width() == 80
        assert attached_widget._pixmap_item.pixmap().height() == 60

    def test_second_frame_updates_frame_counter(self, attached_widget: LivePreviewWidget) -> None:
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0))
        attached_widget._view_model._poll()
        buffer.push(_make_frame(1))
        attached_widget._view_model._poll()

        assert attached_widget._frame_label.text() == "Frame: 1"


class TestEmptyState:
    def test_shows_empty_state_before_any_frame(self, widget: LivePreviewWidget) -> None:
        assert widget._view_stack.currentIndex() == _EMPTY_STATE_PAGE
        assert widget._empty_state_title_label.text() == "No Camera Connected"
        assert "Connect" in widget._empty_state_subtitle_label.text()

    def test_first_frame_switches_to_live_view(self, attached_widget: LivePreviewWidget) -> None:
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0))
        attached_widget._view_model._poll()
        assert attached_widget._view_stack.currentIndex() == _LIVE_VIEW_PAGE

    def test_reset_returns_to_empty_state(self, attached_widget: LivePreviewWidget) -> None:
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0))
        attached_widget._view_model._poll()
        assert attached_widget._view_stack.currentIndex() == _LIVE_VIEW_PAGE

        attached_widget.reset()

        assert attached_widget._view_stack.currentIndex() == _EMPTY_STATE_PAGE
        assert attached_widget._pixmap_item.pixmap().isNull() is True
        assert attached_widget._frame_label.text() == "Frame: --"
        assert attached_widget._fps_label.text() == "FPS: --"

    def test_frame_after_reset_switches_back_to_live_view(
        self, attached_widget: LivePreviewWidget
    ) -> None:
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0))
        attached_widget._view_model._poll()
        attached_widget.reset()

        buffer.push(_make_frame(1))
        attached_widget._view_model._poll()

        assert attached_widget._view_stack.currentIndex() == _LIVE_VIEW_PAGE
        assert attached_widget._frame_label.text() == "Frame: 1"


class TestZoomControls:
    def test_zoom_in_increases_reported_percent(self, widget: LivePreviewWidget) -> None:
        widget.zoom_in()
        assert widget._zoom_percent > 100.0
        assert "%" in widget._zoom_label.text()

    def test_zoom_out_decreases_reported_percent(self, widget: LivePreviewWidget) -> None:
        widget.zoom_out()
        assert widget._zoom_percent < 100.0

    def test_full_resolution_resets_to_100_percent(self, widget: LivePreviewWidget) -> None:
        widget.zoom_in()
        widget.full_resolution()
        assert widget._zoom_percent == pytest.approx(100.0)

    def test_fit_to_window_noop_before_any_frame(self, widget: LivePreviewWidget) -> None:
        widget.fit_to_window()
        assert widget._pixmap_item.pixmap().isNull() is True


class TestInteractionModes:
    def test_crosshair_click_sets_preview_state(self, attached_widget: LivePreviewWidget) -> None:
        attached_widget._set_interaction_mode("crosshair")
        attached_widget._on_point_clicked(12.0, 34.0)

        assert attached_widget.preview.crosshair is True
        assert attached_widget.preview.crosshair_position == (12, 34)

    def test_switching_away_from_crosshair_hides_it(
        self, attached_widget: LivePreviewWidget
    ) -> None:
        attached_widget._set_interaction_mode("crosshair")
        attached_widget._on_point_clicked(12.0, 34.0)
        attached_widget._set_interaction_mode("pan")

        assert attached_widget.preview.crosshair is False

    def test_roi_selection_sets_roi_and_emits_signal(
        self, attached_widget: LivePreviewWidget, qtbot
    ) -> None:
        attached_widget._set_interaction_mode("roi")
        with qtbot.waitSignal(attached_widget.roi_selected, timeout=2000) as blocker:
            attached_widget._on_region_selected(5.0, 5.0, 20.0, 15.0)

        roi = blocker.args[0]
        assert roi.offset_x == 5
        assert roi.offset_y == 5
        assert roi.width == 20
        assert roi.height == 15
        assert attached_widget.preview.show_roi is True

    def test_switching_away_from_roi_hides_it(self, attached_widget: LivePreviewWidget) -> None:
        attached_widget._set_interaction_mode("roi")
        attached_widget._on_region_selected(5.0, 5.0, 20.0, 15.0)
        attached_widget._set_interaction_mode("pan")

        assert attached_widget.preview.show_roi is False


class TestOverlayToggles:
    def test_grid_toggle_sets_preview_flag(self, attached_widget: LivePreviewWidget) -> None:
        attached_widget._on_grid_toggled(True)
        assert attached_widget.preview.overlay_grid is True
        attached_widget._on_grid_toggled(False)
        assert attached_widget.preview.overlay_grid is False

    def test_timestamp_toggle_causes_wall_clock_capture_on_next_frame(
        self, attached_widget: LivePreviewWidget
    ) -> None:
        attached_widget._on_timestamp_toggled(True)
        buffer: RingBuffer = attached_widget._buffer  # type: ignore[attr-defined]
        buffer.push(_make_frame(0, host_timestamp_ns=1_000_000))
        attached_widget._view_model._poll()

        assert attached_widget._wall_clock_ref is not None


class TestPreviewProperty:
    def test_raises_before_attach(self, widget: LivePreviewWidget) -> None:
        with pytest.raises(RuntimeError):
            _ = widget.preview

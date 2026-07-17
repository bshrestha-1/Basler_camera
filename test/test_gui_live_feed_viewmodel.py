"""Tests for glas.gui.viewmodels.live_feed_viewmodel."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.frame import Frame
from glas.gui.viewmodels.live_feed_viewmodel import LiveFeedViewModel
from glas.ringbuffer import RingBuffer


def _make_frame(frame_id: int) -> Frame:
    image = np.full((10, 20), frame_id % 256, dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        image=image,
        pixel_format="Mono8",
        host_timestamp_ns=frame_id,
        device_timestamp_ticks=frame_id,
    )


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestAttachDetach:
    def test_not_attached_before_attach(self, qapp: QApplication) -> None:
        vm = LiveFeedViewModel()
        assert vm.is_attached is False
        assert vm.preview is None

    def test_attach_starts_polling(self, qapp: QApplication) -> None:
        vm = LiveFeedViewModel()
        buffer = RingBuffer(capacity=4)
        vm.attach(buffer)
        assert vm.is_attached is True
        assert vm.preview is not None
        vm.detach()

    def test_detach_stops_polling(self, qapp: QApplication) -> None:
        vm = LiveFeedViewModel()
        buffer = RingBuffer(capacity=4)
        vm.attach(buffer)
        vm.detach()
        assert vm.is_attached is False
        assert vm.preview is None


class TestFrameReady:
    def test_new_frame_in_buffer_emits_frame_ready(self, qapp: QApplication, qtbot) -> None:
        vm = LiveFeedViewModel()
        buffer = RingBuffer(capacity=4)
        buffer.push(_make_frame(0))
        vm.attach(buffer)
        try:
            with qtbot.waitSignal(vm.frame_ready, timeout=10000) as blocker:
                pass
            assert isinstance(blocker.args[0], Frame)
            assert blocker.args[0].frame_id == 0
        finally:
            vm.detach()

    def test_empty_buffer_does_not_emit(self, qapp: QApplication, qtbot) -> None:
        vm = LiveFeedViewModel()
        buffer = RingBuffer(capacity=4)
        vm.attach(buffer)
        try:
            with qtbot.assertNotEmitted(vm.frame_ready, wait=200):
                pass
        finally:
            vm.detach()

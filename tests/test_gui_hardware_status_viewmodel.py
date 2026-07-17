"""Tests for glas.gui.viewmodels.hardware_status_viewmodel."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.gui.viewmodels.hardware_status_viewmodel import (
    DeviceStatus,
    HardwareStatusViewModel,
)
from glas.monitor import PerformanceSnapshot
from glas.ringbuffer import RingBuffer


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestDeviceStatusModel:
    def test_defaults_and_frozen(self) -> None:
        status = DeviceStatus(name="LabJack T7", connected=True)
        assert status.detail == ""
        with pytest.raises(Exception):  # noqa: B017,PT011 - pydantic ValidationError on frozen set
            status.connected = False  # type: ignore[misc]


class TestAttachDetach:
    def test_status_updated_carries_snapshot_after_attach(
        self, qapp: QApplication, qtbot, tmp_path: Path
    ) -> None:
        vm = HardwareStatusViewModel()
        buffer = RingBuffer(capacity=4)
        try:
            with qtbot.waitSignal(vm.status_updated, timeout=2000) as blocker:
                vm.attach(buffer, str(tmp_path))
            snapshot, devices = blocker.args
            assert isinstance(snapshot, PerformanceSnapshot)
            assert devices == {}
        finally:
            vm.detach()

    def test_detach_stops_further_polling(self, qapp: QApplication, qtbot, tmp_path: Path) -> None:
        vm = HardwareStatusViewModel()
        buffer = RingBuffer(capacity=4)
        vm.attach(buffer, str(tmp_path))
        vm.detach()
        with qtbot.assertNotEmitted(vm.status_updated, wait=1200):
            pass


class TestDeviceRegistry:
    def test_register_device_emits_snapshot_none_before_attach(
        self, qapp: QApplication, qtbot
    ) -> None:
        vm = HardwareStatusViewModel()
        with qtbot.waitSignal(vm.status_updated, timeout=2000) as blocker:
            vm.register_device(
                DeviceStatus(name="LabJack T7", connected=True, detail="192.168.1.5")
            )
        snapshot, devices = blocker.args
        assert snapshot is None
        assert devices == {
            "LabJack T7": DeviceStatus(name="LabJack T7", connected=True, detail="192.168.1.5")
        }

    def test_unregister_device_removes_it(self, qapp: QApplication, qtbot) -> None:
        vm = HardwareStatusViewModel()
        vm.register_device(DeviceStatus(name="LabJack T7", connected=True))
        with qtbot.waitSignal(vm.status_updated, timeout=2000) as blocker:
            vm.unregister_device("LabJack T7")
        _snapshot, devices = blocker.args
        assert devices == {}

"""Tests for glas.gui.viewmodels.analysis_viewmodel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.analysis.packing import PackingSummary
from glas.dataset import Dataset
from glas.frame import Frame
from glas.gui.viewmodels.analysis_viewmodel import AnalysisViewModel
from glas.metadata import DatasetMetadata


def _make_packable_dataset(tmp_path: Path, *, frame_count: int = 2, size: int = 64) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=size,
        height=size,
        created_at_utc="2026-07-16T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(image, (20, 20), 6, 255, -1)
        cv2.circle(image, (44, 44), 6, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 10_000_000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestRunPacking:
    def test_success_emits_started_then_finished_with_summary(
        self, qapp: QApplication, qtbot, tmp_path: Path
    ) -> None:
        folder = _make_packable_dataset(tmp_path)
        vm = AnalysisViewModel()

        started_kinds: list[str] = []
        vm.analysis_started.connect(started_kinds.append)

        with qtbot.waitSignal(vm.analysis_finished, timeout=10000) as finished:
            vm.run_packing(folder)

        assert started_kinds == ["packing"]
        kind, result = finished.args
        assert kind == "packing"
        assert isinstance(result, PackingSummary)
        assert vm.is_running is False

    def test_missing_dataset_emits_analysis_failed(
        self, qapp: QApplication, qtbot, tmp_path: Path
    ) -> None:
        vm = AnalysisViewModel()
        missing_folder = tmp_path / "does-not-exist"

        with qtbot.waitSignal(vm.analysis_failed, timeout=5000) as blocker:
            vm.run_packing(missing_folder)

        kind, message = blocker.args
        assert kind == "packing"
        assert isinstance(message, str) and message
        assert vm.is_running is False


class TestConcurrencyGuard:
    def test_second_call_while_running_is_rejected_immediately(
        self, qapp: QApplication, qtbot, tmp_path: Path
    ) -> None:
        folder = _make_packable_dataset(tmp_path)
        vm = AnalysisViewModel()

        received: list[tuple[str, str]] = []
        vm.analysis_failed.connect(lambda kind, msg: received.append((kind, msg)))

        vm.run_packing(folder)
        assert vm.is_running is True
        vm.run_packing(folder)  # rejected synchronously, no thread started

        assert received == [("packing", "Another analysis is already running.")]

        with qtbot.waitSignal(vm.analysis_finished, timeout=10000):
            pass
        assert vm.is_running is False

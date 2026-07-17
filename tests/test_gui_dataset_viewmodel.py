"""Tests for glas.gui.viewmodels.dataset_viewmodel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.dataset import Dataset
from glas.experiment import ExperimentManager, ExperimentSummary, build_experiment_extra
from glas.export import ExportResult
from glas.frame import Frame
from glas.gui.viewmodels.dataset_viewmodel import DatasetViewModel
from glas.metadata import DatasetMetadata


def _make_experiment(manager: ExperimentManager, *, name: str = "") -> Path:
    folder = manager.new_folder()
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=4,
        height=4,
        created_at_utc="2026-07-13T00:00:00+00:00",
        extra=build_experiment_extra(name=name),
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    dataset.append_frame(
        Frame(
            frame_id=0,
            image=np.zeros((4, 4), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
    )
    dataset.finalize()
    return folder


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def vm(qapp: QApplication, tmp_path: Path) -> DatasetViewModel:
    return DatasetViewModel(tmp_path)


class TestRefresh:
    def test_emits_experiments_changed_with_created_experiments(
        self, vm: DatasetViewModel, qtbot
    ) -> None:
        _make_experiment(vm.manager, name="run-a")
        with qtbot.waitSignal(vm.experiments_changed, timeout=2000) as blocker:
            vm.refresh()
        (experiments,) = blocker.args
        assert len(experiments) == 1
        assert isinstance(experiments[0], ExperimentSummary)

    def test_name_filter_narrows_results(self, vm: DatasetViewModel, qtbot) -> None:
        _make_experiment(vm.manager, name="alpha")
        _make_experiment(vm.manager, name="beta")
        with qtbot.waitSignal(vm.experiments_changed, timeout=2000) as blocker:
            vm.refresh(name_contains="alpha")
        (experiments,) = blocker.args
        assert len(experiments) == 1


class TestDeleteDuplicate:
    def test_delete_removes_experiment_and_refreshes(self, vm: DatasetViewModel, qtbot) -> None:
        folder = _make_experiment(vm.manager, name="to-delete")
        summary = vm.manager.get_experiment(folder.name)
        with qtbot.waitSignal(vm.experiments_changed, timeout=2000) as blocker:
            vm.delete(summary.run_id)
        (experiments,) = blocker.args
        assert experiments == []

    def test_delete_unknown_run_id_emits_error(self, vm: DatasetViewModel, qtbot) -> None:
        with qtbot.waitSignal(vm.error_occurred, timeout=2000):
            vm.delete("not-a-real-run-id")

    def test_duplicate_returns_new_summary_and_refreshes(self, vm: DatasetViewModel, qtbot) -> None:
        folder = _make_experiment(vm.manager, name="original")
        summary = vm.manager.get_experiment(folder.name)
        with qtbot.waitSignal(vm.experiments_changed, timeout=2000):
            copy = vm.duplicate(summary.run_id, new_name="copy")
        assert copy is not None
        assert copy.run_id != summary.run_id


class TestExport:
    def test_export_tiff_emits_export_finished(
        self, vm: DatasetViewModel, qtbot, tmp_path: Path
    ) -> None:
        folder = _make_experiment(vm.manager, name="to-export")
        summary = vm.manager.get_experiment(folder.name)
        output_dir = tmp_path / "exported"
        with qtbot.waitSignal(vm.export_finished, timeout=5000) as blocker:
            vm.export(summary.run_id, output_dir, "tiff")
        (result,) = blocker.args
        assert isinstance(result, ExportResult)

    def test_export_unknown_run_id_emits_error(
        self, vm: DatasetViewModel, qtbot, tmp_path: Path
    ) -> None:
        with qtbot.waitSignal(vm.error_occurred, timeout=2000):
            vm.export("not-a-real-run-id", tmp_path / "out", "tiff")

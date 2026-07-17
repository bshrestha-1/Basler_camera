"""Tests for glas.gui.widgets.dataset_browser_widget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from glas.dataset import Dataset
from glas.experiment import (
    ExperimentManager,
    PhysicalParameters,
    build_experiment_extra,
    build_physical_parameters_extra,
)
from glas.frame import Frame
from glas.gui.viewmodels.dataset_viewmodel import DatasetViewModel
from glas.gui.widgets.dataset_browser_widget import DatasetBrowserWidget
from glas.metadata import DatasetMetadata


def _make_experiment(manager: ExperimentManager, *, name: str = "", material: str = "") -> Path:
    folder = manager.new_folder()
    extra = build_experiment_extra(name=name, tags=["a", "b"])
    if material:
        extra = build_physical_parameters_extra(PhysicalParameters(material=material), extra=extra)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=8,
        height=8,
        created_at_utc="2026-07-16T00:00:00+00:00",
        extra=extra,
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    dataset.append_frame(
        Frame(
            frame_id=0,
            image=np.zeros((8, 8), dtype=np.uint8),
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


class TestInitialLoad:
    def test_table_populated_with_existing_experiments(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="alpha")
        _make_experiment(vm.manager, name="beta")

        widget = DatasetBrowserWidget(vm)
        assert widget._table.rowCount() == 2

    def test_selection_actions_disabled_with_no_selection(self, vm: DatasetViewModel) -> None:
        widget = DatasetBrowserWidget(vm)
        assert widget._export_button.isEnabled() is False
        assert widget._duplicate_button.isEnabled() is False
        assert widget._delete_button.isEnabled() is False


class TestSelection:
    def test_selecting_a_row_populates_detail_labels(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="alpha", material="glass beads")
        widget = DatasetBrowserWidget(vm)

        widget._table.selectRow(0)

        assert widget._run_id_label.text() == "Run0001"
        assert widget._name_label.text() == "alpha"
        assert "glass beads" in widget._physical_params_label.text()
        assert widget._thumbnail_label.pixmap().isNull() is False
        assert widget._export_button.isEnabled() is True

    def test_no_physical_parameters_shows_placeholder(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="beta")
        widget = DatasetBrowserWidget(vm)

        widget._table.selectRow(0)

        assert widget._physical_params_label.text() == "--"


class TestSearch:
    def test_name_filter_narrows_table(self, qapp: QApplication, tmp_path: Path) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="alpha")
        _make_experiment(vm.manager, name="beta")
        widget = DatasetBrowserWidget(vm)

        widget._name_filter_edit.setText("alpha")
        widget.refresh()

        assert widget._table.rowCount() == 1


class TestDeleteAndDuplicate:
    def test_delete_removes_row(self, qapp: QApplication, tmp_path: Path, monkeypatch) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="to-delete")
        widget = DatasetBrowserWidget(vm)
        widget._table.selectRow(0)

        monkeypatch.setattr(
            "glas.gui.widgets.dataset_browser_widget.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )
        widget._on_delete_clicked()

        assert widget._table.rowCount() == 0

    def test_delete_cancelled_keeps_row(
        self, qapp: QApplication, tmp_path: Path, monkeypatch
    ) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="keep-me")
        widget = DatasetBrowserWidget(vm)
        widget._table.selectRow(0)

        monkeypatch.setattr(
            "glas.gui.widgets.dataset_browser_widget.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.No,
        )
        widget._on_delete_clicked()

        assert widget._table.rowCount() == 1

    def test_duplicate_adds_a_row(self, qapp: QApplication, tmp_path: Path) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="original")
        widget = DatasetBrowserWidget(vm)
        widget._table.selectRow(0)

        widget._on_duplicate_clicked()

        assert widget._table.rowCount() == 2


class TestExport:
    def test_export_tiff_shows_completion_dialog(
        self, qapp: QApplication, tmp_path: Path, qtbot, monkeypatch
    ) -> None:
        vm = DatasetViewModel(tmp_path)
        _make_experiment(vm.manager, name="to-export")
        widget = DatasetBrowserWidget(vm)
        widget._table.selectRow(0)

        output_dir = tmp_path / "exported"
        monkeypatch.setattr(
            "glas.gui.widgets.dataset_browser_widget.QFileDialog.getExistingDirectory",
            lambda *args, **kwargs: str(output_dir),
        )
        shown = {}
        monkeypatch.setattr(
            "glas.gui.widgets.dataset_browser_widget.QMessageBox.information",
            lambda *args, **kwargs: shown.setdefault("called", True),
        )

        with qtbot.waitSignal(vm.export_finished, timeout=5000):
            widget._on_export_clicked()

        assert shown.get("called") is True

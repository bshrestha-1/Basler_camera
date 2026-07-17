"""Tests for glas.gui.widgets.analysis_panel_widget."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.dataset import Dataset
from glas.frame import Frame
from glas.gui.viewmodels.analysis_viewmodel import AnalysisViewModel
from glas.gui.widgets.analysis_panel_widget import AnalysisPanelWidget
from glas.metadata import DatasetMetadata


def _make_packable_dataset(tmp_path: Path, *, frame_count: int = 3, size: int = 100) -> Path:
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
        cv2.circle(image, (30, 30), 6, 255, -1)
        cv2.circle(image, (70, 70), 6, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 100_000_000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def widget(qapp: QApplication) -> AnalysisPanelWidget:
    vm = AnalysisViewModel()
    return AnalysisPanelWidget(vm)


class TestTabStructure:
    def test_has_nine_tabs(self, widget: AnalysisPanelWidget) -> None:
        assert widget._tabs.count() == 9

    def test_tab_titles_match_expected_order(self, widget: AnalysisPanelWidget) -> None:
        titles = [widget._tabs.tabText(i) for i in range(widget._tabs.count())]
        assert titles == [
            "Tracking",
            "Detection (YOLO)",
            "Brazil Nut",
            "Convection",
            "Packing",
            "Segregation",
            "Segmentation (SAM2)",
            "Vibration",
            "Histograms",
        ]

    def test_only_histograms_tab_is_disabled(self, widget: AnalysisPanelWidget) -> None:
        for i in range(widget._tabs.count() - 1):
            assert widget._tabs.isTabEnabled(i) is True
        assert widget._tabs.isTabEnabled(widget._tabs.count() - 1) is False

    def test_tracking_tab_has_no_export_button(self, widget: AnalysisPanelWidget) -> None:
        tracking_tab = widget._analysis_tabs["tracking"]
        assert tracking_tab._export_button.isVisible() is False

    def test_detection_tab_has_extra_weights_field(self, widget: AnalysisPanelWidget) -> None:
        detection_tab = widget._analysis_tabs["detection"]
        assert detection_tab._extra_field_edit is not None
        assert detection_tab._extra_field_edit.text() == ""

    def test_segmentation_tab_has_default_model_id(self, widget: AnalysisPanelWidget) -> None:
        from glas.gui.viewmodels.analysis_viewmodel import DEFAULT_SAM2_MODEL_ID

        segmentation_tab = widget._analysis_tabs["segmentation"]
        assert segmentation_tab._extra_field_edit is not None
        assert segmentation_tab._extra_field_edit.text() == DEFAULT_SAM2_MODEL_ID

    def test_non_ai_tabs_have_no_extra_field(self, widget: AnalysisPanelWidget) -> None:
        assert widget._analysis_tabs["tracking"]._extra_field_edit is None
        assert widget._analysis_tabs["packing"]._extra_field_edit is None


class TestRunningPacking:
    def test_run_populates_result_and_enables_export(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot
    ) -> None:
        folder = _make_packable_dataset(tmp_path)
        tab = widget._analysis_tabs["packing"]
        tab._path_edit.setText(str(folder))

        with qtbot.waitSignal(widget._view_model.analysis_finished, timeout=10000):
            tab._on_run_clicked()

        assert tab._status_label.text() == "Done"
        assert "Frames: 3" in tab._result_text.toPlainText()
        assert tab._export_button.isEnabled() is True

    def test_export_plot_writes_a_file(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot
    ) -> None:
        folder = _make_packable_dataset(tmp_path)
        tab = widget._analysis_tabs["packing"]
        tab._path_edit.setText(str(folder))

        with qtbot.waitSignal(widget._view_model.analysis_finished, timeout=10000):
            tab._on_run_clicked()

        output_path = tmp_path / "packing_summary.png"
        tab._export_plot(tab._last_result, output_path)
        assert output_path.exists()

    def test_run_with_missing_dataset_shows_error(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot
    ) -> None:
        tab = widget._analysis_tabs["packing"]
        tab._path_edit.setText(str(tmp_path / "does-not-exist"))

        with qtbot.waitSignal(widget._view_model.analysis_failed, timeout=5000):
            tab._on_run_clicked()

        assert tab._status_label.text().startswith("Error")
        assert tab._export_button.isEnabled() is False

    def test_run_disables_all_other_run_buttons(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot
    ) -> None:
        folder = _make_packable_dataset(tmp_path)
        packing_tab = widget._analysis_tabs["packing"]
        segregation_tab = widget._analysis_tabs["segregation"]
        packing_tab._path_edit.setText(str(folder))

        with qtbot.waitSignal(widget._view_model.analysis_started, timeout=5000):
            packing_tab._on_run_clicked()
        assert segregation_tab._run_button.isEnabled() is False

        with qtbot.waitSignal(widget._view_model.analysis_finished, timeout=10000):
            pass
        assert segregation_tab._run_button.isEnabled() is True


class TestAiDependencyMissing:
    """Exercises AnalysisViewModel.ai_dependency_missing end-to-end through the widget.

    QMessageBox.warning() is a genuinely blocking modal call -- it must be
    monkeypatched here so these tests don't hang waiting for a human to
    click "OK".
    """

    def test_detection_tab_shows_dialog_and_sets_error_status(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import glas.gui.ai_dialog as ai_dialog_module

        dialog_calls: list[tuple[object, list[str]]] = []
        monkeypatch.setattr(
            ai_dialog_module.QMessageBox,
            "warning",
            lambda parent, title, text: dialog_calls.append((title, text)),
        )
        monkeypatch.setattr(
            "glas.gui.viewmodels.analysis_viewmodel.missing_ai_packages",
            lambda: ["torch", "ultralytics"],
        )

        detection_tab = widget._analysis_tabs["detection"]
        detection_tab._path_edit.setText(str(tmp_path / "dataset"))
        detection_tab._extra_field_edit.setText("weights.pt")

        with qtbot.waitSignal(widget._view_model.ai_dependency_missing, timeout=5000):
            detection_tab._on_run_clicked()

        assert len(dialog_calls) == 1
        assert "torch" in dialog_calls[0][1]
        assert "ultralytics" in dialog_calls[0][1]
        assert detection_tab._status_label.text().startswith("Error")

    def test_segmentation_tab_shows_dialog_and_sets_error_status(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import glas.gui.ai_dialog as ai_dialog_module

        dialog_calls: list[tuple[object, list[str]]] = []
        monkeypatch.setattr(
            ai_dialog_module.QMessageBox,
            "warning",
            lambda parent, title, text: dialog_calls.append((title, text)),
        )
        monkeypatch.setattr(
            "glas.gui.viewmodels.analysis_viewmodel.missing_ai_packages",
            lambda: ["sam2"],
        )

        segmentation_tab = widget._analysis_tabs["segmentation"]
        segmentation_tab._path_edit.setText(str(tmp_path / "dataset"))

        with qtbot.waitSignal(widget._view_model.ai_dependency_missing, timeout=5000):
            segmentation_tab._on_run_clicked()

        assert len(dialog_calls) == 1
        assert "sam2" in dialog_calls[0][1]
        assert segmentation_tab._status_label.text().startswith("Error")

    def test_missing_dependency_does_not_start_a_background_thread(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import glas.gui.ai_dialog as ai_dialog_module

        monkeypatch.setattr(ai_dialog_module.QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(
            "glas.gui.viewmodels.analysis_viewmodel.missing_ai_packages",
            lambda: ["torch"],
        )

        detection_tab = widget._analysis_tabs["detection"]
        detection_tab._path_edit.setText(str(tmp_path / "dataset"))

        with qtbot.waitSignal(widget._view_model.ai_dependency_missing, timeout=5000):
            detection_tab._on_run_clicked()

        assert widget._view_model.is_running is False

    def test_run_buttons_re_enabled_after_missing_dependency(
        self, widget: AnalysisPanelWidget, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import glas.gui.ai_dialog as ai_dialog_module

        monkeypatch.setattr(ai_dialog_module.QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(
            "glas.gui.viewmodels.analysis_viewmodel.missing_ai_packages",
            lambda: ["torch"],
        )

        detection_tab = widget._analysis_tabs["detection"]
        packing_tab = widget._analysis_tabs["packing"]
        detection_tab._path_edit.setText(str(tmp_path / "dataset"))

        with qtbot.waitSignal(widget._view_model.ai_dependency_missing, timeout=5000):
            detection_tab._on_run_clicked()

        assert packing_tab._run_button.isEnabled() is True

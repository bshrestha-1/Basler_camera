"""The analysis panel: tabs running the real glas.analysis pipelines on a background thread.

Wraps :class:`~glas.gui.viewmodels.analysis_viewmodel.AnalysisViewModel`.
Every tab except Histograms calls a real ``run_*`` method that delegates
straight through to :mod:`glas.analysis`/:mod:`glas.accelerometer`/
:mod:`glas.ai` -- there is no analysis logic here, only marshaling a
folder/file path (and, for Detection/Segmentation, a second field --
YOLO weights or a SAM2 model id) into a call and a result object back
into text and an optional exported plot. Histograms is an explicit,
disabled placeholder tab: no backend function exists for it yet, and a
tab that looked functional but silently did nothing would be worse than
one that says so. Detection and Segmentation are the two AI-backed tabs --
:attr:`~glas.gui.viewmodels.analysis_viewmodel.AnalysisViewModel.ai_dependency_missing`
is wired to a modal dialog (:mod:`glas.gui.ai_dialog`) instead of the
usual inline "analysis failed" status, per the project's requirement
that a missing AI dependency always surface as a clear, actionable
dialog.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from glas.accelerometer import VibrationMetrics, import_accelerometer_csv, plot_vibration_signal
from glas.ai.sam2_segmenter import SegmentationSummary
from glas.analysis import (
    BrazilNutTrajectory,
    ConvectionSummary,
    PackingSummary,
    SegregationSummary,
    plot_brazil_nut_trajectory,
    plot_packing_summary,
    plot_segregation_summary,
    plot_velocity_heatmap,
)
from glas.gui.ai_dialog import show_missing_ai_dependencies_dialog
from glas.gui.viewmodels.analysis_viewmodel import (
    DEFAULT_REPORT_FILENAME,
    DEFAULT_SAM2_MODEL_ID,
    AnalysisViewModel,
)


def _summarize_tracking(result: dict[int, list[Any]]) -> str:
    total_observations = sum(len(track) for track in result.values())
    return f"Tracks: {len(result)}\nTotal observations: {total_observations}"


def _summarize_brazil_nut(result: BrazilNutTrajectory) -> str:
    lines = [f"Track ID: {result.track_id}", f"Observations: {len(result.frame_ids)}"]
    if result.rise_time_s is not None:
        lines.append(f"Rise time: {result.rise_time_s:.2f} s")
    if result.heights_px:
        lines.append(f"Final height: {result.heights_px[-1]:.1f} px")
    return "\n".join(lines)


def _summarize_convection(result: ConvectionSummary) -> str:
    lines = [f"Frame pairs: {len(result.frame_ids)}"]
    if result.circulations:
        lines.append(f"Final circulation: {result.circulations[-1]:.2f} px²/s")
        lines.append(
            f"Mean circulation: {sum(result.circulations) / len(result.circulations):.2f} px²/s"
        )
    return "\n".join(lines)


def _summarize_packing(result: PackingSummary) -> str:
    lines = [f"Frames: {len(result.frame_ids)}"]
    if result.metrics:
        final = result.metrics[-1]
        lines.append(f"Final packing fraction: {final.packing_fraction:.3f}")
        lines.append(f"Final particle count: {final.particle_count}")
    return "\n".join(lines)


def _summarize_segregation(result: SegregationSummary) -> str:
    lines = [f"Frames: {len(result.frame_ids)}"]
    if result.metrics:
        final = result.metrics[-1]
        lines.append(f"Final segregation index: {final.segregation_index:.3f}")
        lines.append(f"Final mixing entropy: {final.mixing_entropy:.3f}")
    return "\n".join(lines)


def _summarize_vibration(result: VibrationMetrics) -> str:
    return (
        f"Frequency: {result.frequency_hz:.2f} Hz\n"
        f"Amplitude: {result.amplitude_m * 1000:.3f} mm\n"
        f"Gamma: {result.gamma:.2f}\n"
        f"Peak acceleration: {result.peak_acceleration_g:.2f} g"
    )


def _summarize_segmentation(result: SegmentationSummary) -> str:
    return (
        f"Particles: {result.particle_count}\n"
        f"Packing fraction: {result.packing_fraction:.3f}\n"
        f"Void fraction: {result.void_fraction:.3f}\n"
        f"Contacts: {len(result.contacts)}"
    )


def _summarize_report(result: Path) -> str:
    return f"Report saved to {result}"


class _AnalysisTab(QWidget):
    """One analysis tab: a path picker, run button, status/result display, and plot export.

    ``extra_field_label`` adds a second text field alongside the path
    picker (YOLO weights for Detection, a SAM2 model id for
    Segmentation) -- when given, ``run`` is called as
    ``run(Path(path), extra_text)`` instead of ``run(Path(path))``.
    """

    def __init__(
        self,
        *,
        kind: str,
        path_label: str,
        run: Callable[..., None],
        summarize: Any,
        export_plot: Any | None,
        extra_field_label: str | None = None,
        extra_field_default: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self._run = run
        self._summarize = summarize
        self._export_plot = export_plot
        self._last_result: Any = None

        self._path_edit = QLineEdit()
        self._browse_button = QPushButton("Browse...")
        self._extra_field_edit = QLineEdit(extra_field_default) if extra_field_label else None
        self._run_button = QPushButton("Run")
        self._status_label = QLabel("Idle")
        self._result_text = QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._export_button = QPushButton("Export Plot...")
        self._export_button.setEnabled(False)
        self._export_button.setVisible(export_plot is not None)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel(path_label))
        path_row.addWidget(self._path_edit, stretch=1)
        path_row.addWidget(self._browse_button)

        button_row = QHBoxLayout()
        button_row.addWidget(self._run_button)
        button_row.addWidget(self._export_button)
        button_row.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(path_row)
        if extra_field_label is not None and self._extra_field_edit is not None:
            extra_row = QHBoxLayout()
            extra_row.addWidget(QLabel(extra_field_label))
            extra_row.addWidget(self._extra_field_edit, stretch=1)
            layout.addLayout(extra_row)
        layout.addLayout(button_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._result_text, stretch=1)

        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)

    def _on_browse_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if chosen:
            self._path_edit.setText(chosen)

    def _on_run_clicked(self) -> None:
        path = self._path_edit.text()
        if not path:
            return
        self._status_label.setText("Running...")
        self._result_text.clear()
        self._export_button.setEnabled(False)
        if self._extra_field_edit is not None:
            self._run(Path(path), self._extra_field_edit.text())
        else:
            self._run(Path(path))

    def _on_export_clicked(self) -> None:
        if self._last_result is None or self._export_plot is None:
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", filter="PNG (*.png)")
        if output_path:
            self._export_plot(self._last_result, Path(output_path))

    def on_finished(self, result: Any) -> None:
        self._status_label.setText("Done")
        self._result_text.setPlainText(self._summarize(result))
        self._last_result = result
        self._export_button.setEnabled(self._export_plot is not None)

    def on_failed(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")
        self._export_button.setEnabled(False)

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_button.setEnabled(enabled)


class AnalysisPanelWidget(QWidget):
    """Tabbed analysis panel: tracking, detection, Brazil nut, convection, packing, segregation,
    segmentation, vibration, report.

    Parameters
    ----------
    view_model : AnalysisViewModel
    """

    def __init__(self, view_model: AnalysisViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view_model = view_model

        self._tabs = QTabWidget()
        self._analysis_tabs: dict[str, _AnalysisTab] = {}

        self._add_tab(
            kind="tracking",
            title="Tracking",
            path_label="Dataset folder:",
            run=view_model.run_tracking,
            summarize=_summarize_tracking,
            export_plot=None,
        )
        self._add_tab(
            kind="detection",
            title="Detection (YOLO)",
            path_label="Dataset folder:",
            run=view_model.run_detection,
            summarize=_summarize_tracking,
            export_plot=None,
            extra_field_label="YOLO weights:",
        )
        self._add_tab(
            kind="brazil_nut",
            title="Brazil Nut",
            path_label="Dataset folder:",
            run=view_model.run_brazil_nut,
            summarize=_summarize_brazil_nut,
            export_plot=plot_brazil_nut_trajectory,
        )
        self._add_tab(
            kind="convection",
            title="Convection",
            path_label="Dataset folder:",
            run=view_model.run_convection,
            summarize=_summarize_convection,
            export_plot=lambda result, path: plot_velocity_heatmap(result.fields[-1], path),
        )
        self._add_tab(
            kind="packing",
            title="Packing",
            path_label="Dataset folder:",
            run=view_model.run_packing,
            summarize=_summarize_packing,
            export_plot=plot_packing_summary,
        )
        self._add_tab(
            kind="segregation",
            title="Segregation",
            path_label="Dataset folder:",
            run=view_model.run_segregation,
            summarize=_summarize_segregation,
            export_plot=plot_segregation_summary,
        )
        self._add_tab(
            kind="segmentation",
            title="Segmentation (SAM2)",
            path_label="Dataset folder:",
            run=view_model.run_segmentation,
            summarize=_summarize_segmentation,
            export_plot=None,
            extra_field_label="SAM2 model id:",
            extra_field_default=DEFAULT_SAM2_MODEL_ID,
        )
        self._add_tab(
            kind="vibration",
            title="Vibration",
            path_label="Accelerometer CSV:",
            run=view_model.run_vibration,
            summarize=_summarize_vibration,
            export_plot=self._export_vibration_plot,
        )
        self._add_tab(
            kind="report",
            title="Report",
            path_label="Dataset folder:",
            run=view_model.run_report,
            summarize=_summarize_report,
            export_plot=None,
            extra_field_label="Report output path:",
            extra_field_default=DEFAULT_REPORT_FILENAME,
        )

        self._add_placeholder_tab(
            "Histograms", "Per-analysis result histograms are not yet implemented."
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)

        view_model.analysis_started.connect(self._on_run_started)
        view_model.analysis_finished.connect(self._on_analysis_finished)
        view_model.analysis_failed.connect(self._on_analysis_failed)
        view_model.ai_dependency_missing.connect(self._on_ai_dependency_missing)

    def _add_tab(
        self,
        *,
        kind: str,
        title: str,
        path_label: str,
        run: Callable[..., None],
        summarize: Any,
        export_plot: Any | None,
        extra_field_label: str | None = None,
        extra_field_default: str = "",
    ) -> None:
        tab = _AnalysisTab(
            kind=kind,
            path_label=path_label,
            run=run,
            summarize=summarize,
            export_plot=export_plot,
            extra_field_label=extra_field_label,
            extra_field_default=extra_field_default,
        )
        self._analysis_tabs[kind] = tab
        self._tabs.addTab(tab, title)

    def _add_placeholder_tab(self, title: str, message: str) -> None:
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel(message)
        label.setEnabled(False)
        layout.addWidget(label)
        layout.addStretch()
        self._tabs.addTab(placeholder, title)
        self._tabs.setTabEnabled(self._tabs.indexOf(placeholder), False)

    def _export_vibration_plot(self, result: VibrationMetrics, output_path: Path) -> None:
        tab = self._analysis_tabs["vibration"]
        recording = import_accelerometer_csv(Path(tab._path_edit.text()))
        plot_vibration_signal(recording, output_path)

    def _on_analysis_finished(self, kind: str, result: Any) -> None:
        self._analysis_tabs[kind].on_finished(result)
        self._set_all_run_buttons_enabled(True)

    def _on_analysis_failed(self, kind: str, message: str) -> None:
        self._analysis_tabs[kind].on_failed(message)
        self._set_all_run_buttons_enabled(True)

    def _on_ai_dependency_missing(self, kind: str, missing: list[str]) -> None:
        show_missing_ai_dependencies_dialog(self, missing)
        self._analysis_tabs[kind].on_failed("Missing AI dependencies -- see dialog.")
        self._set_all_run_buttons_enabled(True)

    def _set_all_run_buttons_enabled(self, enabled: bool) -> None:
        for tab in self._analysis_tabs.values():
            tab.set_run_enabled(enabled)

    def _on_run_started(self) -> None:
        self._set_all_run_buttons_enabled(False)

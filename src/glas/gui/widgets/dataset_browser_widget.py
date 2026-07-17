"""The dataset browser panel: search, preview, export, delete, and duplicate past experiments.

Wraps :class:`~glas.gui.viewmodels.dataset_viewmodel.DatasetViewModel`.
The one piece of read-only backend access this widget makes directly
(rather than through the ViewModel) is :func:`glas.dataset.iter_frames`,
to render a thumbnail of an experiment's first frame -- reading a frame
back for display is presentation, not a mutation, the same reasoning
:class:`~glas.gui.widgets.live_preview_widget.LivePreviewWidget` already
applies to :func:`glas.display.render_frame`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glas.dataset import iter_frames
from glas.display import render_frame
from glas.experiment import ExperimentSummary, get_physical_parameters
from glas.export import ExportFormat
from glas.gui.viewmodels.dataset_viewmodel import DatasetViewModel

_TABLE_HEADERS = ("Run ID", "Name", "Frames", "Tags", "Created")
_THUMBNAIL_SIZE = 200
_EXPORT_FORMATS: tuple[ExportFormat, ...] = ("tiff", "png", "mp4", "avi", "gif")


def _bgr_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a BGR ``uint8`` image (as :func:`glas.display.render_frame` returns) to a pixmap."""
    height, width = image.shape[0], image.shape[1]
    rgb = image[:, :, ::-1].copy()
    qimage = QImage(rgb.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)


class DatasetBrowserWidget(QWidget):
    """Search, preview, export, delete, and duplicate previously recorded experiments.

    Parameters
    ----------
    view_model : DatasetViewModel
    """

    def __init__(self, view_model: DatasetViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view_model = view_model
        self._experiments: list[ExperimentSummary] = []

        self._name_filter_edit = QLineEdit()
        self._name_filter_edit.setPlaceholderText("Filter by name...")
        self._tag_filter_edit = QLineEdit()
        self._tag_filter_edit.setPlaceholderText("Filter by tag...")
        self._search_button = QPushButton("Search")

        self._table = QTableWidget(0, len(_TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(_TABLE_HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._thumbnail_label = QLabel("No preview")
        self._thumbnail_label.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)

        self._detail_group = QGroupBox("Metadata")
        self._detail_form = QFormLayout(self._detail_group)
        self._run_id_label = QLabel("--")
        self._name_label = QLabel("--")
        self._notes_label = QLabel("--")
        self._camera_model_label = QLabel("--")
        self._frame_count_label = QLabel("--")
        self._created_label = QLabel("--")
        self._tags_label = QLabel("--")
        self._physical_params_label = QLabel("--")
        self._detail_form.addRow("Run ID:", self._run_id_label)
        self._detail_form.addRow("Name:", self._name_label)
        self._detail_form.addRow("Notes:", self._notes_label)
        self._detail_form.addRow("Camera:", self._camera_model_label)
        self._detail_form.addRow("Frames:", self._frame_count_label)
        self._detail_form.addRow("Created:", self._created_label)
        self._detail_form.addRow("Tags:", self._tags_label)
        self._detail_form.addRow("Physical parameters:", self._physical_params_label)

        self._export_format_combo = QComboBox()
        self._export_format_combo.addItems(_EXPORT_FORMATS)
        self._export_button = QPushButton("Export...")
        self._duplicate_button = QPushButton("Duplicate")
        self._delete_button = QPushButton("Delete")

        self._set_selection_actions_enabled(False)

        self._build_layout()
        self._connect_signals()
        self.refresh()

    def _build_layout(self) -> None:
        search_row = QHBoxLayout()
        search_row.addWidget(self._name_filter_edit)
        search_row.addWidget(self._tag_filter_edit)
        search_row.addWidget(self._search_button)

        preview_column = QVBoxLayout()
        preview_column.addWidget(self._thumbnail_label)
        preview_column.addWidget(self._detail_group)

        actions_row = QHBoxLayout()
        actions_row.addWidget(self._export_format_combo)
        actions_row.addWidget(self._export_button)
        actions_row.addWidget(self._duplicate_button)
        actions_row.addWidget(self._delete_button)
        preview_column.addLayout(actions_row)
        preview_column.addStretch()

        content_row = QHBoxLayout()
        content_row.addWidget(self._table, stretch=2)
        content_row.addLayout(preview_column, stretch=1)

        layout = QVBoxLayout(self)
        layout.addLayout(search_row)
        layout.addLayout(content_row)

    def _connect_signals(self) -> None:
        self._search_button.clicked.connect(self.refresh)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._export_button.clicked.connect(self._on_export_clicked)
        self._duplicate_button.clicked.connect(self._on_duplicate_clicked)
        self._delete_button.clicked.connect(self._on_delete_clicked)
        self._view_model.experiments_changed.connect(self._on_experiments_changed)
        self._view_model.export_finished.connect(self._on_export_finished)
        self._view_model.error_occurred.connect(self._on_error)

    def refresh(self) -> None:
        """Re-query experiments using the current name/tag filter fields."""
        self._view_model.refresh(
            name_contains=self._name_filter_edit.text() or None,
            tag=self._tag_filter_edit.text() or None,
        )

    def _on_experiments_changed(self, experiments: list[ExperimentSummary]) -> None:
        self._experiments = experiments
        self._table.setRowCount(len(experiments))
        for row, summary in enumerate(experiments):
            self._table.setItem(row, 0, QTableWidgetItem(summary.run_id))
            self._table.setItem(row, 1, QTableWidgetItem(summary.name))
            self._table.setItem(row, 2, QTableWidgetItem(str(summary.frame_count)))
            self._table.setItem(row, 3, QTableWidgetItem(", ".join(summary.tags)))
            self._table.setItem(row, 4, QTableWidgetItem(summary.created_at_utc))
        self._set_selection_actions_enabled(False)
        self._show_preview(None)

    def _selected_summary(self) -> ExperimentSummary | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        return self._experiments[rows[0].row()]

    def _on_selection_changed(self) -> None:
        summary = self._selected_summary()
        self._set_selection_actions_enabled(summary is not None)
        self._show_preview(summary)

    def _set_selection_actions_enabled(self, enabled: bool) -> None:
        self._export_button.setEnabled(enabled)
        self._duplicate_button.setEnabled(enabled)
        self._delete_button.setEnabled(enabled)

    def _show_preview(self, summary: ExperimentSummary | None) -> None:
        if summary is None:
            self._thumbnail_label.setText("No preview")
            self._thumbnail_label.setPixmap(QPixmap())
            self._run_id_label.setText("--")
            self._name_label.setText("--")
            self._notes_label.setText("--")
            self._camera_model_label.setText("--")
            self._frame_count_label.setText("--")
            self._created_label.setText("--")
            self._tags_label.setText("--")
            self._physical_params_label.setText("--")
            return

        self._run_id_label.setText(summary.run_id)
        self._name_label.setText(summary.name or "(unnamed)")
        self._notes_label.setText(summary.notes or "--")
        self._camera_model_label.setText(summary.camera_model)
        self._frame_count_label.setText(str(summary.frame_count))
        self._created_label.setText(summary.created_at_utc)
        self._tags_label.setText(", ".join(summary.tags) if summary.tags else "--")
        self._physical_params_label.setText(self._format_physical_parameters(summary))

        try:
            frame = next(iter_frames(summary.folder))
        except (StopIteration, OSError):
            self._thumbnail_label.setText("No preview")
            self._thumbnail_label.setPixmap(QPixmap())
            return
        image = render_frame(frame)
        pixmap = _bgr_to_qpixmap(image).scaled(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self._thumbnail_label.setPixmap(pixmap)

    def _format_physical_parameters(self, summary: ExperimentSummary) -> str:
        parameters = get_physical_parameters(summary.metadata)
        filled = {
            key: value for key, value in parameters.model_dump().items() if value not in (None, "")
        }
        if not filled:
            return "--"
        return ", ".join(f"{key}={value}" for key, value in filled.items())

    def _on_export_clicked(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return
        export_format = self._export_format_combo.currentText()
        is_image_sequence = export_format in ("tiff", "png")
        if is_image_sequence:
            chosen = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        else:
            chosen, _ = QFileDialog.getSaveFileName(self, "Export As", filter=f"*.{export_format}")
        if not chosen:
            return
        self._view_model.export(summary.run_id, Path(chosen), export_format)  # type: ignore[arg-type]

    def _on_export_finished(self, result: Any) -> None:
        QMessageBox.information(self, "Export Complete", f"Exported to {result.output_path}")

    def _on_duplicate_clicked(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return
        self._view_model.duplicate(summary.run_id)

    def _on_delete_clicked(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return
        confirmed = QMessageBox.question(
            self,
            "Delete Experiment",
            f"Permanently delete {summary.run_id} ({summary.name or 'unnamed'})?",
        )
        if confirmed == QMessageBox.StandardButton.Yes:
            self._view_model.delete(summary.run_id)

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "Dataset Browser Error", message)

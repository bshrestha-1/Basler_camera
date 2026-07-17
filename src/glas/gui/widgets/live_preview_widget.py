"""The live camera preview panel: the largest, most central panel in the main window.

Wraps :class:`~glas.gui.viewmodels.live_feed_viewmodel.LiveFeedViewModel`.
Every overlay this widget can draw (crosshair, ROI box, reference grid,
FPS text) is rendered by the existing, Qt-free
:func:`glas.display.render_frame` -- the same function
:class:`~glas.display.PreviewWindow` uses for the CLI's own preview
window -- so the GUI and CLI can never show visually inconsistent
overlays. Only the wall-clock timestamp text and the Qt-specific
viewport zoom/pan (a pure display transform, distinct from
:attr:`~glas.preview.Preview.zoom`'s server-side crop) are computed here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QRubberBand,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from glas.camera_validator import ROI
from glas.display import render_frame, render_histogram
from glas.frame import Frame
from glas.gui.viewmodels.live_feed_viewmodel import LiveFeedViewModel
from glas.preview import Preview
from glas.timestamps import WallClockReference

_ZOOM_STEP = 1.25
_HISTOGRAM_WIDTH = 256
_HISTOGRAM_HEIGHT = 80

InteractionMode = Literal["pan", "crosshair", "roi"]


def _format_wall_clock_text(wall_clock_ns: int) -> str:
    """Format a wall-clock nanosecond timestamp as a local ``HH:MM:SS.mmm`` string."""
    seconds = wall_clock_ns / 1e9
    return (
        datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone().strftime("%H:%M:%S.%f")[:-3]
    )


def _to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a BGR or grayscale ``uint8`` numpy image to a :class:`QPixmap`."""
    height, width = image.shape[0], image.shape[1]
    if image.ndim == 2:
        qimage = QImage(image.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
    else:
        rgb = image[:, :, ::-1].copy()  # BGR -> RGB
        qimage = QImage(rgb.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)


class _ImageView(QGraphicsView):
    """A :class:`QGraphicsView` supporting pan, click-to-place, and drag-to-select."""

    point_clicked = Signal(float, float)
    region_selected = Signal(float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode: InteractionMode = "pan"
        self._rubber_band: QRubberBand | None = None
        self._origin: QPoint | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_mode(self, mode: InteractionMode) -> None:
        """Switch interaction mode, updating the pan drag behavior accordingly."""
        self.mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self.mode == "crosshair" and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.pos())
            self.point_clicked.emit(point.x(), point.y())
            return
        if self.mode == "roi" and event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
            self._rubber_band.setGeometry(QRect(self._origin, self._origin))
            self._rubber_band.show()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._rubber_band is not None and self._origin is not None:
            self._rubber_band.setGeometry(QRect(self._origin, event.pos()).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._rubber_band is not None and self._origin is not None:
            rect = QRect(self._origin, event.pos()).normalized()
            top_left = self.mapToScene(rect.topLeft())
            bottom_right = self.mapToScene(rect.bottomRight())
            self._rubber_band.hide()
            self._rubber_band = None
            self._origin = None
            width = bottom_right.x() - top_left.x()
            height = bottom_right.y() - top_left.y()
            if width > 0 and height > 0:
                self.region_selected.emit(top_left.x(), top_left.y(), width, height)
            return
        super().mouseReleaseEvent(event)


class LivePreviewWidget(QWidget):
    """The main live camera preview panel.

    Parameters
    ----------
    view_model : LiveFeedViewModel
        Drives this widget; call :meth:`~LiveFeedViewModel.attach` on it
        (from the owner assembling the main window) to start streaming
        frames here.
    """

    roi_selected = Signal(object)

    def __init__(self, view_model: LiveFeedViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view_model = view_model
        self._wall_clock_ref: WallClockReference | None = None
        self._show_timestamp = False
        self._roi: ROI | None = None
        self._zoom_percent = 100.0

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._view = _ImageView(self)
        self._view.setScene(self._scene)
        self._view.point_clicked.connect(self._on_point_clicked)
        self._view.region_selected.connect(self._on_region_selected)

        self._frame_label = QLabel("Frame: --")
        self._fps_label = QLabel("FPS: --")
        self._zoom_label = QLabel("Zoom: 100%")
        self._histogram_label = QLabel()
        self._histogram_label.setFixedSize(_HISTOGRAM_WIDTH, _HISTOGRAM_HEIGHT)

        self._toolbar = self._build_toolbar()

        status_row = QHBoxLayout()
        status_row.addWidget(self._frame_label)
        status_row.addWidget(self._fps_label)
        status_row.addWidget(self._zoom_label)
        status_row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._view, stretch=1)
        layout.addLayout(status_row)
        layout.addWidget(self._histogram_label)

        self._view_model.frame_ready.connect(self._on_frame_ready)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar(self)

        zoom_in = toolbar.addAction("Zoom In")
        zoom_in.triggered.connect(self.zoom_in)
        zoom_out = toolbar.addAction("Zoom Out")
        zoom_out.triggered.connect(self.zoom_out)
        fit = toolbar.addAction("Fit to Window")
        fit.triggered.connect(self.fit_to_window)
        full_res = toolbar.addAction("100%")
        full_res.triggered.connect(self.full_resolution)
        toolbar.addSeparator()

        self._pan_action = toolbar.addAction("Pan")
        self._pan_action.setCheckable(True)
        self._pan_action.setChecked(True)
        self._pan_action.triggered.connect(lambda: self._set_interaction_mode("pan"))

        self._crosshair_action = toolbar.addAction("Crosshair")
        self._crosshair_action.setCheckable(True)
        self._crosshair_action.triggered.connect(
            lambda: self._set_interaction_mode(
                "crosshair" if self._crosshair_action.isChecked() else "pan"
            )
        )

        self._roi_action = toolbar.addAction("Select ROI")
        self._roi_action.setCheckable(True)
        self._roi_action.triggered.connect(
            lambda: self._set_interaction_mode("roi" if self._roi_action.isChecked() else "pan")
        )
        toolbar.addSeparator()

        self._grid_action = toolbar.addAction("Grid")
        self._grid_action.setCheckable(True)
        self._grid_action.triggered.connect(self._on_grid_toggled)

        self._timestamp_action = toolbar.addAction("Timestamp")
        self._timestamp_action.setCheckable(True)
        self._timestamp_action.triggered.connect(self._on_timestamp_toggled)

        return toolbar

    def _set_interaction_mode(self, mode: InteractionMode) -> None:
        self._view.set_mode(mode)
        self._pan_action.setChecked(mode == "pan")
        self._crosshair_action.setChecked(mode == "crosshair")
        self._roi_action.setChecked(mode == "roi")
        if mode != "crosshair":
            self.preview.crosshair = False
        if mode != "roi":
            self.preview.show_roi = False

    def _on_grid_toggled(self, checked: bool) -> None:
        self.preview.overlay_grid = checked

    def _on_timestamp_toggled(self, checked: bool) -> None:
        self._show_timestamp = checked

    def _on_point_clicked(self, x: float, y: float) -> None:
        self.preview.crosshair = True
        self.preview.crosshair_position = (int(x), int(y))

    def _on_region_selected(self, x: float, y: float, width: float, height: float) -> None:
        self._roi = ROI(
            offset_x=max(int(x), 0),
            offset_y=max(int(y), 0),
            width=max(int(width), 1),
            height=max(int(height), 1),
        )
        self.preview.show_roi = True
        self.roi_selected.emit(self._roi)

    @property
    def preview(self) -> Preview:
        """The underlying :class:`~glas.preview.Preview` this widget renders."""
        if self._view_model.preview is None:
            raise RuntimeError("LiveFeedViewModel is not attached to a buffer yet.")
        return self._view_model.preview

    def zoom_in(self) -> None:
        """Zoom the viewport in by one step (does not affect the source crop)."""
        self._view.scale(_ZOOM_STEP, _ZOOM_STEP)
        self._zoom_percent *= _ZOOM_STEP
        self._update_zoom_label()

    def zoom_out(self) -> None:
        """Zoom the viewport out by one step (does not affect the source crop)."""
        self._view.scale(1 / _ZOOM_STEP, 1 / _ZOOM_STEP)
        self._zoom_percent /= _ZOOM_STEP
        self._update_zoom_label()

    def fit_to_window(self) -> None:
        """Scale the viewport so the whole frame fits within the visible area."""
        if self._pixmap_item.pixmap().isNull():
            return
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_percent = self._view.transform().m11() * 100.0
        self._update_zoom_label()

    def full_resolution(self) -> None:
        """Reset the viewport to 1 screen pixel per source-image pixel."""
        self._view.resetTransform()
        self._zoom_percent = 100.0
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        self._zoom_label.setText(f"Zoom: {self._zoom_percent:.0f}%")

    def _on_frame_ready(self, frame: Frame) -> None:
        if self._wall_clock_ref is None:
            self._wall_clock_ref = WallClockReference.capture()

        timestamp_text = None
        if self._show_timestamp:
            wall_ns = self._wall_clock_ref.to_wall_clock_ns(frame.host_timestamp_ns)
            timestamp_text = _format_wall_clock_text(wall_ns)

        image = render_frame(
            frame,
            zoom=self.preview.zoom,
            crosshair=self.preview.crosshair,
            crosshair_position=self.preview.crosshair_position,
            roi=self._roi if self.preview.show_roi else None,
            fps=self.preview.fps(),
            overlay_grid=self.preview.overlay_grid,
            timestamp_text=timestamp_text,
        )
        pixmap = _to_qpixmap(image)
        was_empty = self._pixmap_item.pixmap().isNull()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        if was_empty:
            self.fit_to_window()

        self._frame_label.setText(f"Frame: {frame.frame_id}")
        self._fps_label.setText(f"FPS: {self.preview.fps():.1f}")

        counts = Preview.histogram(frame)
        histogram_image = render_histogram(counts, width=_HISTOGRAM_WIDTH, height=_HISTOGRAM_HEIGHT)
        self._histogram_label.setPixmap(_to_qpixmap(histogram_image))

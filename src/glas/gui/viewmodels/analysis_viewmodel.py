"""ViewModel running the real GLAS analysis pipelines from the analysis panel widget.

Every ``run_*`` method delegates to the exact same function
``glas analyze``/``glas brazil-nut``/``glas convection``/``glas packing``/
``glas segregation``/``glas accelerometer analyze``/``glas ai detect``/
``glas ai segment`` call from the CLI (:mod:`glas.analysis`,
:mod:`glas.accelerometer`, :mod:`glas.ai`) -- there is no GUI-specific
analysis logic to drift out of sync. Each call runs on a background
:class:`~PySide6.QtCore.QThread` so a slow analysis (a large recording, a
YOLO/SAM2 model) never freezes the UI.

``run_detection``/``run_segmentation`` (the two AI-backed analyses) check
:func:`~glas.ai.dependencies.missing_ai_packages` *before* starting their
background thread, emitting :attr:`AnalysisViewModel.ai_dependency_missing`
instead if any are absent -- :class:`~glas.gui.widgets.analysis_panel_widget.AnalysisPanelWidget`
turns that into the install-hint dialog the GLAS AI dependency contract
requires, rather than a normal inline "analysis failed" status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from glas.accelerometer import VibrationMetrics, analyze_vibration
from glas.ai.dependencies import missing_ai_packages
from glas.ai.sam2_segmenter import Sam2Segmenter, SegmentationSummary, compute_segmentation_summary
from glas.ai.yolo_detector import track_dataset_yolo
from glas.analysis import (
    BrazilNutTrajectory,
    ConvectionSummary,
    PackingSummary,
    SegregationSummary,
    analyze_brazil_nut,
    analyze_convection,
    analyze_packing,
    analyze_segregation,
    track_dataset,
)
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA, detect_particles
from glas.dataset import iter_frames
from glas.exceptions import AIModelError

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"


def _segment_last_frame(
    folder: Path, model_id: str, *, min_area: float = DEFAULT_MIN_AREA
) -> SegmentationSummary:
    """Segment every particle in a dataset's last frame with SAM2, prompted by classical detection.

    Classical blob detection (not a trained YOLO model) supplies the box
    prompts here, so the Segmentation tab works with nothing but a
    dataset folder and a SAM2 model id -- no YOLO training required
    first, matching the "load pretrained weights and begin analysis"
    inference-only path.
    """
    last_frame = None
    for frame in iter_frames(folder):
        last_frame = frame
    if last_frame is None:
        raise AIModelError("Dataset has no frames.")

    detections = detect_particles(last_frame.image, min_area=min_area)
    segmenter = Sam2Segmenter(model_id=model_id)
    segments = segmenter.segment_frame(last_frame.image, detections)
    return compute_segmentation_summary(segments, last_frame.image.shape[:2])


class _AnalysisWorker(QThread):
    """Runs one analysis callable off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self, fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any], parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - must reach the UI, not silently kill the thread
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class AnalysisViewModel(QObject):
    """Runs particle tracking, Brazil nut, convection, packing, segregation, and vibration analyses.

    Only one analysis runs at a time (a second call while one is still
    running is rejected via :attr:`analysis_failed`, not queued) -- kept
    simple deliberately, since a lab operator running an analysis is
    watching it, not batching many at once.

    Signals
    -------
    analysis_started(str)
        Emitted with a ``kind`` string (``"tracking"``, ``"brazil_nut"``,
        ``"convection"``, ``"packing"``, ``"segregation"``,
        ``"vibration"``, ``"detection"``, or ``"segmentation"``) when a
        background run begins.
    analysis_finished(str, object)
        Emitted with ``(kind, result)`` on success. ``result`` is
        whichever type the underlying function returns (e.g.
        :class:`~glas.analysis.BrazilNutTrajectory`).
    analysis_failed(str, str)
        Emitted with ``(kind, message)`` instead of ``analysis_finished``
        on failure.
    ai_dependency_missing(str, list)
        Emitted with ``(kind, missing_packages)`` instead of either of the
        above when ``run_detection``/``run_segmentation`` is called
        without ``torch``/``ultralytics``/``sam2`` installed -- no
        background thread is started at all.
    """

    analysis_started = Signal(str)
    analysis_finished = Signal(str, object)
    analysis_failed = Signal(str, str)
    ai_dependency_missing = Signal(str, list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _AnalysisWorker | None = None

    @property
    def is_running(self) -> bool:
        """``True`` if an analysis is currently in progress."""
        return self._worker is not None and self._worker.isRunning()

    def run_tracking(self, folder: Path, **kwargs: Any) -> None:
        """Run :func:`glas.analysis.track_dataset` in the background."""
        self._run("tracking", track_dataset, folder, **kwargs)

    def run_brazil_nut(self, folder: Path, **kwargs: Any) -> None:
        """Run :func:`glas.analysis.analyze_brazil_nut` in the background."""
        self._run("brazil_nut", analyze_brazil_nut, folder, **kwargs)

    def run_convection(self, folder: Path, **kwargs: Any) -> None:
        """Run :func:`glas.analysis.analyze_convection` in the background."""
        self._run("convection", analyze_convection, folder, **kwargs)

    def run_packing(self, folder: Path, **kwargs: Any) -> None:
        """Run :func:`glas.analysis.analyze_packing` in the background."""
        self._run("packing", analyze_packing, folder, **kwargs)

    def run_segregation(self, folder: Path, **kwargs: Any) -> None:
        """Run :func:`glas.analysis.analyze_segregation` in the background."""
        self._run("segregation", analyze_segregation, folder, **kwargs)

    def run_vibration(self, csv_path: Path, **kwargs: Any) -> None:
        """Run :func:`glas.accelerometer.analyze_vibration` in the background."""
        self._run("vibration", analyze_vibration, csv_path, **kwargs)

    def run_detection(self, folder: Path, weights: str, **kwargs: Any) -> None:
        """Run :func:`glas.ai.yolo_detector.track_dataset_yolo` in the background.

        Emits :attr:`ai_dependency_missing` instead of starting a thread
        if ``torch``/``ultralytics`` is not installed.
        """
        self._run_ai("detection", track_dataset_yolo, folder, weights, **kwargs)

    def run_segmentation(self, folder: Path, model_id: str = DEFAULT_SAM2_MODEL_ID) -> None:
        """Segment every particle in a dataset's last frame with SAM2, in the background.

        Emits :attr:`ai_dependency_missing` instead of starting a thread
        if ``torch``/``sam2`` is not installed.
        """
        self._run_ai("segmentation", _segment_last_frame, folder, model_id)

    def _run_ai(self, kind: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        missing = missing_ai_packages()
        if missing:
            self.ai_dependency_missing.emit(kind, missing)
            return
        self._run(kind, fn, *args, **kwargs)

    def _run(self, kind: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        if self.is_running:
            self.analysis_failed.emit(kind, "Another analysis is already running.")
            return

        # Parented to self (not explicitly deleteLater'd): deleting a QThread
        # via its own finished->deleteLater is a known crash risk once
        # something else still references it (here, self._worker) -- Qt's
        # normal parent/child ownership cleans it up when this ViewModel is,
        # which is enough for the rare, human-triggered runs this drives.
        worker = _AnalysisWorker(fn, args, kwargs, parent=self)
        worker.finished_ok.connect(lambda result, k=kind: self.analysis_finished.emit(k, result))
        worker.failed.connect(lambda message, k=kind: self.analysis_failed.emit(k, message))
        self._worker = worker
        self.analysis_started.emit(kind)
        worker.start()


__all__ = [
    "AnalysisViewModel",
    "BrazilNutTrajectory",
    "ConvectionSummary",
    "DEFAULT_SAM2_MODEL_ID",
    "PackingSummary",
    "SegmentationSummary",
    "SegregationSummary",
    "VibrationMetrics",
]

"""Qt ViewModels translating the Qt-free GLAS backend into signals/slots for the GUI widgets.

Each ViewModel wraps exactly one backend class or module (:class:`~glas.camera.Camera`,
:class:`~glas.controller.RecorderController`, :class:`~glas.preview.Preview`,
:class:`~glas.monitor.PerformanceMonitor`, :class:`~glas.experiment.ExperimentManager`,
:mod:`glas.analysis`/:mod:`glas.accelerometer`) and holds no business logic of its own --
widgets in :mod:`glas.gui.widgets` connect to these signals and call these methods, and never
touch the backend directly.
"""

from __future__ import annotations

from glas.gui.viewmodels.analysis_viewmodel import (
    DEFAULT_SAM2_MODEL_ID,
    AnalysisViewModel,
    BrazilNutTrajectory,
    ConvectionSummary,
    PackingSummary,
    SegmentationSummary,
    SegregationSummary,
    VibrationMetrics,
)
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel
from glas.gui.viewmodels.dataset_viewmodel import DatasetViewModel
from glas.gui.viewmodels.hardware_status_viewmodel import DeviceStatus, HardwareStatusViewModel
from glas.gui.viewmodels.live_feed_viewmodel import LiveFeedViewModel
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel

__all__ = [
    "AnalysisViewModel",
    "BrazilNutTrajectory",
    "CameraViewModel",
    "ConvectionSummary",
    "DatasetViewModel",
    "DEFAULT_SAM2_MODEL_ID",
    "DeviceStatus",
    "HardwareStatusViewModel",
    "LiveFeedViewModel",
    "PackingSummary",
    "RecordingViewModel",
    "SegmentationSummary",
    "SegregationSummary",
    "VibrationMetrics",
]

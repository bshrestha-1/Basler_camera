"""Qt widgets (View) for the GLAS desktop GUI.

Every widget here is a thin :class:`~PySide6.QtWidgets.QWidget` subclass
with no business logic of its own -- each is constructed with (and only
talks to) one ViewModel from :mod:`glas.gui.viewmodels`, which is in turn
the only thing that talks to the Qt-free GLAS backend.
"""

from __future__ import annotations

from glas.gui.widgets.analysis_panel_widget import AnalysisPanelWidget
from glas.gui.widgets.camera_controls_widget import CameraControlsWidget
from glas.gui.widgets.dataset_browser_widget import DatasetBrowserWidget
from glas.gui.widgets.experiment_metadata_widget import ExperimentMetadataWidget
from glas.gui.widgets.hardware_status_widget import HardwareStatusWidget
from glas.gui.widgets.live_preview_widget import LivePreviewWidget
from glas.gui.widgets.log_console_widget import LogConsoleWidget
from glas.gui.widgets.recording_controls_widget import RecordingControlsWidget

__all__ = [
    "AnalysisPanelWidget",
    "CameraControlsWidget",
    "DatasetBrowserWidget",
    "ExperimentMetadataWidget",
    "HardwareStatusWidget",
    "LivePreviewWidget",
    "LogConsoleWidget",
    "RecordingControlsWidget",
]

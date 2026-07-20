"""The GLAS main window: assembles every panel into one dockable desktop application.

The only widget-to-widget orchestration in the whole GUI layer lives here
-- everywhere else, a widget talks to exactly one ViewModel and nothing
else. Two pieces of orchestration exist:

1. Live preview needs frames even when nothing is being recorded, but
   :class:`~glas.recorder.Recorder` only runs its own
   :class:`~glas.acquisition.Acquisition` while actively recording. This
   window therefore runs its own standalone ``Acquisition`` for
   live-preview-only viewing, and switches
   :class:`~glas.gui.viewmodels.live_feed_viewmodel.LiveFeedViewModel`
   over to the ``Recorder``'s own ring buffer for the duration of a
   recording (safe: :class:`~glas.preview.Preview` only ever
   ``peek()``s), switching back once it stops.
2. :class:`~glas.gui.widgets.recording_controls_widget.RecordingControlsWidget`
   and :class:`~glas.gui.widgets.experiment_metadata_widget.ExperimentMetadataWidget`
   are separate panels, but starting a recording needs both: this window
   merges the metadata form's :class:`~glas.experiment.PhysicalParameters`
   into the ``extra`` dict every recording is started with.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from glas.acquisition import Acquisition
from glas.camera import Camera
from glas.controller import RecorderController
from glas.experiment import build_physical_parameters_extra
from glas.gui.logging_bridge import QtLogHandler
from glas.gui.theme import apply_theme
from glas.gui.viewmodels import (
    AnalysisViewModel,
    CameraViewModel,
    DatasetViewModel,
    HardwareStatusViewModel,
    LiveFeedViewModel,
    RecordingViewModel,
)
from glas.gui.widgets import (
    AnalysisPanelWidget,
    CameraControlsWidget,
    DatasetBrowserWidget,
    ExperimentMetadataWidget,
    HardwareStatusWidget,
    LivePreviewWidget,
    LogConsoleWidget,
    RecordingControlsWidget,
)
from glas.recorder import Recorder

_ORGANIZATION = "GLAS"
_APPLICATION = "GLAS"
_BOTTOM_DOCK_MAX_HEIGHT = 260
"""Height cap for the bottom dock group (Analysis/Dataset Browser/Log
Console), so it can never grow to compete with the live preview for
vertical space. `QMainWindow.resizeDocks()` cannot reliably size a dock
area relative to the central widget (only relative to sibling dock
widgets -- see `_apply_default_dock_sizes`'s docstring), so a maximum
size is the one technique that reliably holds up across window resizes.
The panel is still shrinkable and still scrolls its own content, so
nothing in it becomes unreachable."""


class MainWindow(QMainWindow):
    """The GLAS desktop application's main window.

    Parameters
    ----------
    base_data_dir : pathlib.Path
        Directory new experiment folders are created under.
    settings : QSettings, optional
        Backing store for window layout/theme persistence. Defaults to
        ``QSettings("GLAS", "GLAS")``; pass an isolated instance in tests
        so they don't read or write the real user's saved settings.
    """

    def __init__(
        self,
        base_data_dir: Path,
        settings: QSettings | None = None,
        parent: QMainWindow | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GLAS - Granular Lab Acquisition System")
        self._settings = (
            settings if settings is not None else QSettings(_ORGANIZATION, _APPLICATION)
        )

        self._camera = Camera()
        self._controller = RecorderController(base_data_dir, camera=self._camera)
        self._preview_acquisition: Acquisition | None = None

        self._camera_vm = CameraViewModel(camera=self._camera)
        self._recording_vm = RecordingViewModel(self._controller)
        self._live_feed_vm = LiveFeedViewModel()
        self._hardware_vm = HardwareStatusViewModel()
        self._dataset_vm = DatasetViewModel(base_data_dir)
        self._analysis_vm = AnalysisViewModel()

        self._log_handler = QtLogHandler()
        logging.getLogger("glas").addHandler(self._log_handler)

        self._live_preview_widget = LivePreviewWidget(self._live_feed_vm)
        self._camera_controls_widget = CameraControlsWidget(self._camera_vm)
        self._experiment_metadata_widget = ExperimentMetadataWidget()
        self._recording_controls_widget = RecordingControlsWidget(
            self._recording_vm,
            extra_provider=self._collect_physical_parameters_extra,
            before_start=self._release_preview_acquisition,
        )
        self._hardware_status_widget = HardwareStatusWidget(
            self._hardware_vm, self._camera_vm, self._recording_vm
        )
        self._analysis_panel_widget = AnalysisPanelWidget(self._analysis_vm)
        self._log_console_widget = LogConsoleWidget(self._log_handler)
        self._dataset_browser_widget = DatasetBrowserWidget(self._dataset_vm)

        self.setCentralWidget(self._live_preview_widget)
        self._docks: dict[str, QDockWidget] = {}
        self._add_dock(
            "Camera Controls", self._camera_controls_widget, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._add_dock(
            "Recording Controls",
            self._recording_controls_widget,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self._add_dock(
            "Experiment Metadata",
            self._experiment_metadata_widget,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self._add_dock(
            "Hardware Status", self._hardware_status_widget, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._add_dock(
            "Analysis", self._analysis_panel_widget, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._add_dock(
            "Dataset Browser", self._dataset_browser_widget, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._add_dock(
            "Log Console", self._log_console_widget, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._apply_default_layout()
        self._default_dock_sizes_applied = False

        self._build_menu_bar()
        self._build_status_bar()
        self._connect_orchestration_signals()
        self._had_saved_state = self._restore_layout()

    def _add_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> None:
        dock = QDockWidget(title, self)
        dock.setObjectName(title.replace(" ", ""))
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(area, dock)
        self._docks[title] = dock

    def _apply_default_layout(self) -> None:
        """Group docks so the live preview -- the primary tool, per the
        imaging-software convention set by pylon Viewer, ImageJ,
        Micro-Manager, NIS-Elements, and ZEN -- dominates the window
        (roughly 60-70% of its area) rather than competing with a wall of
        equally-sized side panels.

        Docks that are only glanced at occasionally (Experiment Metadata,
        Hardware Status; Dataset Browser, Log Console) are tabified behind
        the one most relevant during live operation (Recording Controls;
        Analysis) instead of each claiming their own strip of space --
        every panel stays reachable via its tab, `View`, or a floating
        drag, exactly as before, but only one at a time competes with the
        preview for room. The bottom group additionally gets a permanent
        height cap (see :data:`_BOTTOM_DOCK_MAX_HEIGHT`); the left/right
        docks' default widths are set separately by
        :meth:`_apply_default_dock_sizes`, once the window is shown --
        see its docstring for why that can't happen here.
        """
        self.resize(1600, 1000)

        self.tabifyDockWidget(self._docks["Recording Controls"], self._docks["Experiment Metadata"])
        self.tabifyDockWidget(self._docks["Recording Controls"], self._docks["Hardware Status"])
        self._docks["Recording Controls"].raise_()

        self.tabifyDockWidget(self._docks["Analysis"], self._docks["Dataset Browser"])
        self.tabifyDockWidget(self._docks["Analysis"], self._docks["Log Console"])
        self._docks["Log Console"].raise_()

        for name in ("Analysis", "Dataset Browser", "Log Console"):
            self._docks[name].setMaximumHeight(_BOTTOM_DOCK_MAX_HEIGHT)

    def _apply_default_dock_sizes(self) -> None:
        """Narrow the left/right docks so the preview gets the rest of the window's width.

        ``QMainWindow.resizeDocks()`` can only size dock widgets relative
        to *sibling dock widgets* in the same splitter row -- it has
        nothing to size the central widget (the live preview) relative to,
        so it silently no-ops here for the top/bottom split (handled
        instead by the bottom dock group's permanent height cap, see
        :data:`_BOTTOM_DOCK_MAX_HEIGHT`). It also does nothing until the
        dock area has actually been laid out once, which only happens
        once the window is shown -- calling it from ``__init__`` (before
        anyone has shown the window) silently does nothing either. Deferred
        to :meth:`showEvent` instead, and only for a first-ever launch
        (:attr:`_had_saved_state` is ``False``) -- a user's own saved
        layout must never be overridden.
        """
        self.resizeDocks(
            [self._docks["Camera Controls"], self._docks["Recording Controls"]],
            [280, 320],
            Qt.Orientation.Horizontal,
        )

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._had_saved_state and not self._default_dock_sizes_applied:
            self._default_dock_sizes_applied = True
            QTimer.singleShot(0, self._apply_default_dock_sizes)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        connect_action = file_menu.addAction("Connect Camera")
        connect_action.triggered.connect(lambda: self._camera_vm.connect_camera())
        disconnect_action = file_menu.addAction("Disconnect Camera")
        disconnect_action.triggered.connect(self._disconnect_camera)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = menu_bar.addMenu("&View")
        self._dark_mode_action = QAction("Dark Mode", self, checkable=True)
        self._dark_mode_action.toggled.connect(self._on_dark_mode_toggled)
        view_menu.addAction(self._dark_mode_action)
        view_menu.addSeparator()
        for dock in self._docks.values():
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        reset_layout_action = view_menu.addAction("Reset Layout")
        reset_layout_action.triggered.connect(self._reset_layout)

        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about_dialog)

    def _build_status_bar(self) -> None:
        status_bar = self.statusBar()
        self._camera_status_label = QLabel("Camera: disconnected")
        self._recording_status_label = QLabel("Recording: idle")
        status_bar.addPermanentWidget(self._camera_status_label)
        status_bar.addPermanentWidget(self._recording_status_label)

    def _connect_orchestration_signals(self) -> None:
        self._camera_vm.connected.connect(self._on_camera_connected)
        self._camera_vm.disconnected.connect(self._on_camera_disconnected)
        self._camera_vm.error_occurred.connect(self._show_error)

        self._recording_vm.recording_started.connect(self._on_recording_started)
        self._recording_vm.recording_stopped.connect(self._on_recording_stopped)
        self._recording_vm.error_occurred.connect(self._show_error)

        self._dataset_vm.error_occurred.connect(self._show_error)

    def _collect_physical_parameters_extra(self) -> dict[str, Any]:
        return build_physical_parameters_extra(self._experiment_metadata_widget.parameters())

    def _on_camera_connected(self) -> None:
        self._camera_status_label.setText(f"Camera: {self._camera.get_info().model_name}")
        self._preview_acquisition = Acquisition(self._camera)
        self._preview_acquisition.start()
        self._live_feed_vm.attach(self._preview_acquisition.buffer)

    def _disconnect_camera(self) -> None:
        """Release the preview acquisition, then disconnect the camera.

        Releasing first (not reacting to :attr:`CameraViewModel.disconnected`
        afterward) matters: :meth:`~glas.camera.Camera.disconnect` closes
        the camera before that signal fires, and the preview acquisition's
        background thread would otherwise still be mid-``RetrieveResult``
        against an already-closed camera.
        """
        self._release_preview_acquisition()
        self._camera_vm.disconnect_camera()

    def _on_camera_disconnected(self) -> None:
        self._camera_status_label.setText("Camera: disconnected")
        self._release_preview_acquisition()
        self._live_preview_widget.reset()

    def _release_preview_acquisition(self) -> None:
        """Synchronously stop and release the live-preview-only acquisition, if any.

        Called both on disconnect and (via ``before_start``) immediately
        before a recording starts -- :class:`~glas.acquisition.Acquisition.stop`
        blocks until its producer thread has actually exited and released
        the camera, so :class:`~glas.recorder.Recorder`'s own acquisition
        can never race this one for the same camera.
        """
        self._live_feed_vm.detach()
        if self._preview_acquisition is not None:
            self._preview_acquisition.stop()
            self._preview_acquisition = None

    def _on_recording_started(self, recorder: Recorder) -> None:
        self._recording_status_label.setText("Recording: active")
        self._live_feed_vm.attach(recorder.buffer)
        self._hardware_vm.attach(recorder.buffer, str(self._controller.base_data_dir))

    def _on_recording_stopped(self) -> None:
        self._recording_status_label.setText("Recording: idle")
        self._live_feed_vm.detach()
        self._hardware_vm.detach()
        if self._camera_vm.is_connected:
            self._preview_acquisition = Acquisition(self._camera)
            self._preview_acquisition.start()
            self._live_feed_vm.attach(self._preview_acquisition.buffer)
        self._dataset_browser_widget.refresh()

    def _on_dark_mode_toggled(self, checked: bool) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, dark=checked)
        self._settings.setValue("darkMode", checked)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "GLAS", message)

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About GLAS",
            "GLAS - Granular Lab Acquisition System\n"
            "A control and analysis platform for granular-material physics experiments.",
        )

    def _restore_layout(self) -> bool:
        """Restore a previously saved window geometry/dock layout, if any.

        Returns
        -------
        bool
            ``True`` if a saved ``windowState`` was found and restored --
            the signal :meth:`showEvent` uses to decide whether it's safe
            to apply the default dock proportions, or whether doing so
            would clobber the user's own saved layout.
        """
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        window_state = self._settings.value("windowState")
        had_saved_state = window_state is not None
        if had_saved_state:
            self.restoreState(window_state)
        dark_mode = self._settings.value("darkMode", False, type=bool)
        self._dark_mode_action.setChecked(bool(dark_mode))
        return had_saved_state

    def _reset_layout(self) -> None:
        self._settings.remove("geometry")
        self._settings.remove("windowState")
        QMessageBox.information(
            self, "GLAS", "Layout will reset to the default the next time GLAS is started."
        )

    def _save_layout(self) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._save_layout()
        if self._recording_vm.controller.progress() is not None:
            self._recording_vm.stop_recording()
        if self._camera_vm.is_connected:
            self._disconnect_camera()
        logging.getLogger("glas").removeHandler(self._log_handler)
        super().closeEvent(event)

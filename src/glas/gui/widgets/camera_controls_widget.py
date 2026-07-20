"""The camera controls panel: selection, exposure/gain/gamma, ROI, trigger, and flip.

Wraps :class:`~glas.gui.viewmodels.camera_viewmodel.CameraViewModel`. Every
control reads its current value and valid range/choices from the
connected :class:`~glas.camera.Camera` itself (via the ViewModel), and
every edit is routed through one of the ViewModel's ``set_*`` methods --
this widget never touches GenICam or raises/catches a camera exception
directly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glas.camera_info import CameraInfo
from glas.camera_validator import ROI
from glas.gui.status_indicators import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, status_dot_html
from glas.gui.viewmodels.camera_viewmodel import CameraViewModel

_AUTO_MODES = ("Off", "Once", "Continuous")
_BINNING_FACTORS = (1, 2, 4)


class CameraControlsWidget(QWidget):
    """The camera selection and parameter-control panel.

    Parameters
    ----------
    view_model : CameraViewModel
    """

    def __init__(self, view_model: CameraViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view_model = view_model
        self._syncing = False

        self._camera_combo = QComboBox()
        self._refresh_button = QPushButton("Refresh")
        self._connect_button = QPushButton("Connect")
        self._disconnect_button = QPushButton("Disconnect")
        self._status_label = QLabel(status_dot_html(COLOR_RED, "Not connected"))
        self._status_label.setTextFormat(Qt.TextFormat.RichText)

        self._pixel_format_combo = QComboBox()

        self._exposure_spin = self._make_double_spin(suffix=" µs")
        self._exposure_auto_combo = QComboBox()
        self._exposure_auto_combo.addItems(_AUTO_MODES)

        self._gain_spin = self._make_double_spin(suffix=" dB")
        self._gain_auto_combo = QComboBox()
        self._gain_auto_combo.addItems(_AUTO_MODES)

        self._gamma_spin = self._make_double_spin(suffix="")

        self._frame_rate_enabled_check = QCheckBox("Limit frame rate")
        self._frame_rate_spin = self._make_double_spin(suffix=" Hz")

        self._roi_width_spin = QSpinBox()
        self._roi_height_spin = QSpinBox()
        self._roi_offset_x_spin = QSpinBox()
        self._roi_offset_y_spin = QSpinBox()
        self._roi_apply_button = QPushButton("Apply ROI")
        self._roi_reset_button = QPushButton("Full Sensor")

        self._binning_h_combo = QComboBox()
        self._binning_v_combo = QComboBox()
        for combo in (self._binning_h_combo, self._binning_v_combo):
            combo.addItems([str(factor) for factor in _BINNING_FACTORS])

        self._reverse_x_check = QCheckBox("Flip horizontal")
        self._reverse_y_check = QCheckBox("Flip vertical")

        self._test_image_check = QCheckBox("Test image mode")
        self._test_image_check.setEnabled(False)
        self._test_image_check.setToolTip(
            "Not supported by the current camera abstraction on this device."
        )

        self._trigger_enabled_check = QCheckBox("Hardware trigger")
        self._trigger_source_combo = QComboBox()
        self._trigger_activation_combo = QComboBox()

        self._settings_widgets: list[QWidget] = [
            self._pixel_format_combo,
            self._exposure_spin,
            self._exposure_auto_combo,
            self._gain_spin,
            self._gain_auto_combo,
            self._gamma_spin,
            self._frame_rate_enabled_check,
            self._frame_rate_spin,
            self._roi_width_spin,
            self._roi_height_spin,
            self._roi_offset_x_spin,
            self._roi_offset_y_spin,
            self._roi_apply_button,
            self._roi_reset_button,
            self._binning_h_combo,
            self._binning_v_combo,
            self._reverse_x_check,
            self._reverse_y_check,
            self._trigger_enabled_check,
            self._trigger_source_combo,
            self._trigger_activation_combo,
        ]
        self._set_settings_enabled(False)

        self._build_layout()
        self._connect_signals()
        self.refresh_camera_list()

    def _make_double_spin(self, *, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setSuffix(suffix)
        spin.setDecimals(2)
        spin.setKeyboardTracking(False)
        return spin

    def _build_layout(self) -> None:
        camera_group = QGroupBox("Camera")
        camera_row = QHBoxLayout()
        camera_row.addWidget(self._camera_combo, stretch=1)
        camera_row.addWidget(self._refresh_button)
        camera_row.addWidget(self._connect_button)
        camera_row.addWidget(self._disconnect_button)
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.addLayout(camera_row)
        camera_layout.addWidget(self._status_label)

        image_group = QGroupBox("Image")
        image_form = QFormLayout(image_group)
        image_form.addRow("Pixel format:", self._pixel_format_combo)
        binning_row = QHBoxLayout()
        binning_row.addWidget(QLabel("H:"))
        binning_row.addWidget(self._binning_h_combo)
        binning_row.addWidget(QLabel("V:"))
        binning_row.addWidget(self._binning_v_combo)
        image_form.addRow("Binning:", binning_row)
        image_form.addRow(self._reverse_x_check)
        image_form.addRow(self._reverse_y_check)
        image_form.addRow(self._test_image_check)

        exposure_group = QGroupBox("Exposure && Gain")
        exposure_form = QFormLayout(exposure_group)
        exposure_form.addRow("Exposure time:", self._exposure_spin)
        exposure_form.addRow("Auto exposure:", self._exposure_auto_combo)
        exposure_form.addRow("Gain:", self._gain_spin)
        exposure_form.addRow("Auto gain:", self._gain_auto_combo)
        exposure_form.addRow("Gamma:", self._gamma_spin)

        rate_group = QGroupBox("Frame Rate")
        rate_form = QFormLayout(rate_group)
        rate_form.addRow(self._frame_rate_enabled_check)
        rate_form.addRow("Target rate:", self._frame_rate_spin)

        roi_group = QGroupBox("ROI")
        roi_form = QFormLayout(roi_group)
        roi_form.addRow("Width:", self._roi_width_spin)
        roi_form.addRow("Height:", self._roi_height_spin)
        roi_form.addRow("Offset X:", self._roi_offset_x_spin)
        roi_form.addRow("Offset Y:", self._roi_offset_y_spin)
        roi_buttons = QHBoxLayout()
        roi_buttons.addWidget(self._roi_apply_button)
        roi_buttons.addWidget(self._roi_reset_button)
        roi_form.addRow(roi_buttons)

        trigger_group = QGroupBox("Trigger")
        trigger_form = QFormLayout(trigger_group)
        trigger_form.addRow(self._trigger_enabled_check)
        trigger_form.addRow("Source:", self._trigger_source_combo)
        trigger_form.addRow("Activation:", self._trigger_activation_combo)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        for group in (
            camera_group,
            image_group,
            exposure_group,
            rate_group,
            roi_group,
            trigger_group,
        ):
            content_layout.addWidget(group)
        content_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _connect_signals(self) -> None:
        self._refresh_button.clicked.connect(self.refresh_camera_list)
        self._connect_button.clicked.connect(self._on_connect_clicked)
        self._disconnect_button.clicked.connect(self._view_model.disconnect_camera)

        self._view_model.connected.connect(self._on_connected)
        self._view_model.disconnected.connect(self._on_disconnected)
        self._view_model.settings_changed.connect(self._refresh_current_values)
        self._view_model.error_occurred.connect(self._on_error)

        self._pixel_format_combo.currentTextChanged.connect(self._on_pixel_format_changed)
        self._exposure_spin.editingFinished.connect(self._on_exposure_changed)
        self._exposure_auto_combo.currentTextChanged.connect(self._on_exposure_auto_changed)
        self._gain_spin.editingFinished.connect(self._on_gain_changed)
        self._gain_auto_combo.currentTextChanged.connect(self._on_gain_auto_changed)
        self._gamma_spin.editingFinished.connect(self._on_gamma_changed)
        self._frame_rate_enabled_check.toggled.connect(self._on_frame_rate_enabled_changed)
        self._frame_rate_spin.editingFinished.connect(self._on_frame_rate_changed)
        self._roi_apply_button.clicked.connect(self._on_roi_apply_clicked)
        self._roi_reset_button.clicked.connect(self._on_roi_reset_clicked)
        self._binning_h_combo.currentTextChanged.connect(self._on_binning_changed)
        self._binning_v_combo.currentTextChanged.connect(self._on_binning_changed)
        self._reverse_x_check.toggled.connect(self._view_model.set_reverse_x)
        self._reverse_y_check.toggled.connect(self._view_model.set_reverse_y)
        self._trigger_enabled_check.toggled.connect(self._on_trigger_toggled)

    def refresh_camera_list(self) -> None:
        """Re-populate the camera-selection combo box from :meth:`CameraViewModel.list_cameras`."""
        self._camera_combo.clear()
        for info in self._view_model.list_cameras():
            self._camera_combo.addItem(f"{info.model_name} ({info.serial_number})", info)

    def _set_settings_enabled(self, enabled: bool) -> None:
        for widget in self._settings_widgets:
            widget.setEnabled(enabled)

    def _on_connect_clicked(self) -> None:
        info: CameraInfo | None = self._camera_combo.currentData()
        self._status_label.setText(status_dot_html(COLOR_YELLOW, "Connecting..."))
        # connect_camera() is a blocking GenICam call; force a repaint now so
        # the "Connecting..." state is actually visible rather than jumping
        # straight from "Not connected" to the final result.
        QApplication.processEvents()
        self._view_model.connect_camera(serial_number=info.serial_number if info else None)

    def _on_connected(self, info: CameraInfo) -> None:
        self._status_label.setText(
            status_dot_html(COLOR_GREEN, f"Connected: {info.model_name} ({info.serial_number})")
        )
        self._set_settings_enabled(True)
        self._populate_choices()
        self._refresh_current_values()

    def _on_disconnected(self) -> None:
        self._status_label.setText(status_dot_html(COLOR_RED, "Not connected"))
        self._set_settings_enabled(False)

    def _on_error(self, message: str) -> None:
        self._status_label.setText(status_dot_html(COLOR_RED, f"Error: {message}"))

    def _populate_choices(self) -> None:
        camera = self._view_model.camera
        self._syncing = True
        try:
            self._pixel_format_combo.clear()
            self._pixel_format_combo.addItems(camera.pixel_format_choices())
            self._trigger_source_combo.clear()
            self._trigger_source_combo.addItems(camera.trigger_source_choices())
            self._trigger_activation_combo.clear()
            self._trigger_activation_combo.addItems(camera.trigger_activation_choices())

            exposure_bounds = camera.exposure_time_bounds_us()
            self._exposure_spin.setRange(exposure_bounds.minimum, exposure_bounds.maximum)
            gain_bounds = camera.gain_bounds_db()
            self._gain_spin.setRange(gain_bounds.minimum, gain_bounds.maximum)
            gamma_bounds = camera.gamma_bounds()
            self._gamma_spin.setRange(gamma_bounds.minimum, gamma_bounds.maximum)
            frame_rate_bounds = camera.frame_rate_bounds_hz()
            self._frame_rate_spin.setRange(frame_rate_bounds.minimum, frame_rate_bounds.maximum)

            roi_bounds = camera.roi_bounds()
            self._roi_width_spin.setRange(
                int(roi_bounds.width.minimum), int(roi_bounds.sensor_width)
            )
            self._roi_height_spin.setRange(
                int(roi_bounds.height.minimum), int(roi_bounds.sensor_height)
            )
            self._roi_offset_x_spin.setRange(0, int(roi_bounds.sensor_width))
            self._roi_offset_y_spin.setRange(0, int(roi_bounds.sensor_height))
        finally:
            self._syncing = False

    def _refresh_current_values(self) -> None:
        if not self._view_model.is_connected:
            return
        camera = self._view_model.camera
        self._syncing = True
        try:
            self._pixel_format_combo.setCurrentText(camera.pixel_format)
            self._exposure_spin.setValue(camera.exposure_time_us)
            self._exposure_auto_combo.setCurrentText(camera.exposure_auto)
            self._gain_spin.setValue(camera.gain_db)
            self._gain_auto_combo.setCurrentText(camera.gain_auto)
            self._gamma_spin.setValue(camera.gamma)
            self._frame_rate_enabled_check.setChecked(camera.frame_rate_enabled)
            self._frame_rate_spin.setValue(camera.frame_rate_hz)

            roi = camera.roi
            self._roi_width_spin.setValue(roi.width)
            self._roi_height_spin.setValue(roi.height)
            self._roi_offset_x_spin.setValue(roi.offset_x)
            self._roi_offset_y_spin.setValue(roi.offset_y)

            horizontal, vertical = camera.binning
            self._binning_h_combo.setCurrentText(str(horizontal))
            self._binning_v_combo.setCurrentText(str(vertical))

            self._reverse_x_check.setChecked(camera.reverse_x)
            self._reverse_y_check.setChecked(camera.reverse_y)

            self._trigger_enabled_check.setChecked(camera.is_hardware_triggered())
        finally:
            self._syncing = False

    def _on_pixel_format_changed(self, value: str) -> None:
        if not self._syncing and value:
            self._view_model.set_pixel_format(value)

    def _on_exposure_changed(self) -> None:
        if not self._syncing:
            self._view_model.set_exposure_time_us(self._exposure_spin.value())

    def _on_exposure_auto_changed(self, value: str) -> None:
        if not self._syncing and value:
            self._view_model.set_exposure_auto(value)

    def _on_gain_changed(self) -> None:
        if not self._syncing:
            self._view_model.set_gain_db(self._gain_spin.value())

    def _on_gain_auto_changed(self, value: str) -> None:
        if not self._syncing and value:
            self._view_model.set_gain_auto(value)

    def _on_gamma_changed(self) -> None:
        if not self._syncing:
            self._view_model.set_gamma(self._gamma_spin.value())

    def _on_frame_rate_enabled_changed(self, checked: bool) -> None:
        if not self._syncing:
            self._view_model.set_frame_rate_enabled(checked)

    def _on_frame_rate_changed(self) -> None:
        if not self._syncing:
            self._view_model.set_frame_rate_hz(self._frame_rate_spin.value())

    def _on_roi_apply_clicked(self) -> None:
        roi = ROI(
            width=self._roi_width_spin.value(),
            height=self._roi_height_spin.value(),
            offset_x=self._roi_offset_x_spin.value(),
            offset_y=self._roi_offset_y_spin.value(),
        )
        self._view_model.set_roi(roi)

    def _on_roi_reset_clicked(self) -> None:
        bounds = self._view_model.camera.roi_bounds()
        roi = ROI(
            width=int(bounds.sensor_width),
            height=int(bounds.sensor_height),
            offset_x=0,
            offset_y=0,
        )
        self._view_model.set_roi(roi)

    def _on_binning_changed(self, _value: str) -> None:
        if not self._syncing:
            self._view_model.set_binning(
                int(self._binning_h_combo.currentText()),
                int(self._binning_v_combo.currentText()),
            )

    def _on_trigger_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        self._view_model.set_hardware_trigger(
            checked,
            source=self._trigger_source_combo.currentText() or "Line1",
            activation=self._trigger_activation_combo.currentText() or "RisingEdge",
        )

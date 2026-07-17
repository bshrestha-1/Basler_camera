"""The experiment metadata panel: scientific parameters attached to a recording.

Collects :class:`~glas.experiment.PhysicalParameters` from a form. Unlike
most widgets in this package, this one has no ViewModel of its own to
wrap -- it makes no backend calls and polls nothing, it only marshals form
values into the existing, Qt-free :class:`~glas.experiment.PhysicalParameters`
value object, exactly the way :class:`~glas.camera_validator.ROI` is
constructed directly from spin-box values in
:class:`~glas.gui.widgets.camera_controls_widget.CameraControlsWidget`.
Whatever owns both this widget and
:class:`~glas.gui.widgets.recording_controls_widget.RecordingControlsWidget`
(the main window) is responsible for passing :meth:`parameters`' result
into :meth:`~glas.gui.viewmodels.recording_viewmodel.RecordingViewModel.start_recording`'s
``extra`` argument via :func:`~glas.experiment.build_physical_parameters_extra`.

Notes/Tags are deliberately not duplicated here -- they are exactly
``DatasetMetadata.notes``/the reserved ``experiment_tags`` extra key, both
already collected once, in
:class:`~glas.gui.widgets.recording_controls_widget.RecordingControlsWidget`,
where "Start" is actually clicked.
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from glas.experiment import PhysicalParameters

_CLOCK_UPDATE_INTERVAL_MS = 1000


def _none_if_zero(value: float) -> float | None:
    """Treat a spin box left at its default ``0.0`` as "not entered".

    None of these physical fields (grain diameter, density, fill depth,
    frequency, amplitude, target acceleration) is a meaningful measurement
    at exactly zero, so the spin box's unavoidable numeric default doubles
    as the "operator hasn't filled this in yet" state.
    """
    return value if value != 0.0 else None


def _make_double_spin(*, suffix: str, maximum: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setSuffix(suffix)
    spin.setDecimals(3)
    spin.setRange(0.0, maximum)
    return spin


class ExperimentMetadataWidget(QWidget):
    """A form collecting :class:`~glas.experiment.PhysicalParameters` for the current session."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._datetime_label = QLabel()
        self._experiment_id_edit = QLineEdit()
        self._operator_edit = QLineEdit()
        self._material_edit = QLineEdit()
        self._grain_diameter_spin = _make_double_spin(suffix=" mm", maximum=1000.0)
        self._grain_density_spin = _make_double_spin(suffix=" kg/m³", maximum=100_000.0)
        self._container_geometry_edit = QLineEdit()
        self._fill_depth_spin = _make_double_spin(suffix=" mm", maximum=10_000.0)
        self._frequency_spin = _make_double_spin(suffix=" Hz", maximum=10_000.0)
        self._amplitude_spin = _make_double_spin(suffix=" mm", maximum=1000.0)
        self._target_acceleration_spin = _make_double_spin(suffix=" g", maximum=1000.0)

        form = QFormLayout()
        form.addRow("Date/Time:", self._datetime_label)
        form.addRow("Experiment ID:", self._experiment_id_edit)
        form.addRow("Operator:", self._operator_edit)
        form.addRow("Material:", self._material_edit)
        form.addRow("Grain diameter:", self._grain_diameter_spin)
        form.addRow("Grain density:", self._grain_density_spin)
        form.addRow("Container geometry:", self._container_geometry_edit)
        form.addRow("Fill depth:", self._fill_depth_spin)
        form.addRow("Frequency:", self._frequency_spin)
        form.addRow("Amplitude:", self._amplitude_spin)
        form.addRow("Target acceleration:", self._target_acceleration_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(_CLOCK_UPDATE_INTERVAL_MS)
        self._clock_timer.timeout.connect(self._update_clock)
        self._update_clock()
        self._clock_timer.start()

    def _update_clock(self) -> None:
        now = datetime.now(timezone.utc).astimezone()
        self._datetime_label.setText(now.strftime("%Y-%m-%d %H:%M:%S %Z"))

    def parameters(self) -> PhysicalParameters:
        """Build a :class:`~glas.experiment.PhysicalParameters` from the current form values."""
        return PhysicalParameters(
            experiment_id=self._experiment_id_edit.text(),
            operator=self._operator_edit.text(),
            material=self._material_edit.text(),
            grain_diameter_mm=_none_if_zero(self._grain_diameter_spin.value()),
            grain_density_kg_m3=_none_if_zero(self._grain_density_spin.value()),
            container_geometry=self._container_geometry_edit.text(),
            fill_depth_mm=_none_if_zero(self._fill_depth_spin.value()),
            frequency_hz=_none_if_zero(self._frequency_spin.value()),
            amplitude_mm=_none_if_zero(self._amplitude_spin.value()),
            target_acceleration_g=_none_if_zero(self._target_acceleration_spin.value()),
        )

    def set_parameters(self, parameters: PhysicalParameters) -> None:
        """Populate the form from an existing :class:`~glas.experiment.PhysicalParameters`."""
        self._experiment_id_edit.setText(parameters.experiment_id)
        self._operator_edit.setText(parameters.operator)
        self._material_edit.setText(parameters.material)
        self._grain_diameter_spin.setValue(parameters.grain_diameter_mm or 0.0)
        self._grain_density_spin.setValue(parameters.grain_density_kg_m3 or 0.0)
        self._container_geometry_edit.setText(parameters.container_geometry)
        self._fill_depth_spin.setValue(parameters.fill_depth_mm or 0.0)
        self._frequency_spin.setValue(parameters.frequency_hz or 0.0)
        self._amplitude_spin.setValue(parameters.amplitude_mm or 0.0)
        self._target_acceleration_spin.setValue(parameters.target_acceleration_g or 0.0)

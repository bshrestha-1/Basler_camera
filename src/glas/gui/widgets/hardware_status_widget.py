"""The hardware status panel: camera link, pipeline, system, and recorder state at a glance.

Wraps three ViewModels rather than one -- a status dashboard is
inherently a rollup across domains, unlike every other widget in this
package, which owns exactly one. The "Other Devices" section renders
whatever :class:`~glas.gui.viewmodels.hardware_status_viewmodel.HardwareStatusViewModel`'s
device registry currently holds, so a future LabJack/NI DAQ/accelerometer/
function-generator/amplifier integration that calls
:meth:`~glas.gui.viewmodels.hardware_status_viewmodel.HardwareStatusViewModel.register_device`
appears here automatically -- this widget never needs to change to show it.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from glas.gui.viewmodels.camera_viewmodel import CameraViewModel
from glas.gui.viewmodels.hardware_status_viewmodel import DeviceStatus, HardwareStatusViewModel
from glas.gui.viewmodels.recording_viewmodel import RecordingViewModel
from glas.monitor import PerformanceSnapshot
from glas.recorder import RecorderProgress, RecorderState


def _format_bps(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1e6:.1f} Mbps"


class HardwareStatusWidget(QWidget):
    """Read-only dashboard of camera, pipeline, system, recorder, and other-device status.

    Parameters
    ----------
    hardware_view_model : HardwareStatusViewModel
    camera_view_model : CameraViewModel
    recording_view_model : RecordingViewModel
    """

    def __init__(
        self,
        hardware_view_model: HardwareStatusViewModel,
        camera_view_model: CameraViewModel,
        recording_view_model: RecordingViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._hardware_view_model = hardware_view_model
        self._camera_view_model = camera_view_model
        self._recording_view_model = recording_view_model

        self._camera_connected_label = QLabel("Disconnected")
        self._usb_bandwidth_label = QLabel("N/A")
        self._temperature_label = QLabel("N/A")
        self._frame_rate_label = QLabel("--")
        self._exposure_label = QLabel("--")
        self._gain_label = QLabel("--")
        self._sync_status_label = QLabel("Free-running")

        self._buffer_usage_label = QLabel("--")
        self._memory_usage_label = QLabel("--")
        self._cpu_usage_label = QLabel("--")
        self._storage_remaining_label = QLabel("--")

        self._recorder_status_label = QLabel("Idle")

        self._devices_group = QGroupBox("Other Devices")
        self._devices_layout = QFormLayout(self._devices_group)
        self._devices_group.setVisible(False)
        self._device_labels: dict[str, QLabel] = {}

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        camera_group = QGroupBox("Camera")
        camera_form = QFormLayout(camera_group)
        camera_form.addRow("Connected:", self._camera_connected_label)
        camera_form.addRow("USB bandwidth:", self._usb_bandwidth_label)
        camera_form.addRow("Temperature:", self._temperature_label)
        camera_form.addRow("Frame rate:", self._frame_rate_label)
        camera_form.addRow("Exposure:", self._exposure_label)
        camera_form.addRow("Gain:", self._gain_label)
        camera_form.addRow("Synchronization:", self._sync_status_label)

        pipeline_group = QGroupBox("Pipeline / System")
        pipeline_form = QFormLayout(pipeline_group)
        pipeline_form.addRow("Buffer usage:", self._buffer_usage_label)
        pipeline_form.addRow("Memory usage:", self._memory_usage_label)
        pipeline_form.addRow("CPU usage:", self._cpu_usage_label)
        pipeline_form.addRow("Storage remaining:", self._storage_remaining_label)

        recorder_group = QGroupBox("Recorder")
        recorder_form = QFormLayout(recorder_group)
        recorder_form.addRow("Status:", self._recorder_status_label)

        layout = QVBoxLayout(self)
        layout.addWidget(camera_group)
        layout.addWidget(pipeline_group)
        layout.addWidget(recorder_group)
        layout.addWidget(self._devices_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._camera_view_model.connected.connect(self._on_camera_connected)
        self._camera_view_model.disconnected.connect(self._on_camera_disconnected)
        self._camera_view_model.settings_changed.connect(self._refresh_camera_values)

        self._hardware_view_model.status_updated.connect(self._on_status_updated)

        self._recording_view_model.recording_started.connect(
            lambda: self._recorder_status_label.setText("Recording")
        )
        self._recording_view_model.recording_paused.connect(
            lambda: self._recorder_status_label.setText("Paused")
        )
        self._recording_view_model.recording_resumed.connect(
            lambda: self._recorder_status_label.setText("Recording")
        )
        self._recording_view_model.recording_stopped.connect(
            lambda: self._recorder_status_label.setText("Idle")
        )
        self._recording_view_model.progress_updated.connect(self._on_progress_updated)

    def _on_camera_connected(self) -> None:
        info = self._camera_view_model.camera.get_info()
        self._camera_connected_label.setText(f"Connected ({info.model_name})")
        self._refresh_camera_values()

    def _on_camera_disconnected(self) -> None:
        self._camera_connected_label.setText("Disconnected")
        self._usb_bandwidth_label.setText("N/A")
        self._temperature_label.setText("N/A")
        self._exposure_label.setText("--")
        self._gain_label.setText("--")
        self._sync_status_label.setText("Free-running")

    def _refresh_camera_values(self) -> None:
        if not self._camera_view_model.is_connected:
            return
        camera = self._camera_view_model.camera

        diagnostics = camera.get_usb_diagnostics()
        self._usb_bandwidth_label.setText(
            f"{_format_bps(diagnostics.link_speed_bps)} "
            f"(max {_format_bps(diagnostics.max_bandwidth_bps)})"
        )

        temperature = camera.temperature_celsius()
        self._temperature_label.setText(
            f"{temperature:.1f} °C" if temperature is not None else "N/A"
        )

        self._exposure_label.setText(f"{camera.exposure_time_us:.1f} µs")
        self._gain_label.setText(f"{camera.gain_db:.1f} dB")
        self._sync_status_label.setText(
            "Hardware-triggered" if camera.is_hardware_triggered() else "Free-running"
        )

    def _on_status_updated(
        self, snapshot: PerformanceSnapshot | None, devices: dict[str, DeviceStatus]
    ) -> None:
        if snapshot is not None:
            self._frame_rate_label.setText(f"{snapshot.fps:.1f} fps")
            self._buffer_usage_label.setText(
                f"{snapshot.buffer_size}/{snapshot.buffer_capacity} "
                f"({snapshot.buffer_occupancy_percent:.0f}%)"
            )
            self._memory_usage_label.setText(
                f"{snapshot.memory_used_mb:.0f} MB ({snapshot.memory_percent:.1f}%)"
            )
            self._cpu_usage_label.setText(f"{snapshot.cpu_percent:.1f}%")
            self._storage_remaining_label.setText(
                f"{snapshot.disk_free_gb:.1f} GB free "
                f"({100 - snapshot.disk_used_percent:.0f}% available)"
            )
        self._sync_devices(devices)

    def _on_progress_updated(self, progress: RecorderProgress) -> None:
        self._recorder_status_label.setText(progress.state.value.capitalize())
        if progress.state == RecorderState.RECORDING:
            self._recorder_status_label.setText(f"Recording ({progress.frame_count} frames)")

    def _sync_devices(self, devices: dict[str, DeviceStatus]) -> None:
        for name in list(self._device_labels):
            if name not in devices:
                label = self._device_labels.pop(name)
                self._devices_layout.removeRow(label)

        for name, status in devices.items():
            text = f"{'Connected' if status.connected else 'Disconnected'}"
            if status.detail:
                text += f" ({status.detail})"
            if name in self._device_labels:
                self._device_labels[name].setText(text)
            else:
                label = QLabel(text)
                self._device_labels[name] = label
                self._devices_layout.addRow(f"{name}:", label)

        self._devices_group.setVisible(bool(devices))

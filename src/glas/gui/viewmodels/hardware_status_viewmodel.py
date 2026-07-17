"""ViewModel driving the hardware status widget.

Polls :meth:`~glas.monitor.PerformanceMonitor.sample` on a
:class:`~PySide6.QtCore.QTimer` for camera/pipeline/system stats, and
maintains a small, open-ended registry of :class:`DeviceStatus` entries
for hardware without its own always-on polling loop (a
:class:`~glas.hardware.waveform_generator.SiglentSDG1032X`, a
:class:`~glas.hardware.daq.LabJackDAQ`/:class:`~glas.hardware.daq.NiDAQ`,
an accelerometer). Registering a new device kind never requires touching
:class:`HardwareStatusViewModel` or the widget it drives -- both just
render whatever is currently registered.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from PySide6.QtCore import QObject, QTimer, Signal

from glas.monitor import PerformanceMonitor, PerformanceSnapshot
from glas.ringbuffer import RingBuffer

DEFAULT_POLL_INTERVAL_MS = 1000


class DeviceStatus(BaseModel):
    """A single hardware device's connection status, for the status panel.

    Attributes
    ----------
    name : str
        Display name, e.g. ``"LabJack T7"`` or ``"Siglent SDG1032X"``.
    connected : bool
        Whether the device is currently reachable.
    detail : str
        Free-text status detail, e.g. an IP address, a serial number, or
        an error message.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    connected: bool
    detail: str = ""


class HardwareStatusViewModel(QObject):
    """Aggregates camera/pipeline/system stats and an open-ended device registry.

    Signals
    -------
    status_updated(object, dict)
        Emitted on every poll and every :meth:`register_device` call, as
        ``(PerformanceSnapshot | None, dict[str, DeviceStatus])``. The
        snapshot is ``None`` until :meth:`attach` has been called (e.g.
        before any recording has started).
    """

    status_updated = Signal(object, dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._monitor: PerformanceMonitor | None = None
        self._devices: dict[str, DeviceStatus] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)

    def attach(self, buffer: RingBuffer, data_dir: str) -> None:
        """Start polling pipeline/system stats for a recording's buffer and data directory."""
        self._monitor = PerformanceMonitor(buffer, data_dir)
        self._timer.start()
        self._poll()

    def detach(self) -> None:
        """Stop polling pipeline/system stats (registered devices are unaffected)."""
        self._timer.stop()
        self._monitor = None

    def register_device(self, status: DeviceStatus) -> None:
        """Add or update one non-camera device's status.

        Future-proofing hook: a LabJack/NI DAQ/accelerometer/function-
        generator/amplifier/environmental-sensor integration calls this
        (typically after its own connect/disconnect or a periodic health
        check) to appear in the status panel -- no changes to this class
        or the widget it drives are needed.
        """
        self._devices[status.name] = status
        self._emit(self._monitor.sample() if self._monitor is not None else None)

    def unregister_device(self, name: str) -> None:
        """Remove a previously registered device's status entry."""
        self._devices.pop(name, None)
        self._emit(self._monitor.sample() if self._monitor is not None else None)

    def _poll(self) -> None:
        if self._monitor is None:
            return
        self._emit(self._monitor.sample())

    def _emit(self, snapshot: PerformanceSnapshot | None) -> None:
        self.status_updated.emit(snapshot, dict(self._devices))

"""ViewModel wrapping :class:`glas.camera.Camera` for camera-control widgets.

No business logic lives here beyond adapting :class:`~glas.camera.Camera`'s
synchronous API to Qt signals a widget can bind to -- GenICam access and
parameter validation stay in :mod:`glas.camera`, exactly as they are for
the CLI. Every setting-mutating method (:meth:`set_exposure_time_us`,
:meth:`set_gain_db`, etc.) is a thin wrapper around one
:class:`~glas.camera.Camera` property assignment, routed through
:meth:`_apply` so a bad value surfaces as :attr:`error_occurred` instead
of an uncaught exception reaching the widget.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from glas.camera import Camera
from glas.camera_info import CameraInfo, detect_cameras
from glas.camera_validator import ROI
from glas.exceptions import CameraError


class CameraViewModel(QObject):
    """Connects/disconnects a camera and exposes its live parameters to the GUI.

    Signals
    -------
    connected(CameraInfo)
        Emitted after a successful :meth:`connect_camera`.
    disconnected()
        Emitted after :meth:`disconnect_camera`.
    settings_changed()
        Emitted after any ``set_*`` method successfully applies a new
        value.
    error_occurred(str)
        Emitted instead of ``connected``/``settings_changed`` when an
        operation fails, with a human-readable message.
    """

    connected = Signal(object)
    disconnected = Signal()
    settings_changed = Signal()
    error_occurred = Signal(str)

    def __init__(self, camera: Camera | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.camera: Camera = camera if camera is not None else Camera()

    @property
    def is_connected(self) -> bool:
        """``True`` if a device is currently open."""
        return self.camera.is_connected

    def list_cameras(self) -> list[CameraInfo]:
        """Enumerate every camera (real or emulated) currently reachable."""
        return detect_cameras()

    def connect_camera(self, serial_number: str | None = None) -> None:
        """Connect to a camera, emitting :attr:`connected` or :attr:`error_occurred`."""
        try:
            info = self.camera.connect(serial_number=serial_number)
        except CameraError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.connected.emit(info)

    def disconnect_camera(self) -> None:
        """Disconnect the camera, emitting :attr:`disconnected`."""
        try:
            self.camera.disconnect()
        except CameraError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.disconnected.emit()

    def _apply(self, action: Callable[[], None]) -> None:
        """Run a single camera-mutating callable, translating failures to a signal."""
        try:
            action()
        except CameraError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.settings_changed.emit()

    def set_pixel_format(self, value: str) -> None:
        """Set :attr:`~glas.camera.Camera.pixel_format`."""
        self._apply(lambda: setattr(self.camera, "pixel_format", value))

    def set_exposure_time_us(self, value: float) -> None:
        """Set :attr:`~glas.camera.Camera.exposure_time_us`."""
        self._apply(lambda: setattr(self.camera, "exposure_time_us", value))

    def set_gain_db(self, value: float) -> None:
        """Set :attr:`~glas.camera.Camera.gain_db`."""
        self._apply(lambda: setattr(self.camera, "gain_db", value))

    def set_gamma(self, value: float) -> None:
        """Set :attr:`~glas.camera.Camera.gamma`."""
        self._apply(lambda: setattr(self.camera, "gamma", value))

    def set_frame_rate_hz(self, value: float) -> None:
        """Set :attr:`~glas.camera.Camera.frame_rate_hz`."""
        self._apply(lambda: setattr(self.camera, "frame_rate_hz", value))

    def set_frame_rate_enabled(self, value: bool) -> None:
        """Set :attr:`~glas.camera.Camera.frame_rate_enabled`."""
        self._apply(lambda: setattr(self.camera, "frame_rate_enabled", value))

    def set_roi(self, value: ROI) -> None:
        """Set :attr:`~glas.camera.Camera.roi`."""
        self._apply(lambda: setattr(self.camera, "roi", value))

    def set_binning(self, horizontal: int, vertical: int) -> None:
        """Set :attr:`~glas.camera.Camera.binning`."""
        self._apply(lambda: setattr(self.camera, "binning", (horizontal, vertical)))

    def set_reverse_x(self, value: bool) -> None:
        """Set :attr:`~glas.camera.Camera.reverse_x`."""
        self._apply(lambda: setattr(self.camera, "reverse_x", value))

    def set_reverse_y(self, value: bool) -> None:
        """Set :attr:`~glas.camera.Camera.reverse_y`."""
        self._apply(lambda: setattr(self.camera, "reverse_y", value))

    def set_exposure_auto(self, value: str) -> None:
        """Set :attr:`~glas.camera.Camera.exposure_auto`."""
        self._apply(lambda: setattr(self.camera, "exposure_auto", value))

    def set_gain_auto(self, value: str) -> None:
        """Set :attr:`~glas.camera.Camera.gain_auto`."""
        self._apply(lambda: setattr(self.camera, "gain_auto", value))

    def set_hardware_trigger(
        self,
        enabled: bool,
        *,
        source: str = "Line1",
        activation: str = "RisingEdge",
    ) -> None:
        """Enable or disable hardware triggering.

        See :meth:`~glas.camera.Camera.enable_hardware_trigger` and
        :meth:`~glas.camera.Camera.disable_hardware_trigger`.
        """
        if enabled:
            self._apply(
                lambda: self.camera.enable_hardware_trigger(source=source, activation=activation)
            )
        else:
            self._apply(self.camera.disable_hardware_trigger)

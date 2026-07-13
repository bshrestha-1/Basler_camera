"""Control of a single Basler camera via pypylon.

:class:`Camera` connects to a Basler USB3 Vision device (such as the ace
acA640-750um), and exposes exposure, gain, region of interest, and pixel
format as validated properties. No image acquisition happens in this
module -- see the acquisition layer (Phase 3) for continuous frame
capture.
"""

from __future__ import annotations

import time
from types import TracebackType

from glas.camera_info import CameraInfo, UsbDiagnostics
from glas.camera_info import get_usb_diagnostics as _get_usb_diagnostics
from glas.camera_validator import (
    ROI,
    NumericRange,
    ROIBounds,
    validate_exposure_time,
    validate_gain,
    validate_pixel_format,
    validate_roi,
)
from glas.exceptions import (
    AcquisitionError,
    CameraConnectionError,
    CameraDriverError,
    CameraFeatureUnavailableError,
    CameraNotFoundError,
)
from glas.frame import Frame
from glas.logger import get_logger

logger = get_logger(__name__)

try:
    from pypylon import genicam, pylon
except ImportError as _exc:  # pragma: no cover - exercised in environments without pypylon
    pylon = None
    genicam = None
    _PYLON_IMPORT_ERROR: Exception | None = _exc
else:
    _PYLON_IMPORT_ERROR = None


def _require_pylon() -> None:
    """Raise :class:`~glas.exceptions.CameraDriverError` if pypylon is unavailable."""
    if pylon is None:
        raise CameraDriverError(
            "pypylon is not installed. Install the Basler pypylon package "
            "(`pip install pypylon`) to use camera features."
        ) from _PYLON_IMPORT_ERROR


def _node_readable(node_map: pylon.NodeMapWrapper, name: str) -> bool:
    node = node_map.GetNode(name)
    return node is not None and genicam.IsReadable(node)


def _node_executable(node_map: pylon.NodeMapWrapper, name: str) -> bool:
    node = node_map.GetNode(name)
    return node is not None and genicam.IsWritable(node)


class Camera:
    """A connection to a single Basler camera.

    Examples
    --------
    >>> with Camera() as camera:  # doctest: +SKIP
    ...     camera.exposure_time_us = 5000.0
    ...     camera.gain_db = 6.0
    ...     print(camera.get_info().model_name)
    """

    def __init__(self) -> None:
        self._camera: pylon.InstantCamera | None = None

    @property
    def is_connected(self) -> bool:
        """``True`` if a device is currently open."""
        return self._camera is not None and self._camera.IsOpen()

    def connect(self, serial_number: str | None = None) -> CameraInfo:
        """Detect and open a Basler camera.

        Parameters
        ----------
        serial_number : str, optional
            Serial number of a specific camera to connect to. If ``None``,
            the first camera pypylon detects is used.

        Returns
        -------
        CameraInfo
            Identifying information for the connected camera.

        Raises
        ------
        CameraDriverError
            If pypylon is not installed or the transport layer cannot be
            queried.
        CameraNotFoundError
            If no camera is detected, or none matches ``serial_number``.
        CameraConnectionError
            If a camera is already connected, or the device cannot be
            opened.
        """
        _require_pylon()
        if self.is_connected:
            raise CameraConnectionError("Camera is already connected; call disconnect() first.")

        try:
            available = pylon.TlFactory.GetInstance().EnumerateDevices()
        except genicam.GenericException as exc:
            raise CameraDriverError(f"Could not enumerate cameras: {exc}") from exc

        if serial_number is not None:
            matches = [info for info in available if info.GetSerialNumber() == serial_number]
            if not matches:
                raise CameraNotFoundError(f"No camera found with serial number {serial_number!r}.")
            device_info = matches[0]
        else:
            if not available:
                raise CameraNotFoundError("No Basler camera detected.")
            device_info = available[0]

        try:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
            camera.Open()
        except genicam.GenericException as exc:
            raise CameraConnectionError(f"Could not connect to camera: {exc}") from exc

        self._camera = camera
        info = self.get_info()
        logger.info("Connected to %s (serial=%s)", info.model_name, info.serial_number)
        return info

    def disconnect(self) -> None:
        """Close the camera connection, if open. Safe to call when already closed."""
        if self._camera is None:
            return
        try:
            if self._camera.IsOpen():
                self._camera.Close()
        finally:
            self._camera = None
        logger.info("Camera disconnected.")

    def __enter__(self) -> Camera:
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    def _require_camera(self) -> pylon.InstantCamera:
        if self._camera is None or not self._camera.IsOpen():
            raise CameraConnectionError("Camera is not connected. Call connect() first.")
        return self._camera

    def get_info(self) -> CameraInfo:
        """Return identifying information for the connected camera.

        Returns
        -------
        CameraInfo

        Raises
        ------
        CameraConnectionError
            If no camera is connected.
        """
        camera = self._require_camera()
        return CameraInfo.from_device_info(camera.GetDeviceInfo())

    def get_usb_diagnostics(self) -> UsbDiagnostics:
        """Return USB link diagnostics for the connected camera.

        Returns
        -------
        UsbDiagnostics

        Raises
        ------
        CameraConnectionError
            If no camera is connected.
        """
        camera = self._require_camera()
        return _get_usb_diagnostics(camera)

    @property
    def exposure_time_us(self) -> float:
        """Exposure time, in microseconds."""
        camera = self._require_camera()
        return float(camera.ExposureTime.GetValue())

    @exposure_time_us.setter
    def exposure_time_us(self, value: float) -> None:
        camera = self._require_camera()
        node = camera.ExposureTime
        bounds = NumericRange(minimum=node.GetMin(), maximum=node.GetMax())
        node.SetValue(validate_exposure_time(value, bounds))

    @property
    def gain_db(self) -> float:
        """Analog gain, in dB."""
        camera = self._require_camera()
        return float(camera.Gain.GetValue())

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        camera = self._require_camera()
        node = camera.Gain
        bounds = NumericRange(minimum=node.GetMin(), maximum=node.GetMax())
        node.SetValue(validate_gain(value, bounds))

    @property
    def roi(self) -> ROI:
        """Current region of interest (crop size and offset, in pixels)."""
        camera = self._require_camera()
        return ROI(
            width=camera.Width.GetValue(),
            height=camera.Height.GetValue(),
            offset_x=camera.OffsetX.GetValue(),
            offset_y=camera.OffsetY.GetValue(),
        )

    @roi.setter
    def roi(self, value: ROI) -> None:
        camera = self._require_camera()

        # Reset offsets to zero first so Width/Height report the sensor's
        # true maximum rather than a range already shrunk by the current
        # offset, then re-validate against those un-shrunk bounds.
        camera.OffsetX.SetValue(0)
        camera.OffsetY.SetValue(0)

        bounds = ROIBounds(
            width=NumericRange(minimum=camera.Width.GetMin(), maximum=camera.Width.GetMax()),
            height=NumericRange(minimum=camera.Height.GetMin(), maximum=camera.Height.GetMax()),
            offset_x=NumericRange(minimum=camera.OffsetX.GetMin(), maximum=camera.OffsetX.GetMax()),
            offset_y=NumericRange(minimum=camera.OffsetY.GetMin(), maximum=camera.OffsetY.GetMax()),
            sensor_width=camera.WidthMax.GetValue(),
            sensor_height=camera.HeightMax.GetValue(),
            width_step=camera.Width.GetInc(),
            height_step=camera.Height.GetInc(),
            offset_x_step=camera.OffsetX.GetInc(),
            offset_y_step=camera.OffsetY.GetInc(),
        )
        validated = validate_roi(value, bounds)

        camera.Width.SetValue(validated.width)
        camera.Height.SetValue(validated.height)
        camera.OffsetX.SetValue(validated.offset_x)
        camera.OffsetY.SetValue(validated.offset_y)

    @property
    def pixel_format(self) -> str:
        """Current pixel format, e.g. ``"Mono8"``."""
        camera = self._require_camera()
        return str(camera.PixelFormat.GetValue())

    @pixel_format.setter
    def pixel_format(self, value: str) -> None:
        camera = self._require_camera()
        node = camera.PixelFormat
        node.SetValue(validate_pixel_format(value, node.GetSymbolics()))

    @property
    def supports_hardware_timestamp(self) -> bool:
        """``True`` if the connected camera exposes a hardware timestamp counter."""
        camera = self._require_camera()
        node_map = camera.GetNodeMap()
        return _node_readable(node_map, "Timestamp") or (
            _node_executable(node_map, "TimestampLatch")
            and _node_readable(node_map, "TimestampLatchValue")
        )

    def get_timestamp(self) -> int:
        """Read the camera's current hardware timestamp counter value.

        Returns
        -------
        int
            Raw device timestamp, in ticks (frequency is device-specific).

        Raises
        ------
        CameraConnectionError
            If no camera is connected.
        CameraFeatureUnavailableError
            If the connected device does not expose a timestamp counter.
        """
        camera = self._require_camera()
        node_map = camera.GetNodeMap()

        if _node_readable(node_map, "Timestamp"):
            return int(node_map.GetNode("Timestamp").GetValue())

        if _node_executable(node_map, "TimestampLatch") and _node_readable(
            node_map, "TimestampLatchValue"
        ):
            node_map.GetNode("TimestampLatch").Execute()
            return int(node_map.GetNode("TimestampLatchValue").GetValue())

        raise CameraFeatureUnavailableError(
            "This camera does not expose a hardware timestamp counter."
        )

    def start_grabbing(self) -> None:
        """Begin continuous frame grabbing.

        Uses pylon's ``GrabStrategy_OneByOne``, which delivers every
        frame the camera produces, in acquisition order -- frame loss
        under this strategy only happens at the driver/USB level
        (reported by :meth:`retrieve_frame` raising
        :class:`~glas.exceptions.AcquisitionError`), not through pylon
        silently discarding frames to keep only the latest one. Safe to
        call when already grabbing.

        Raises
        ------
        CameraConnectionError
            If no camera is connected.
        """
        camera = self._require_camera()
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    def stop_grabbing(self) -> None:
        """Stop continuous frame grabbing. Safe to call when not grabbing or not connected."""
        if self._camera is not None and self._camera.IsOpen() and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    @property
    def is_grabbing(self) -> bool:
        """``True`` if the camera is currently in continuous grab mode."""
        return self._camera is not None and self._camera.IsOpen() and self._camera.IsGrabbing()

    def retrieve_frame(self, frame_id: int, timeout_ms: int = 1000) -> Frame | None:
        """Retrieve the next frame from the camera.

        Must be called after :meth:`start_grabbing`.

        Parameters
        ----------
        frame_id : int
            Sequence number to stamp onto the returned
            :class:`~glas.frame.Frame`. ``Camera`` does not track
            acquisition-session frame numbering itself; the caller
            (typically :class:`~glas.acquisition.Acquisition`) supplies
            it.
        timeout_ms : int, default 1000
            Maximum time to wait for a frame to arrive.

        Returns
        -------
        Frame or None
            The retrieved frame, or ``None`` if no frame arrived within
            ``timeout_ms`` -- an expected, non-error condition in a
            polling acquisition loop.

        Raises
        ------
        CameraConnectionError
            If no camera is connected, or the camera is not currently
            grabbing.
        AcquisitionError
            If the grab itself failed (e.g. an incomplete frame reported
            by the driver), as opposed to an ordinary timeout.
        """
        camera = self._require_camera()
        if not camera.IsGrabbing():
            raise CameraConnectionError("Camera is not grabbing. Call start_grabbing() first.")

        try:
            result = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        except genicam.GenericException as exc:
            raise AcquisitionError(f"Grab failed: {exc}") from exc

        try:
            if not result.IsValid():
                return None  # ordinary timeout, not an error

            if not result.GrabSucceeded():
                raise AcquisitionError(
                    f"Grab failed: {result.GetErrorDescription()} "
                    f"(error code {result.GetErrorCode()})."
                )

            return Frame(
                frame_id=frame_id,
                image=result.Array.copy(),
                pixel_format=self.pixel_format,
                host_timestamp_ns=time.perf_counter_ns(),
                device_timestamp_ticks=int(result.GetTimeStamp()),
            )
        finally:
            result.Release()

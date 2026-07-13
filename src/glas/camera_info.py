"""Camera discovery and device information for Basler cameras.

Wraps the parts of the pypylon transport-layer API needed to enumerate
connected Basler devices and read USB link diagnostics, without requiring
an already-open :class:`~glas.camera.Camera`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from glas.exceptions import CameraDriverError
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


class CameraInfo(BaseModel):
    """Static identifying information for a Basler camera device.

    Attributes
    ----------
    serial_number : str
        Device serial number, unique per physical camera.
    model_name : str
        Camera model, e.g. ``"acA640-750um"``.
    vendor_name : str
        Device vendor, e.g. ``"Basler"``.
    device_class : str
        Transport layer device class, e.g. ``"BaslerUsb"``.
    friendly_name : str
        Human-readable name combining model and serial number.
    full_name : str
        Fully qualified device identifier used internally by pylon.
    """

    model_config = ConfigDict(frozen=True)

    serial_number: str
    model_name: str
    vendor_name: str
    device_class: str
    friendly_name: str
    full_name: str

    @classmethod
    def from_device_info(cls, device_info: pylon.DeviceInfo) -> CameraInfo:
        """Build a :class:`CameraInfo` from a pypylon device info object.

        Parameters
        ----------
        device_info : pypylon.pylon.DeviceInfo
            Device metadata as returned by
            :meth:`pypylon.pylon.TlFactory.EnumerateDevices` or
            :meth:`pypylon.pylon.InstantCamera.GetDeviceInfo`.

        Returns
        -------
        CameraInfo
        """
        return cls(
            serial_number=device_info.GetSerialNumber(),
            model_name=device_info.GetModelName(),
            vendor_name=device_info.GetVendorName(),
            device_class=device_info.GetDeviceClass(),
            friendly_name=device_info.GetFriendlyName(),
            full_name=device_info.GetFullName(),
        )


def detect_cameras() -> list[CameraInfo]:
    """Enumerate all Basler cameras currently reachable by pylon.

    Returns
    -------
    list of CameraInfo
        One entry per detected device. Empty if no camera is connected;
        that is a normal result, not an error.

    Raises
    ------
    CameraDriverError
        If pypylon is not installed or the transport layer cannot be
        queried.
    """
    _require_pylon()
    try:
        device_infos = pylon.TlFactory.GetInstance().EnumerateDevices()
    except genicam.GenericException as exc:
        raise CameraDriverError(f"Could not enumerate cameras: {exc}") from exc

    cameras = [CameraInfo.from_device_info(info) for info in device_infos]
    logger.debug("Detected %d camera(s).", len(cameras))
    return cameras


class UsbDiagnostics(BaseModel):
    """USB link diagnostics for a connected camera.

    Fields are ``None`` when the connected device does not expose the
    corresponding node -- for example, non-USB transports or emulated
    cameras. USB diagnostics are informational and vary by transport layer
    and device model, so missing data is not treated as an error.

    Attributes
    ----------
    link_speed_bps : int or None
        Negotiated USB link speed, in bits per second.
    max_bandwidth_bps : int or None
        Maximum bandwidth available to the device, in bits per second.
    throughput_limit_bps : int or None
        Configured device link throughput limit, in bits per second.
    throughput_limit_enabled : bool or None
        Whether the throughput limit is currently enforced.
    usb_speed_mode : str or None
        Negotiated USB speed mode, e.g. ``"SuperSpeed"``.
    """

    model_config = ConfigDict(frozen=True)

    link_speed_bps: int | None
    max_bandwidth_bps: int | None
    throughput_limit_bps: int | None
    throughput_limit_enabled: bool | None
    usb_speed_mode: str | None


def _first_readable(
    candidates: Iterable[tuple[pylon.NodeMapWrapper, str]],
) -> Any | None:
    for node_map, name in candidates:
        node = node_map.GetNode(name)
        if node is not None and genicam.IsReadable(node):
            return node.GetValue()
    return None


def _read_int_if_available(node_map: pylon.NodeMapWrapper, name: str) -> int | None:
    node = node_map.GetNode(name)
    if node is None or not genicam.IsReadable(node):
        return None
    return int(node.GetValue())


def get_usb_diagnostics(camera: pylon.InstantCamera) -> UsbDiagnostics:
    """Read USB link diagnostics from an open camera.

    Checks a curated set of candidate node names across the device's main,
    transport-layer, and stream-grabber node maps, since the exact node
    that exposes link speed / bandwidth varies between Basler device
    families and firmware versions.

    Parameters
    ----------
    camera : pypylon.pylon.InstantCamera
        An already-open camera instance.

    Returns
    -------
    UsbDiagnostics
    """
    _require_pylon()
    node_map = camera.GetNodeMap()
    tl_node_map = camera.GetTLNodeMap()
    try:
        stream_node_map = camera.GetStreamGrabberNodeMap()
    except genicam.GenericException:
        stream_node_map = None

    node_maps = [node_map, tl_node_map]
    if stream_node_map is not None:
        node_maps.append(stream_node_map)

    link_speed = _first_readable(
        (nm, name)
        for nm in node_maps
        for name in ("DeviceLinkSpeed", "BslDeviceLinkSpeed", "LinkSpeed")
    )
    max_bandwidth = _first_readable(
        (nm, name) for nm in node_maps for name in ("MaxBandwidth", "MaximumTransferSize")
    )
    usb_speed_mode = _first_readable(
        (nm, name)
        for nm in node_maps
        for name in ("BslUSBSpeedMode", "BslUsbSpeedMode", "UsbSpeedMode")
    )

    throughput_limit_mode = node_map.GetNode("DeviceLinkThroughputLimitMode")
    throughput_limit_enabled = (
        str(throughput_limit_mode.GetValue()) == "On"
        if throughput_limit_mode is not None and genicam.IsReadable(throughput_limit_mode)
        else None
    )

    return UsbDiagnostics(
        link_speed_bps=int(link_speed) if link_speed is not None else None,
        max_bandwidth_bps=int(max_bandwidth) if max_bandwidth is not None else None,
        throughput_limit_bps=_read_int_if_available(node_map, "DeviceLinkThroughputLimit"),
        throughput_limit_enabled=throughput_limit_enabled,
        usb_speed_mode=str(usb_speed_mode) if usb_speed_mode is not None else None,
    )

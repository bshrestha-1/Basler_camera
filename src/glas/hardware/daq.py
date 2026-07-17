"""Analog input DAQ (data acquisition) device support: LabJack and National Instruments.

Both vendors' Python SDKs (``labjack-ljm`` and ``nidaqmx``) require a
proprietary driver runtime to be installed on the host, and neither is a
hard dependency of GLAS -- most installations won't have this specific
hardware. Each class here defers importing its vendor SDK until
:meth:`~AnalogInputDAQ.connect` is called (mirroring how
:mod:`glas.camera` defers importing ``pypylon``), and accepts the SDK
module itself as an injectable constructor parameter so the surrounding
logic (channel bookkeeping, error wrapping) can be unit-tested with a
fake module standing in for the real one, with no physical DAQ or vendor
driver installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from glas.exceptions import InstrumentConnectionError


class AnalogInputDAQ(ABC):
    """Common interface for reading analog input channels from a DAQ device."""

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the device.

        Raises
        ------
        InstrumentConnectionError
            If the vendor SDK is not installed, or the device cannot be
            reached.
        """

    @abstractmethod
    def read_channel(self, channel: int) -> float:
        """Read a single analog input channel, in volts.

        Raises
        ------
        InstrumentConnectionError
            If not connected, or the read fails.
        """

    def read_channels(self, channels: Sequence[int]) -> dict[int, float]:
        """Read several analog input channels, in volts.

        The default implementation calls :meth:`read_channel` once per
        channel; subclasses may override this if their SDK supports a
        more efficient batched read.

        Raises
        ------
        InstrumentConnectionError
            If not connected, or a read fails.
        """
        return {channel: self.read_channel(channel) for channel in channels}

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the device. Safe to call when not connected."""

    def __enter__(self) -> AnalogInputDAQ:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class LabJackDAQ(AnalogInputDAQ):
    """Analog input via a LabJack DAQ (T-series or U-series), using LabJack's LJM library.

    Parameters
    ----------
    device_type : str, default "ANY"
    connection_type : str, default "ANY"
    identifier : str, default "ANY"
        Passed straight through to ``ljm.openS()`` -- see LabJack's LJM
        User's Guide for the accepted values (e.g. ``device_type="T7"``,
        ``connection_type="USB"``, or ``identifier="470012345"`` for a
        specific unit's serial number).
    """

    def __init__(
        self,
        *,
        device_type: str = "ANY",
        connection_type: str = "ANY",
        identifier: str = "ANY",
        _ljm: Any | None = None,
    ) -> None:
        self._device_type = device_type
        self._connection_type = connection_type
        self._identifier = identifier
        self._ljm = _ljm
        self._handle: int | None = None

    def connect(self) -> None:
        if self._ljm is None:
            try:
                from labjack import ljm
            except ImportError as exc:
                raise InstrumentConnectionError(
                    "labjack-ljm is not installed. Install it (`pip install labjack-ljm`) "
                    "and the LabJack LJM driver to use LabJackDAQ."
                ) from exc
            self._ljm = ljm

        try:
            self._handle = self._ljm.openS(
                self._device_type, self._connection_type, self._identifier
            )
        except Exception as exc:
            # LJM raises its own exception type (ljm.LJMError), which we
            # can't reference by name without an unconditional import --
            # wrapping the generic Exception here is deliberate, not a
            # style lapse.
            raise InstrumentConnectionError(f"Could not connect to LabJack device: {exc}") from exc

    def _require_handle(self) -> int:
        if self._handle is None:
            raise InstrumentConnectionError("LabJackDAQ is not connected. Call connect() first.")
        return self._handle

    def read_channel(self, channel: int) -> float:
        handle = self._require_handle()
        assert self._ljm is not None  # connect() sets both together
        try:
            return float(self._ljm.eReadName(handle, f"AIN{channel}"))
        except Exception as exc:
            raise InstrumentConnectionError(f"Could not read AIN{channel}: {exc}") from exc

    def close(self) -> None:
        if self._handle is not None and self._ljm is not None:
            self._ljm.close(self._handle)
        self._handle = None


class NiDAQ(AnalogInputDAQ):
    """Analog input via a National Instruments DAQ device, using the ``nidaqmx`` package.

    Parameters
    ----------
    device_name : str
        The device's name as configured in NI-MAX (e.g. ``"Dev1"``).
    """

    def __init__(self, device_name: str, *, _nidaqmx: Any | None = None) -> None:
        self._device_name = device_name
        self._nidaqmx = _nidaqmx
        self._connected = False

    def connect(self) -> None:
        if self._nidaqmx is None:
            try:
                import nidaqmx
            except ImportError as exc:
                raise InstrumentConnectionError(
                    "nidaqmx is not installed. Install it (`pip install nidaqmx`) and the "
                    "NI-DAQmx driver runtime to use NiDAQ."
                ) from exc
            self._nidaqmx = nidaqmx
        self._connected = True

    def _require_connected(self) -> None:
        if not self._connected:
            raise InstrumentConnectionError("NiDAQ is not connected. Call connect() first.")

    def read_channel(self, channel: int) -> float:
        self._require_connected()
        assert self._nidaqmx is not None  # connect() sets this
        try:
            # A short-lived task per read -- the standard nidaqmx pattern
            # for a one-off acquisition, and simpler and more robust than
            # mutating a long-lived task's channel list between reads.
            with self._nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(f"{self._device_name}/ai{channel}")
                return float(task.read())
        except Exception as exc:
            raise InstrumentConnectionError(
                f"Could not read {self._device_name}/ai{channel}: {exc}"
            ) from exc

    def close(self) -> None:
        self._connected = False

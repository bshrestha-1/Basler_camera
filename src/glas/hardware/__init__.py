"""Lab instrument integration: function generators, shakers, DAQ, and oscilloscopes.

Basler camera hardware triggering lives on :class:`glas.camera.Camera`
itself (:meth:`~glas.camera.Camera.enable_hardware_trigger` and friends),
since it extends the camera's existing GenICam feature access rather than
talking to a separate instrument.

Every class here is built on an injectable transport or vendor-SDK
module, so command-building and error-handling logic is unit-testable
without physical hardware -- see each module's own docstring for how.
"""

from __future__ import annotations

from glas.hardware.daq import AnalogInputDAQ, LabJackDAQ, NiDAQ
from glas.hardware.oscilloscope import SCPIOscilloscope
from glas.hardware.scpi import (
    DEFAULT_SCPI_PORT,
    SCPIInstrument,
    SCPITransport,
    SocketSCPITransport,
)
from glas.hardware.shaker import ShakerCalibration, ShakerController
from glas.hardware.waveform_generator import SiglentSDG1032X

__all__ = [
    "DEFAULT_SCPI_PORT",
    "SCPITransport",
    "SocketSCPITransport",
    "SCPIInstrument",
    "SiglentSDG1032X",
    "SCPIOscilloscope",
    "ShakerCalibration",
    "ShakerController",
    "AnalogInputDAQ",
    "LabJackDAQ",
    "NiDAQ",
]

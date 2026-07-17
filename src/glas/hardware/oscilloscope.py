"""Generic SCPI oscilloscope wrapper.

Unlike the Siglent SDG1032X (whose ``BSWV`` command syntax is documented
and stable across the whole SDG-X series), oscilloscope SCPI dialects
vary significantly between vendors -- Siglent SDS, Keysight InfiniiVision,
Rigol DS/MSO, Tektronix, and others each use different command syntax
even for basic operations like reading a channel's voltage. Rather than
guess at (and risk getting wrong) a specific vendor's command set, this
module only adds the universal IEEE 488.2 commands inherited from
:class:`~glas.hardware.scpi.SCPIInstrument`, plus one small,
vendor-agnostic helper for the common pattern of a SCPI measurement query
returning a bare number. Use :meth:`~glas.hardware.scpi.SCPIInstrument.write`/
:meth:`~glas.hardware.scpi.SCPIInstrument.query` directly, with the exact
command syntax from the connected oscilloscope's own programming guide,
for anything model-specific (channel scaling, trigger setup, waveform
capture, measurement selection).
"""

from __future__ import annotations

from glas.exceptions import InstrumentCommandError
from glas.hardware.scpi import SCPIInstrument


class SCPIOscilloscope(SCPIInstrument):
    """A generic SCPI-compliant oscilloscope.

    Parameters
    ----------
    transport : SCPITransport
        An already-connected SCPI transport (e.g.
        :class:`~glas.hardware.scpi.SocketSCPITransport`, or a fake for
        testing).
    """

    def query_float(self, command: str) -> float:
        """Send a raw SCPI query and parse its response as a float.

        Parameters
        ----------
        command : str
            A raw SCPI query expected to return a single numeric value,
            e.g. a measurement query for a channel's peak-to-peak
            voltage. The exact command syntax depends on the connected
            oscilloscope's SCPI dialect -- see its programming guide.

        Returns
        -------
        float

        Raises
        ------
        InstrumentCommandError
            If the response is not a valid number.
        """
        response = self.query(command)
        try:
            return float(response)
        except ValueError as exc:
            raise InstrumentCommandError(
                f"Expected a numeric response to {command!r}, got {response!r}."
            ) from exc

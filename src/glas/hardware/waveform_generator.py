"""Siglent SDG1032X arbitrary waveform generator control.

Built on :class:`~glas.hardware.scpi.SCPIInstrument`, using the
``BSWV`` ("basic wave") command family documented in Siglent's SDG-X
series programming guide -- the common subset needed to drive a shaker
at a target sinusoidal frequency and amplitude. Consult the SDG1032X's
own programming guide for anything beyond this (arbitrary waveforms,
modulation, sweeps, and so on).
"""

from __future__ import annotations

from glas.hardware.scpi import SCPIInstrument

_CHANNELS = (1, 2)


def _require_channel(channel: int) -> None:
    if channel not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}, got {channel}.")


class SiglentSDG1032X(SCPIInstrument):
    """A Siglent SDG1032X dual-channel arbitrary waveform generator.

    Parameters
    ----------
    transport : SCPITransport
        An already-connected SCPI transport (e.g.
        :class:`~glas.hardware.scpi.SocketSCPITransport` pointed at the
        instrument's IP address on port 5025, or a fake for testing).
    """

    def set_sine_wave(
        self,
        channel: int,
        *,
        frequency_hz: float,
        amplitude_vpp: float,
        offset_v: float = 0.0,
        phase_deg: float = 0.0,
    ) -> None:
        """Configure a channel to output a sine wave.

        Parameters
        ----------
        channel : int
            Output channel, ``1`` or ``2``.
        frequency_hz : float
            Frequency, in Hz. Must be positive.
        amplitude_vpp : float
            Peak-to-peak amplitude, in volts. Must be positive.
        offset_v : float, default 0.0
            DC offset, in volts.
        phase_deg : float, default 0.0
            Phase, in degrees.

        Raises
        ------
        ValueError
            If ``channel`` is not ``1`` or ``2``, or ``frequency_hz``/
            ``amplitude_vpp`` is not positive.
        """
        _require_channel(channel)
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")
        if amplitude_vpp <= 0:
            raise ValueError(f"amplitude_vpp must be positive, got {amplitude_vpp}.")

        self.write(
            f"C{channel}:BSWV WVTP,SINE,"
            f"FRQ,{frequency_hz}HZ,"
            f"AMP,{amplitude_vpp}V,"
            f"OFST,{offset_v}V,"
            f"PHSE,{phase_deg}"
        )

    def set_frequency(self, channel: int, frequency_hz: float) -> None:
        """Change a channel's frequency without altering its other wave settings.

        Parameters
        ----------
        channel : int
            Output channel, ``1`` or ``2``.
        frequency_hz : float
            Frequency, in Hz. Must be positive.

        Raises
        ------
        ValueError
            If ``channel`` is not ``1`` or ``2``, or ``frequency_hz`` is
            not positive.
        """
        _require_channel(channel)
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")
        self.write(f"C{channel}:BSWV FRQ,{frequency_hz}HZ")

    def set_amplitude(self, channel: int, amplitude_vpp: float) -> None:
        """Change a channel's amplitude without altering its other wave settings.

        Parameters
        ----------
        channel : int
            Output channel, ``1`` or ``2``.
        amplitude_vpp : float
            Peak-to-peak amplitude, in volts. Must be positive.

        Raises
        ------
        ValueError
            If ``channel`` is not ``1`` or ``2``, or ``amplitude_vpp`` is
            not positive.
        """
        _require_channel(channel)
        if amplitude_vpp <= 0:
            raise ValueError(f"amplitude_vpp must be positive, got {amplitude_vpp}.")
        self.write(f"C{channel}:BSWV AMP,{amplitude_vpp}V")

    def enable_output(self, channel: int) -> None:
        """Turn a channel's output on.

        Parameters
        ----------
        channel : int
            Output channel, ``1`` or ``2``.

        Raises
        ------
        ValueError
            If ``channel`` is not ``1`` or ``2``.
        """
        _require_channel(channel)
        self.write(f"C{channel}:OUTP ON")

    def disable_output(self, channel: int) -> None:
        """Turn a channel's output off.

        Parameters
        ----------
        channel : int
            Output channel, ``1`` or ``2``.

        Raises
        ------
        ValueError
            If ``channel`` is not ``1`` or ``2``.
        """
        _require_channel(channel)
        self.write(f"C{channel}:OUTP OFF")

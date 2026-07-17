"""Shaker control (Modal Shop 2025E), driven via the SDG1032X waveform generator.

The Modal Shop 2025E is a power amplifier that drives a shaker's coil
directly from an analog input signal -- it's a gain stage, not a
digitally addressable instrument, so there is no SCPI (or any other)
protocol to talk to it over. What GLAS *can* control digitally is the
waveform generator feeding the amplifier's input: frequency and drive
voltage. This module computes the drive voltage needed to reach a target
vibration intensity (Gamma, see :func:`glas.accelerometer.compute_gamma`)
at a given frequency, from a calibration curve measured once for the
specific shaker + amplifier + fixture + mounted-mass combination in use,
then sends it to the waveform generator.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from glas.hardware.waveform_generator import SiglentSDG1032X

DEFAULT_CHANNEL = 1


class ShakerCalibration(BaseModel):
    """The measured relationship between drive voltage and resulting acceleration.

    A shaker + amplifier + mounted-fixture system's gain (how many g of
    acceleration result from a given drive voltage) depends on all of
    those together, and on frequency (most shakers are not flat across
    their operating range) -- there is no universal constant to assume.
    Measure ``volts_per_g`` at the frequency intended for use (drive a
    known small voltage, read the resulting peak acceleration from an
    accelerometer via :mod:`glas.accelerometer`, and divide), and
    re-measure if the fixture or mounted mass changes.

    Attributes
    ----------
    volts_per_g : float
        Drive voltage (peak-to-peak, at the waveform generator's output)
        required to produce 1 g of peak acceleration at the shaker head,
        at :attr:`frequency_hz`.
    frequency_hz : float
        The frequency this calibration was measured at.
    """

    model_config = ConfigDict(frozen=True)

    volts_per_g: float = Field(gt=0)
    frequency_hz: float = Field(gt=0)


class ShakerController:
    """Drives a Modal Shop 2025E shaker (via its power amplifier) to a target Gamma.

    The 2025E itself is an analog gain stage with no digital control
    interface -- this controller only manages the SDG1032X waveform
    generator feeding it, computing the drive voltage a
    :class:`ShakerCalibration` says will produce the requested vibration
    intensity.

    Parameters
    ----------
    generator : SiglentSDG1032X
        The waveform generator driving the amplifier's input.
    calibration : ShakerCalibration
        This shaker/amplifier/fixture combination's measured
        voltage-to-acceleration relationship.
    channel : int, default 1
        Which generator channel is wired to the amplifier.
    """

    def __init__(
        self,
        generator: SiglentSDG1032X,
        calibration: ShakerCalibration,
        *,
        channel: int = DEFAULT_CHANNEL,
    ) -> None:
        self._generator = generator
        self._calibration = calibration
        self._channel = channel

    def required_drive_voltage(self, gamma: float) -> float:
        """Compute the drive voltage needed to reach a target Gamma.

        Parameters
        ----------
        gamma : float
            Target dimensionless vibration intensity (see
            :func:`glas.accelerometer.compute_gamma`) -- peak
            acceleration in units of g. Must be positive.

        Returns
        -------
        float
            Required peak-to-peak drive voltage, in volts, per this
            controller's calibration.

        Raises
        ------
        ValueError
            If ``gamma`` is not positive.
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        return gamma * self._calibration.volts_per_g

    def set_target_gamma(self, gamma: float, *, frequency_hz: float | None = None) -> float:
        """Drive the shaker to a target Gamma at a given frequency.

        Parameters
        ----------
        gamma : float
            Target Gamma. Must be positive.
        frequency_hz : float, optional
            Drive frequency, in Hz. Defaults to the frequency the
            calibration was measured at -- pass an explicit value only
            after confirming the shaker's response is flat enough over
            the range in use that the same calibration still applies
            (see :class:`ShakerCalibration`).

        Returns
        -------
        float
            The drive voltage sent to the waveform generator, in volts.

        Raises
        ------
        ValueError
            If ``gamma`` or ``frequency_hz`` is not positive.
        """
        if frequency_hz is None:
            frequency_hz = self._calibration.frequency_hz
        elif frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")

        voltage = self.required_drive_voltage(gamma)
        self._generator.set_sine_wave(
            self._channel, frequency_hz=frequency_hz, amplitude_vpp=voltage
        )
        return voltage

    def start(self) -> None:
        """Enable the waveform generator's output, starting the shaker."""
        self._generator.enable_output(self._channel)

    def stop(self) -> None:
        """Disable the waveform generator's output, stopping the shaker."""
        self._generator.disable_output(self._channel)

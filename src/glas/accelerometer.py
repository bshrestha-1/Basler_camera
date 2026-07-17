"""Accelerometer import, vibration analysis, and frame synchronization.

Imports a PCB Piezotronics 352C22 accelerometer recording (exported as a
CSV file by whatever DAQ software captured it), computes the standard
vibration diagnostics used in granular-material physics -- frequency,
displacement amplitude, and the dimensionless acceleration Gamma -- and
aligns the recording's timeline with a recorded camera dataset's frames::

    import_accelerometer_csv() -> AccelerometerRecording -> compute_vibration_frequency()
                                            |                          |
                                            |                 compute_vibration_amplitude()
                                            |                          |
                                            |                    compute_gamma()
                                            |
                                  synchronize_with_frames()

:func:`analyze_vibration` runs the frequency/amplitude/Gamma pipeline
over a CSV file in one call, the same role
:func:`glas.analysis.analyze_packing` plays for its own phase.
Hardware-triggered synchronization (so the two clocks share a common
zero point exactly) is Phase 17's concern; here, :func:`synchronize_with_frames`
aligns the two recordings' relative timelines, assuming they started at
the same moment unless an explicit ``offset_s`` is given.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- see glas.analysis.brazil_nut
# for why this is always headless-safe, unconditionally, with no pre-flight check needed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from glas.exceptions import AccelerometerError  # noqa: E402
from glas.frame import Frame  # noqa: E402
from glas.plotting import apply_publication_style, savefig_publication, style_axes  # noqa: E402

DEFAULT_SENSITIVITY_MV_PER_G = 10.0
"""Nominal PCB 352C22 sensitivity, in mV/g. This varies by unit and is
printed on its calibration certificate -- use the specific accelerometer's
actual calibrated value for accurate results; this default is only a
typical starting point for a quick check."""

GRAVITY_M_S2 = 9.80665
"""Standard gravity, in m/s^2 (used to convert acceleration in g to SI units)."""

ValueUnits = Literal["g", "volts"]


@dataclass(frozen=True)
class AccelerometerRecording:
    """A single accelerometer channel's time-series recording.

    Like :class:`glas.frame.Frame`, a plain dataclass rather than a
    Pydantic model, since its fields are numpy arrays (see ``Frame``'s
    own docstring for why).

    Attributes
    ----------
    times_s : numpy.ndarray
        Sample timestamps, in seconds, relative to the start of this
        recording (i.e. ``times_s[0]`` need not be ``0``, but the array
        is strictly increasing).
    acceleration_g : numpy.ndarray
        Acceleration at each sample, in units of standard gravity (g).
    """

    times_s: NDArray[np.float64]
    acceleration_g: NDArray[np.float64]


class VibrationMetrics(BaseModel):
    """Frequency, amplitude, and Gamma for a single accelerometer recording.

    Attributes
    ----------
    frequency_hz : float
        The dominant vibration frequency, in Hz (see
        :func:`compute_vibration_frequency`).
    amplitude_m : float
        Displacement amplitude, in meters, derived from the peak
        acceleration and the dominant frequency assuming sinusoidal
        motion (see :func:`compute_vibration_amplitude`).
    gamma : float
        The dimensionless vibration intensity ``Gamma = peak
        acceleration / g``, the standard control parameter for vibrated
        granular beds (see :func:`compute_gamma`).
    peak_acceleration_g : float
        The peak (maximum absolute) measured acceleration, in g.
    """

    model_config = ConfigDict(frozen=True)

    frequency_hz: float
    amplitude_m: float
    gamma: float
    peak_acceleration_g: float


def import_accelerometer_csv(
    path: Path,
    *,
    time_column: str = "time_s",
    value_column: str = "voltage_v",
    value_units: ValueUnits = "volts",
    sensitivity_mv_per_g: float = DEFAULT_SENSITIVITY_MV_PER_G,
) -> AccelerometerRecording:
    """Import a PCB 352C22 accelerometer recording from a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        CSV file with a header row, as exported by DAQ software. Must
        contain at least ``time_column`` and ``value_column``.
    time_column : str, default "time_s"
        Header of the column holding sample timestamps, in seconds.
    value_column : str, default "voltage_v"
        Header of the column holding the measured value.
    value_units : {"volts", "g"}, default "volts"
        Whether ``value_column`` already holds acceleration in g
        (``"g"``, no conversion applied), or raw sensor output voltage
        (``"volts"``, converted via ``sensitivity_mv_per_g``).
    sensitivity_mv_per_g : float, default 10.0
        The accelerometer's calibrated sensitivity, in mV/g. Only used
        when ``value_units="volts"``. Use the value from the specific
        unit's calibration certificate, not the nominal default, for
        accurate results.

    Returns
    -------
    AccelerometerRecording

    Raises
    ------
    AccelerometerError
        If ``path`` doesn't exist or can't be read, the CSV has no
        header, ``time_column`` or ``value_column`` is missing, a row's
        values aren't numeric, the file has fewer than 2 data rows, or
        timestamps are not strictly increasing.
    ValueError
        If ``value_units == "volts"`` and ``sensitivity_mv_per_g`` is not
        positive.
    """
    if value_units == "volts" and sensitivity_mv_per_g <= 0:
        raise ValueError(f"sensitivity_mv_per_g must be positive, got {sensitivity_mv_per_g}.")

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AccelerometerError(f"{path} has no header row.")
            if time_column not in reader.fieldnames:
                raise AccelerometerError(
                    f"{path} has no column named {time_column!r}. "
                    f"Available columns: {list(reader.fieldnames)}."
                )
            if value_column not in reader.fieldnames:
                raise AccelerometerError(
                    f"{path} has no column named {value_column!r}. "
                    f"Available columns: {list(reader.fieldnames)}."
                )

            times: list[float] = []
            values: list[float] = []
            for row_index, row in enumerate(reader):
                try:
                    times.append(float(row[time_column]))
                    values.append(float(row[value_column]))
                except (TypeError, ValueError) as exc:
                    raise AccelerometerError(
                        f"{path}, row {row_index + 2}: could not parse "
                        f"{time_column!r}={row[time_column]!r} / "
                        f"{value_column!r}={row[value_column]!r} as numbers."
                    ) from exc
    except FileNotFoundError as exc:
        raise AccelerometerError(f"Accelerometer file not found: {path}") from exc
    except OSError as exc:
        raise AccelerometerError(f"Could not read {path}: {exc}") from exc

    if len(times) < 2:
        raise AccelerometerError(f"{path} has {len(times)} data row(s); at least 2 are needed.")

    times_s = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times_s) <= 0):
        raise AccelerometerError(
            f"{path}: timestamps in column {time_column!r} are not strictly increasing."
        )

    raw = np.asarray(values, dtype=np.float64)
    acceleration_g = raw if value_units == "g" else raw * 1000.0 / sensitivity_mv_per_g

    return AccelerometerRecording(times_s=times_s, acceleration_g=acceleration_g)


def compute_vibration_frequency(recording: AccelerometerRecording) -> float:
    """Find the dominant vibration frequency in an accelerometer recording.

    Computes the magnitude spectrum of the (mean-removed) acceleration
    signal via a real FFT, assuming uniform sampling at the recording's
    nominal sample rate (``(n - 1) / (times_s[-1] - times_s[0])``), and
    returns the frequency of the largest non-DC peak.

    Parameters
    ----------
    recording : AccelerometerRecording

    Returns
    -------
    float
        Dominant frequency, in Hz.

    Raises
    ------
    AccelerometerError
        If ``recording`` has fewer than 4 samples (too few for a
        meaningful frequency estimate).
    """
    n = recording.times_s.size
    if n < 4:
        raise AccelerometerError(
            f"Cannot estimate vibration frequency from {n} sample(s); at least 4 are needed."
        )

    duration_s = float(recording.times_s[-1] - recording.times_s[0])
    sample_rate_hz = (n - 1) / duration_s

    signal = recording.acceleration_g - np.mean(recording.acceleration_g)
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    magnitude = np.abs(spectrum)
    magnitude[0] = 0.0  # exclude DC
    peak_index = int(np.argmax(magnitude))

    return float(frequencies[peak_index])


def compute_gamma(recording: AccelerometerRecording) -> float:
    """Compute Gamma, the dimensionless vibration intensity.

    ``Gamma = peak acceleration / g`` is the standard control parameter
    for a sinusoidally vibrated granular bed. Since the accelerometer
    already measures acceleration directly (rather than displacement),
    this is simply the peak measured acceleration expressed in units of
    g -- the ``g`` in the numerator and denominator of the textbook
    definition (``Gamma = A * omega^2 / g``) cancel.

    Parameters
    ----------
    recording : AccelerometerRecording

    Returns
    -------
    float
        Gamma (dimensionless).
    """
    return float(np.max(np.abs(recording.acceleration_g)))


def compute_vibration_amplitude(recording: AccelerometerRecording, frequency_hz: float) -> float:
    """Compute displacement amplitude from peak acceleration and frequency.

    Assumes sinusoidal motion ``x(t) = A sin(omega t)``, for which the
    acceleration amplitude is ``A * omega^2`` -- so
    ``A = peak_acceleration / omega^2``.

    Parameters
    ----------
    recording : AccelerometerRecording
    frequency_hz : float
        Vibration frequency, in Hz (see :func:`compute_vibration_frequency`).

    Returns
    -------
    float
        Displacement amplitude, in meters.

    Raises
    ------
    ValueError
        If ``frequency_hz`` is not positive.
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")

    peak_acceleration_m_s2 = compute_gamma(recording) * GRAVITY_M_S2
    omega = 2 * math.pi * frequency_hz
    return peak_acceleration_m_s2 / omega**2


def plot_vibration_signal(recording: AccelerometerRecording, output_path: Path) -> Path:
    """Plot an accelerometer recording's time-domain signal, saved as a PNG.

    Uses matplotlib's non-interactive ``"Agg"`` backend (see
    :mod:`glas.analysis.brazil_nut` for why this is always headless-safe),
    so this never requires or attempts to open a display.

    Parameters
    ----------
    recording : AccelerometerRecording
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(recording.times_s, recording.acceleration_g, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Accelerometer signal")
    ax.axhline(0, color="gray", linewidth=0.8)
    style_axes(ax)

    fig.tight_layout()
    return savefig_publication(fig, output_path)


def analyze_vibration(
    path: Path,
    *,
    time_column: str = "time_s",
    value_column: str = "voltage_v",
    value_units: ValueUnits = "volts",
    sensitivity_mv_per_g: float = DEFAULT_SENSITIVITY_MV_PER_G,
    plot_path: Path | None = None,
) -> VibrationMetrics:
    """Import an accelerometer CSV and compute its frequency, amplitude, and Gamma.

    Convenience wrapper combining :func:`import_accelerometer_csv` with
    :func:`compute_vibration_frequency`, :func:`compute_vibration_amplitude`,
    and :func:`compute_gamma` (and, optionally, :func:`plot_vibration_signal`)
    into one call over a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        See :func:`import_accelerometer_csv`.
    time_column, value_column, value_units, sensitivity_mv_per_g : see
        :func:`import_accelerometer_csv`.
    plot_path : pathlib.Path, optional
        If given, also save a time-domain signal plot here (see
        :func:`plot_vibration_signal`).

    Returns
    -------
    VibrationMetrics
    """
    recording = import_accelerometer_csv(
        path,
        time_column=time_column,
        value_column=value_column,
        value_units=value_units,
        sensitivity_mv_per_g=sensitivity_mv_per_g,
    )
    frequency_hz = compute_vibration_frequency(recording)
    amplitude_m = compute_vibration_amplitude(recording, frequency_hz)
    gamma = compute_gamma(recording)

    if plot_path is not None:
        plot_vibration_signal(recording, plot_path)

    return VibrationMetrics(
        frequency_hz=frequency_hz,
        amplitude_m=amplitude_m,
        gamma=gamma,
        peak_acceleration_g=gamma,
    )


def synchronize_with_frames(
    recording: AccelerometerRecording,
    frames: Sequence[Frame],
    *,
    offset_s: float = 0.0,
) -> NDArray[np.float64]:
    """Find the nearest accelerometer sample in time for each frame.

    Both recordings' timelines are shifted to start at zero (the first
    sample / first frame), then aligned assuming they started at the same
    real-world moment -- true if both were started manually at roughly
    the same time, or exactly true with hardware triggering (a Phase 17
    concern). Pass ``offset_s`` to correct for a known difference: a
    positive value means the accelerometer recording started earlier than
    the camera recording (so an accelerometer sample that far into its
    own recording lines up with frame ``0``).

    Parameters
    ----------
    recording : AccelerometerRecording
    frames : sequence of Frame
        Frames to synchronize with, e.g. from
        :func:`glas.dataset.iter_frames`. Uses each frame's
        ``host_timestamp_ns``.
    offset_s : float, default 0.0
        Time offset, in seconds, added to the accelerometer recording's
        relative timeline before matching (see above).

    Returns
    -------
    numpy.ndarray
        One acceleration value (in g), per frame, from the nearest
        accelerometer sample in time -- same length and order as
        ``frames``.

    Raises
    ------
    ValueError
        If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames must not be empty.")

    frame_start_ns = frames[0].host_timestamp_ns
    frame_relative_s = np.asarray(
        [(frame.host_timestamp_ns - frame_start_ns) / 1e9 for frame in frames], dtype=np.float64
    )
    accel_relative_s = recording.times_s - recording.times_s[0] + offset_s

    insert_at = np.searchsorted(accel_relative_s, frame_relative_s)
    right = np.clip(insert_at, 0, accel_relative_s.size - 1)
    left = np.clip(insert_at - 1, 0, accel_relative_s.size - 1)

    use_left = np.abs(accel_relative_s[left] - frame_relative_s) <= np.abs(
        accel_relative_s[right] - frame_relative_s
    )
    nearest_indices = np.where(use_left, left, right)

    return recording.acceleration_g[nearest_indices]

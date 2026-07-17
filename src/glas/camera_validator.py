"""Pure validation logic for Basler camera control parameters.

This module has no dependency on pypylon: it validates plain Python values
against explicit numeric ranges or allowed sets supplied by the caller.
:mod:`glas.camera` is responsible for querying those ranges from the
connected device and calling into this module before writing any value to
hardware.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from glas.exceptions import CameraConfigurationError


class NumericRange(BaseModel):
    """An inclusive ``[minimum, maximum]`` bound for a numeric parameter."""

    model_config = ConfigDict(frozen=True)

    minimum: float
    maximum: float


class ROI(BaseModel):
    """A region of interest: sensor crop size and offset, in pixels."""

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    offset_x: int
    offset_y: int


class ROIBounds(BaseModel):
    """Valid ranges and increments for each :class:`ROI` field.

    Attributes
    ----------
    width, height, offset_x, offset_y : NumericRange
        Independent per-field bounds. ``width``/``height`` should be
        queried from the device with the offsets reset to zero, so they
        reflect the sensor's true maximum rather than a range already
        constrained by a nonzero offset.
    sensor_width, sensor_height : int
        Physical sensor dimensions in pixels; ``offset + size`` may not
        exceed these.
    width_step, height_step, offset_x_step, offset_y_step : int
        Minimum increment each field must be a multiple of.
    """

    model_config = ConfigDict(frozen=True)

    width: NumericRange
    height: NumericRange
    offset_x: NumericRange
    offset_y: NumericRange
    sensor_width: int
    sensor_height: int
    width_step: int = 1
    height_step: int = 1
    offset_x_step: int = 1
    offset_y_step: int = 1


def validate_exposure_time(value: float, bounds: NumericRange) -> float:
    """Validate a proposed exposure time.

    Parameters
    ----------
    value : float
        Proposed exposure time, in microseconds.
    bounds : NumericRange
        Minimum and maximum exposure time supported by the device.

    Returns
    -------
    float
        ``value``, unchanged, if valid.

    Raises
    ------
    CameraConfigurationError
        If ``value`` is outside ``bounds``.
    """
    if not bounds.minimum <= value <= bounds.maximum:
        raise CameraConfigurationError(
            f"Exposure time {value} us is out of range [{bounds.minimum}, {bounds.maximum}] us."
        )
    return value


def validate_gain(value: float, bounds: NumericRange) -> float:
    """Validate a proposed gain value.

    Parameters
    ----------
    value : float
        Proposed gain, in dB.
    bounds : NumericRange
        Minimum and maximum gain supported by the device.

    Returns
    -------
    float
        ``value``, unchanged, if valid.

    Raises
    ------
    CameraConfigurationError
        If ``value`` is outside ``bounds``.
    """
    if not bounds.minimum <= value <= bounds.maximum:
        raise CameraConfigurationError(
            f"Gain {value} dB is out of range [{bounds.minimum}, {bounds.maximum}] dB."
        )
    return value


def validate_numeric_range(
    value: float, bounds: NumericRange, *, field_name: str, unit: str = ""
) -> float:
    """Validate a proposed value against a device-reported numeric range.

    A generic counterpart to :func:`validate_exposure_time`/
    :func:`validate_gain` for camera parameters that don't warrant their
    own named validator (gamma, frame rate, binning, ...).

    Parameters
    ----------
    value : float
        Proposed value.
    bounds : NumericRange
        Minimum and maximum supported by the device.
    field_name : str
        Human-readable name used in the error message, e.g. ``"gamma"``.
    unit : str, optional
        Unit suffix used in the error message, e.g. ``"Hz"``.

    Returns
    -------
    float
        ``value``, unchanged, if valid.

    Raises
    ------
    CameraConfigurationError
        If ``value`` is outside ``bounds``.
    """
    if not bounds.minimum <= value <= bounds.maximum:
        suffix = f" {unit}" if unit else ""
        raise CameraConfigurationError(
            f"{field_name} {value}{suffix} is out of range "
            f"[{bounds.minimum}, {bounds.maximum}]{suffix}."
        )
    return value


def validate_pixel_format(value: str, allowed: Sequence[str]) -> str:
    """Validate a proposed pixel format against the device's supported set.

    Parameters
    ----------
    value : str
        Proposed pixel format name, e.g. ``"Mono8"``.
    allowed : Sequence[str]
        Pixel format names the device supports.

    Returns
    -------
    str
        ``value``, unchanged, if valid.

    Raises
    ------
    CameraConfigurationError
        If ``value`` is not in ``allowed``.
    """
    if value not in allowed:
        raise CameraConfigurationError(
            f"Pixel format {value!r} is not supported by this camera. "
            f"Supported formats: {', '.join(allowed)}."
        )
    return value


def validate_roi(roi: ROI, bounds: ROIBounds) -> ROI:
    """Validate a proposed region of interest against device bounds.

    Checks each field's range and step alignment, and that the crop stays
    within the physical sensor. All violations are collected and reported
    together rather than failing on the first one found.

    Parameters
    ----------
    roi : ROI
        Proposed region of interest.
    bounds : ROIBounds
        Bounds and increments reported by the device.

    Returns
    -------
    ROI
        ``roi``, unchanged, if valid.

    Raises
    ------
    CameraConfigurationError
        If any field is out of range, misaligned, or the crop exceeds the
        sensor. :attr:`~glas.exceptions.CameraConfigurationError.errors`
        lists every violation found.
    """
    errors: list[str] = []

    def _check_field(name: str, value: int, field_range: NumericRange, step: int) -> None:
        if not field_range.minimum <= value <= field_range.maximum:
            errors.append(
                f"{name}={value} is out of range [{field_range.minimum}, {field_range.maximum}]."
            )
        elif (value - int(field_range.minimum)) % step != 0:
            errors.append(f"{name}={value} is not a multiple of the required step {step}.")

    _check_field("width", roi.width, bounds.width, bounds.width_step)
    _check_field("height", roi.height, bounds.height, bounds.height_step)
    _check_field("offset_x", roi.offset_x, bounds.offset_x, bounds.offset_x_step)
    _check_field("offset_y", roi.offset_y, bounds.offset_y, bounds.offset_y_step)

    if roi.offset_x + roi.width > bounds.sensor_width:
        errors.append(
            f"offset_x + width = {roi.offset_x + roi.width} exceeds sensor width "
            f"{bounds.sensor_width}."
        )
    if roi.offset_y + roi.height > bounds.sensor_height:
        errors.append(
            f"offset_y + height = {roi.offset_y + roi.height} exceeds sensor height "
            f"{bounds.sensor_height}."
        )

    if errors:
        raise CameraConfigurationError(
            f"Invalid region of interest ({len(errors)} problem(s)).", errors=errors
        )
    return roi

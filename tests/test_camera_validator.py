"""Tests for glas.camera_validator.

These are pure logic tests: no pypylon or camera hardware is involved.
"""

from __future__ import annotations

import pytest

from glas.camera_validator import (
    ROI,
    NumericRange,
    ROIBounds,
    validate_exposure_time,
    validate_gain,
    validate_pixel_format,
    validate_roi,
)
from glas.exceptions import CameraConfigurationError


def test_validate_exposure_time_accepts_in_range_value() -> None:
    bounds = NumericRange(minimum=10.0, maximum=1000.0)
    assert validate_exposure_time(500.0, bounds) == 500.0


def test_validate_exposure_time_accepts_boundary_values() -> None:
    bounds = NumericRange(minimum=10.0, maximum=1000.0)
    assert validate_exposure_time(10.0, bounds) == 10.0
    assert validate_exposure_time(1000.0, bounds) == 1000.0


def test_validate_exposure_time_rejects_below_minimum() -> None:
    bounds = NumericRange(minimum=10.0, maximum=1000.0)
    with pytest.raises(CameraConfigurationError):
        validate_exposure_time(5.0, bounds)


def test_validate_exposure_time_rejects_above_maximum() -> None:
    bounds = NumericRange(minimum=10.0, maximum=1000.0)
    with pytest.raises(CameraConfigurationError):
        validate_exposure_time(2000.0, bounds)


def test_validate_gain_accepts_in_range_value() -> None:
    bounds = NumericRange(minimum=0.0, maximum=24.0)
    assert validate_gain(6.0, bounds) == 6.0


def test_validate_gain_rejects_out_of_range() -> None:
    bounds = NumericRange(minimum=0.0, maximum=24.0)
    with pytest.raises(CameraConfigurationError):
        validate_gain(-1.0, bounds)
    with pytest.raises(CameraConfigurationError):
        validate_gain(25.0, bounds)


def test_validate_pixel_format_accepts_supported_value() -> None:
    assert validate_pixel_format("Mono8", ["Mono8", "Mono12"]) == "Mono8"


def test_validate_pixel_format_rejects_unsupported_value() -> None:
    with pytest.raises(CameraConfigurationError):
        validate_pixel_format("RGB8", ["Mono8", "Mono12"])


def _bounds(sensor_width: int = 640, sensor_height: int = 480) -> ROIBounds:
    return ROIBounds(
        width=NumericRange(minimum=1, maximum=sensor_width),
        height=NumericRange(minimum=1, maximum=sensor_height),
        offset_x=NumericRange(minimum=0, maximum=sensor_width - 1),
        offset_y=NumericRange(minimum=0, maximum=sensor_height - 1),
        sensor_width=sensor_width,
        sensor_height=sensor_height,
    )


def test_validate_roi_accepts_full_frame() -> None:
    roi = ROI(width=640, height=480, offset_x=0, offset_y=0)
    assert validate_roi(roi, _bounds()) == roi


def test_validate_roi_accepts_centered_crop() -> None:
    roi = ROI(width=320, height=240, offset_x=160, offset_y=120)
    assert validate_roi(roi, _bounds()) == roi


def test_validate_roi_rejects_width_exceeding_sensor() -> None:
    roi = ROI(width=800, height=480, offset_x=0, offset_y=0)
    with pytest.raises(CameraConfigurationError) as exc_info:
        validate_roi(roi, _bounds())
    assert any("width" in message for message in exc_info.value.errors)


def test_validate_roi_rejects_offset_pushing_crop_past_sensor() -> None:
    roi = ROI(width=320, height=240, offset_x=500, offset_y=0)
    with pytest.raises(CameraConfigurationError) as exc_info:
        validate_roi(roi, _bounds())
    assert any("offset_x + width" in message for message in exc_info.value.errors)


def test_validate_roi_rejects_step_misalignment() -> None:
    bounds = ROIBounds(
        width=NumericRange(minimum=1, maximum=640),
        height=NumericRange(minimum=1, maximum=480),
        offset_x=NumericRange(minimum=0, maximum=639),
        offset_y=NumericRange(minimum=0, maximum=479),
        sensor_width=640,
        sensor_height=480,
        width_step=4,
    )
    # With minimum=1 and step=4, valid widths are 1, 5, 9, ..., 321, 325;
    # 322 falls between two valid steps.
    roi = ROI(width=322, height=480, offset_x=0, offset_y=0)
    with pytest.raises(CameraConfigurationError) as exc_info:
        validate_roi(roi, bounds)
    assert any("multiple of the required step" in message for message in exc_info.value.errors)


def test_validate_roi_collects_multiple_violations() -> None:
    roi = ROI(width=800, height=600, offset_x=0, offset_y=0)
    with pytest.raises(CameraConfigurationError) as exc_info:
        validate_roi(roi, _bounds())
    assert len(exc_info.value.errors) >= 2

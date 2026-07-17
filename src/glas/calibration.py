"""Spatial calibration: converting pixel measurements into physical units.

Every earlier analysis phase (tracking, Brazil nut, packing, segregation,
AI segmentation) reports sizes, positions, and velocities in pixels --
correct for comparing frames within one recording, but not directly
publishable, since a paper needs millimeters, not "42.3 px". A
:class:`SpatialCalibration` bridges the gap: measure a known real-world
distance once per camera/lens/working-distance setup, and every later
pixel measurement converts to millimeters with one multiplication.

    known real-world distance -> calibrate_from_known_distance()/calibrate_from_checkerboard()
        -> SpatialCalibration -> save_calibration()/load_calibration()

Two calibration methods are provided:

- :func:`calibrate_from_known_distance` -- the simplest possible method:
  click (or otherwise measure) two pixel points a known real distance
  apart (e.g. a ruler laid in the field of view). No special equipment.
- :func:`calibrate_from_checkerboard` -- the standard machine-vision
  method: a checkerboard pattern of known square size gives many
  independent spacing measurements averaged into one more precise
  result, at the cost of needing a printed checkerboard.

Nothing elsewhere in GLAS requires a calibration to exist -- every
analysis function continues to work in pixels with no changes. A
calibration is an optional, explicit conversion step applied by the
caller (CLI, GUI, or a report) wherever physical units are wanted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from glas.exceptions import CalibrationError, JSONValidationError


class SpatialCalibration(BaseModel):
    """A pixel-to-millimeter conversion factor for one camera/lens/working-distance setup.

    Attributes
    ----------
    mm_per_pixel : float
        Physical size of one pixel, in millimeters. Must be positive.
    method : {"two_point", "checkerboard"}
        How this calibration was computed.
    created_at_utc : str
        ISO 8601 UTC timestamp the calibration was computed.
    notes : str
        Free-text notes (e.g. which lens/working distance this applies
        to -- a calibration is only valid for the exact optical setup it
        was measured under).
    """

    model_config = ConfigDict(frozen=True)

    mm_per_pixel: float = Field(gt=0)
    method: Literal["two_point", "checkerboard"]
    created_at_utc: str = Field(min_length=1)
    notes: str = ""

    def px_to_mm(self, pixels: float) -> float:
        """Convert a pixel length (or distance) to millimeters."""
        return pixels * self.mm_per_pixel

    def mm_to_px(self, millimeters: float) -> float:
        """Convert a millimeter length (or distance) to pixels."""
        return millimeters / self.mm_per_pixel

    def px_to_mm_area(self, pixels_squared: float) -> float:
        """Convert a pixel area to square millimeters."""
        return pixels_squared * self.mm_per_pixel**2


def calibrate_from_known_distance(
    point_a: tuple[float, float], point_b: tuple[float, float], distance_mm: float
) -> SpatialCalibration:
    """Compute a calibration from two pixel points a known real-world distance apart.

    The simplest calibration method: lay a ruler (or any object of known
    length) in the field of view, identify two points on it in a captured
    frame, and give their pixel coordinates and the real distance between
    them.

    Parameters
    ----------
    point_a, point_b : tuple of (float, float)
        ``(x, y)`` pixel coordinates of the two reference points.
    distance_mm : float
        Real-world distance between ``point_a`` and ``point_b``, in
        millimeters. Must be positive.

    Returns
    -------
    SpatialCalibration

    Raises
    ------
    CalibrationError
        If ``distance_mm`` is not positive, or ``point_a``/``point_b``
        coincide (zero pixel distance -- can't compute a scale from it).
    """
    if distance_mm <= 0:
        raise CalibrationError(f"distance_mm must be positive, got {distance_mm}.")
    pixel_distance = float(np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))
    if pixel_distance == 0:
        raise CalibrationError("point_a and point_b coincide -- cannot compute a scale from them.")
    return SpatialCalibration(
        mm_per_pixel=distance_mm / pixel_distance,
        method="two_point",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def calibrate_from_checkerboard(
    image: NDArray[np.integer], pattern_size: tuple[int, int], square_size_mm: float
) -> SpatialCalibration:
    """Compute a calibration from a checkerboard pattern of known square size.

    Locates every internal checkerboard corner (via ``cv2.findChessboardCorners``,
    refined with ``cv2.cornerSubPix``), measures the pixel spacing between
    every pair of horizontally/vertically adjacent corners, and averages
    them -- far more precise than a single two-point measurement, since it
    combines many independent spacing measurements across the whole
    pattern.

    Parameters
    ----------
    image : numpy.ndarray
        Mono or color image containing the full checkerboard pattern,
        unobstructed.
    pattern_size : tuple of (int, int)
        Number of *internal* corners ``(columns, rows)`` -- for a
        checkerboard with an 8x6 grid of squares, this is ``(7, 5)``.
    square_size_mm : float
        Physical size of one checkerboard square's side, in millimeters.
        Must be positive.

    Returns
    -------
    SpatialCalibration

    Raises
    ------
    CalibrationError
        If ``square_size_mm`` is not positive, or the checkerboard
        pattern cannot be found in ``image``.
    """
    if square_size_mm <= 0:
        raise CalibrationError(f"square_size_mm must be positive, got {square_size_mm}.")

    mono = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mono_u8 = mono if mono.dtype == np.uint8 else cv2.convertScaleAbs(mono)

    found, corners = cv2.findChessboardCorners(mono_u8, pattern_size)
    if not found:
        raise CalibrationError(
            f"Could not find a {pattern_size[0]}x{pattern_size[1]} checkerboard pattern in the "
            "given image."
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(mono_u8, corners, (11, 11), (-1, -1), criteria)

    columns, rows = pattern_size
    grid = corners.reshape(rows, columns, 2)

    horizontal_spacings = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical_spacings = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    all_spacings = np.concatenate([horizontal_spacings.ravel(), vertical_spacings.ravel()])
    mean_spacing_px = float(np.mean(all_spacings))

    if mean_spacing_px == 0:
        raise CalibrationError("Detected checkerboard corners are degenerate (zero spacing).")

    return SpatialCalibration(
        mm_per_pixel=square_size_mm / mean_spacing_px,
        method="checkerboard",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_calibration(calibration: SpatialCalibration, path: Path) -> None:
    """Write ``calibration`` to ``path`` as pretty-printed JSON.

    Parameters
    ----------
    calibration : SpatialCalibration
    path : pathlib.Path
        Destination file. Parent directories are created if missing.

    Raises
    ------
    CalibrationError
        If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(calibration.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise CalibrationError(f"Could not write calibration file {path}: {exc}") from exc


def load_calibration(path: Path) -> SpatialCalibration:
    """Read and validate a :class:`SpatialCalibration` from a JSON file.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    SpatialCalibration

    Raises
    ------
    CalibrationError
        If the file does not exist, cannot be read or parsed as JSON, or
        does not match the expected structure.
    """
    if not path.is_file():
        raise CalibrationError(f"Calibration file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Could not read calibration file {path}: {exc}") from exc
    try:
        return SpatialCalibration.model_validate(data)
    except ValidationError as exc:
        raise CalibrationError(
            str(JSONValidationError.from_pydantic(exc, context="Calibration file"))
        ) from exc

"""Pure image-processing utilities for particle detection and frame-to-frame linking.

:mod:`glas.analysis.particle_tracking` builds trajectories out of what
this module provides: per-frame particle detections
(:func:`detect_particles`) and a way to link one frame's detections onto
the previous frame's (:func:`link_nearest`). Everything here is a pure
function over numpy arrays / plain data -- no state, no I/O -- so it's
fully unit-testable without a camera, a dataset, or a display, the same
design already used for :mod:`glas.preview` and :mod:`glas.display`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MIN_AREA = 4.0


class Detection(BaseModel):
    """A single particle detected in one frame, in image pixel coordinates.

    Attributes
    ----------
    x, y : float
        Centroid position, in pixels.
    radius : float
        Equivalent radius -- the radius of a circle with the same area as
        the detected blob (``sqrt(area / pi)``), a standard particle-sizing
        convention that stays meaningful for imperfectly circular blobs.
    area : float
        Blob area, in pixels, as reported by ``cv2.contourArea``.
    """

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    radius: float = Field(ge=0)
    area: float = Field(ge=0)


def to_uint8_mono(image: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Scale an arbitrary-dtype mono image to 8-bit, for cv2's 8-bit-only Otsu threshold.

    Shared with :mod:`glas.ai.sam2_train`, which needs the same
    thresholding preprocessing to bootstrap SAM2 mask ground truth.
    """
    if image.dtype == np.uint8:
        return cast("NDArray[np.uint8]", image)
    max_value = np.iinfo(image.dtype).max
    return (image.astype(np.float64) * (255.0 / max_value)).astype(np.uint8)


def detect_particles(
    image: NDArray[np.integer],
    *,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
) -> list[Detection]:
    """Detect particle-like blobs in a single mono image.

    Thresholds the image to a binary mask, then finds each connected
    blob's centroid and equivalent radius via contour moments.

    Parameters
    ----------
    image : numpy.ndarray
        Mono image, shape ``(height, width)``. Non-``uint8`` dtypes are
        scaled to 8-bit first (Otsu thresholding is only defined for
        8-bit images in OpenCV).
    min_area : float, default 4.0
        Minimum blob area, in pixels, to be reported as a detection.
        Filters out single-pixel noise.
    max_area : float, optional
        Maximum blob area, in pixels. ``None`` means no upper bound --
        useful for filtering out large lit regions that aren't particles
        (e.g. reflections, or several merged/overlapping particles).
    threshold : int, optional
        Explicit threshold, in the 0-255 scale of the (possibly
        rescaled) 8-bit image. ``None`` (the default) computes it
        automatically via Otsu's method.
    invert : bool, default False
        ``False`` treats pixels *above* the threshold as particles
        (bright particles on a dark background). ``True`` treats pixels
        *below* the threshold as particles (dark particles on a bright
        background).

    Returns
    -------
    list of Detection
        One entry per detected blob, in contour-scan order (top-to-bottom,
        left-to-right). Empty if nothing passes the area filter.
    """
    mono = to_uint8_mono(image)

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if threshold is None:
        _, binary = cv2.threshold(mono, 0, 255, threshold_type + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(mono, threshold, 255, threshold_type)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue  # degenerate contour (e.g. a line), no well-defined centroid

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        radius = math.sqrt(area / math.pi)
        detections.append(Detection(x=x, y=y, radius=radius, area=area))

    return detections


def link_nearest(
    previous: Sequence[Detection], current: Sequence[Detection], max_distance: float
) -> list[tuple[int, int]]:
    """Greedily match ``current`` detections to the nearest ``previous`` one.

    Parameters
    ----------
    previous, current : sequence of Detection
        Detections from two consecutive frames.
    max_distance : float
        Maximum pixel distance between a ``previous`` and a ``current``
        detection for them to be considered a possible match.

    Returns
    -------
    list of (int, int)
        ``(previous_index, current_index)`` pairs, each index appearing
        at most once, sorted by ``previous_index``.

    Notes
    -----
    This is a greedy nearest-neighbor match, not a globally optimal
    assignment (unlike the Hungarian algorithm): every candidate pair
    within ``max_distance`` is considered in increasing distance order,
    and the first available match commits both indices. This can
    occasionally produce a suboptimal linking when two particles' plausible
    matches cross paths between frames, but it needs no extra dependency
    (no ``scipy``) and is accurate as long as particles move less than
    roughly half their typical inter-particle spacing between frames --
    true for most granular-material video at typical frame rates. Revisit
    with ``scipy.optimize.linear_sum_assignment`` if dense, fast-crossing
    trajectories turn out to matter in practice.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, prev in enumerate(previous):
        for j, curr in enumerate(current):
            distance = math.hypot(curr.x - prev.x, curr.y - prev.y)
            if distance <= max_distance:
                candidates.append((distance, i, j))
    candidates.sort(key=lambda item: item[0])

    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, i, j in candidates:
        if i in matched_previous or j in matched_current:
            continue
        matched_previous.add(i)
        matched_current.add(j)
        pairs.append((i, j))

    pairs.sort()
    return pairs

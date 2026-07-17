"""SAM2-based particle segmentation: exact outlines instead of a bounding box.

    Camera frame + box prompt -> Sam2Segmenter.segment() -> ParticleSegment -> ShapeMetrics

Where :mod:`glas.ai.yolo_detector` (or classical blob detection) answers
"where is each particle, roughly," SAM2 refines that into "what is this
particle's exact pixel outline" -- from there GLAS computes area,
perimeter, orientation, aspect ratio, contact area between touching
grains, packing fraction, and void fraction, all directly from the mask
rather than a circular-particle approximation.

:class:`Sam2Segmenter` most naturally prompts from
:class:`~glas.ai.yolo_detector.YoloDetection`/
:class:`~glas.analysis.tracking_utils.Detection` boxes -- run YOLO (or
classical detection) first, then hand each box to
:meth:`Sam2Segmenter.segment_frame` for pixel-accurate refinement.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from glas.ai.dependencies import import_build_sam2, import_sam2_image_predictor, import_torch
from glas.analysis.tracking_utils import Detection
from glas.exceptions import AIModelError
from glas.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BOX_MARGIN_PX = 4.0
DEFAULT_CONTACT_DILATION_PX = 1


class ShapeMetrics(BaseModel):
    """Per-particle shape measurements computed from an exact segmentation mask.

    Attributes
    ----------
    area_px : float
        Exact mask area, in pixels (a pixel count, not a circular
        approximation).
    perimeter_px : float
        Contour perimeter, in pixels.
    centroid_x, centroid_y : float
        Mask centroid, in pixels.
    orientation_deg : float
        Angle of the fitted ellipse's major axis, in degrees
        (``cv2.fitEllipse`` convention: ``0``-``180``).
    aspect_ratio : float
        Fitted ellipse's major axis length divided by its minor axis
        length; ``1.0`` for a perfect circle, larger for elongated
        particles.
    """

    model_config = ConfigDict(frozen=True)

    area_px: float = Field(ge=0)
    perimeter_px: float = Field(ge=0)
    centroid_x: float
    centroid_y: float
    orientation_deg: float
    aspect_ratio: float = Field(ge=1)


class ParticleSegment:
    """One particle's segmentation mask, prediction quality, and shape metrics.

    Like :class:`glas.frame.Frame`, this is a deliberate exception to
    GLAS's usual Pydantic-model convention: :attr:`mask` is a full-frame
    boolean array (up to a full sensor frame in size, one per detected
    particle per frame), so validating it on every construction would add
    real overhead for a field that's already correct by construction --
    SAM2's own inference output -- and Pydantic gets no meaningful
    shape/dtype validation out of a raw boolean array anyway. Equality is
    identity-based for the same reason ``Frame`` documents:
    array-``==``-array is elementwise, not a single bool.

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean array, shape ``(height, width)`` -- ``True`` where the
        particle is present, in the source frame's coordinate space.
    score : float
        SAM2's own predicted IoU/quality score for this mask, in
        ``[0, 1]``.
    metrics : ShapeMetrics
        Shape measurements derived from :attr:`mask`.
    """

    __slots__ = ("mask", "score", "metrics")

    def __init__(self, mask: NDArray[np.bool_], score: float, metrics: ShapeMetrics) -> None:
        self.mask = mask
        self.score = score
        self.metrics = metrics


def compute_shape_metrics(mask: NDArray[np.bool_]) -> ShapeMetrics:
    """Compute area, perimeter, centroid, orientation, and aspect ratio from a mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean (or 0/1) array, shape ``(height, width)``.

    Returns
    -------
    ShapeMetrics

    Raises
    ------
    AIModelError
        If ``mask`` contains no foreground pixels (an empty mask has no
        well-defined contour).
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise AIModelError("Cannot compute shape metrics for an empty segmentation mask.")
    contour = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(contour, True))
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        ys, xs = np.nonzero(mask)
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())

    if len(contour) >= 5:
        (_, _), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)
        orientation_deg = float(angle)
        aspect_ratio = float(major_axis / minor_axis) if minor_axis > 0 else 1.0
    else:
        orientation_deg = 0.0
        aspect_ratio = 1.0

    return ShapeMetrics(
        area_px=float(np.count_nonzero(mask)),
        perimeter_px=perimeter,
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        orientation_deg=orientation_deg,
        aspect_ratio=max(aspect_ratio, 1.0),
    )


def compute_contact_area(
    mask_a: NDArray[np.bool_],
    mask_b: NDArray[np.bool_],
    *,
    dilation_px: int = DEFAULT_CONTACT_DILATION_PX,
) -> int:
    """Measure the shared boundary between two particles' masks, in pixels.

    Dilates each mask by ``dilation_px`` and intersects it with the
    other's original mask -- two particles that are merely adjacent (not
    overlapping, which shouldn't happen for two distinct real particles)
    register a nonzero contact area proportional to how much of their
    boundaries touch.

    Parameters
    ----------
    mask_a, mask_b : numpy.ndarray
        Boolean arrays, same shape.
    dilation_px : int, default 1
        Dilation radius, in pixels. Must be positive.

    Returns
    -------
    int
        Contact area, in pixels. ``0`` if the particles don't touch
        within ``dilation_px``.

    Raises
    ------
    ValueError
        If ``dilation_px`` is not positive, or the masks' shapes differ.
    """
    if dilation_px <= 0:
        raise ValueError(f"dilation_px must be positive, got {dilation_px}.")
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shapes differ: {mask_a.shape} vs {mask_b.shape}.")

    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), np.uint8)
    dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel)
    dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel)
    contact = (dilated_a.astype(bool) & mask_b) | (dilated_b.astype(bool) & mask_a)
    return int(np.count_nonzero(contact))


class SegmentationSummary(BaseModel):
    """Packing statistics for every particle segmented in one frame.

    Attributes
    ----------
    particle_count : int
        Number of segmented particles.
    packing_fraction : float
        Fraction of the frame's pixels covered by any particle mask, in
        ``[0, 1]`` -- the exact-mask equivalent of
        :func:`glas.analysis.packing.compute_packing_metrics`'s
        circular-particle approximation.
    void_fraction : float
        ``1 - packing_fraction``.
    contacts : list of tuple of (int, int, int)
        ``(i, j, contact_area_px)`` for every pair of particles (by index
        into the segment list passed to
        :func:`compute_segmentation_summary`) with a nonzero contact
        area.
    """

    model_config = ConfigDict(frozen=True)

    particle_count: int = Field(ge=0)
    packing_fraction: float = Field(ge=0, le=1)
    void_fraction: float = Field(ge=0, le=1)
    contacts: list[tuple[int, int, int]] = Field(default_factory=list)


def compute_segmentation_summary(
    segments: Sequence[ParticleSegment],
    frame_shape: tuple[int, int],
    *,
    contact_dilation_px: int = DEFAULT_CONTACT_DILATION_PX,
) -> SegmentationSummary:
    """Compute packing fraction, void fraction, and pairwise contacts from a frame's segments.

    Parameters
    ----------
    segments : sequence of ParticleSegment
        Every particle segmented in one frame (see
        :meth:`Sam2Segmenter.segment_frame`).
    frame_shape : tuple of (int, int)
        ``(height, width)`` of the source frame.
    contact_dilation_px : int, default 1
        See :func:`compute_contact_area`.

    Returns
    -------
    SegmentationSummary
    """
    height, width = frame_shape
    frame_area = height * width
    covered = np.zeros((height, width), dtype=bool)
    for segment in segments:
        covered |= segment.mask
    packing_fraction = float(np.count_nonzero(covered)) / frame_area if frame_area > 0 else 0.0

    contacts: list[tuple[int, int, int]] = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            contact_area = compute_contact_area(
                segments[i].mask, segments[j].mask, dilation_px=contact_dilation_px
            )
            if contact_area > 0:
                contacts.append((i, j, contact_area))

    return SegmentationSummary(
        particle_count=len(segments),
        packing_fraction=packing_fraction,
        void_fraction=1.0 - packing_fraction,
        contacts=contacts,
    )


def _box_from_detection(detection: Detection, *, margin_px: float) -> NDArray[np.float64]:
    half = detection.radius + margin_px
    return np.array(
        [detection.x - half, detection.y - half, detection.x + half, detection.y + half],
        dtype=np.float64,
    )


class Sam2Segmenter:
    """Wraps a SAM2 image predictor for per-particle mask segmentation.

    Parameters
    ----------
    model_id : str, optional
        A Hugging Face Hub SAM2 model id (e.g.
        ``"facebook/sam2.1-hiera-large"``), downloaded and loaded
        automatically. Mutually exclusive with ``config_file``/
        ``checkpoint_path``.
    config_file : str, optional
        A SAM2 model config name or path (e.g. after training a custom
        model with :mod:`glas.ai.sam2_train`). Must be given together
        with ``checkpoint_path``.
    checkpoint_path : str or pathlib.Path, optional
        Local checkpoint file (``.pt``) matching ``config_file``.
    device : str, optional
        Inference device. ``None`` uses CUDA if available, else CPU.

    Raises
    ------
    ValueError
        If neither (nor both) of ``model_id`` and
        ``config_file``/``checkpoint_path`` are given.
    AIDependencyError
        If ``torch`` or ``sam2`` is not installed.
    AIModelError
        If the model cannot be loaded.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        config_file: str | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        _predictor_cls: Any | None = None,
        _build_sam2: Any | None = None,
        _torch: Any | None = None,
    ) -> None:
        has_pretrained = model_id is not None
        has_local = config_file is not None or checkpoint_path is not None
        if has_pretrained and has_local:
            raise ValueError("Pass either model_id or config_file+checkpoint_path, not both.")
        if not has_pretrained and (config_file is None or checkpoint_path is None):
            raise ValueError("Pass either model_id, or both config_file and checkpoint_path.")

        torch = import_torch(_torch)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        predictor_cls = import_sam2_image_predictor(_predictor_cls)
        try:
            if model_id is not None:
                self._predictor = predictor_cls.from_pretrained(model_id, device=resolved_device)
            else:
                build_sam2 = import_build_sam2(_build_sam2)
                model = build_sam2(config_file, str(checkpoint_path), device=resolved_device)
                self._predictor = predictor_cls(model)
        except Exception as exc:
            raise AIModelError(f"Could not load SAM2 model: {exc}") from exc

        self._device = resolved_device

    def segment(self, image: NDArray[np.integer], box: NDArray[np.float64]) -> ParticleSegment:
        """Segment one particle from a single box prompt.

        Parameters
        ----------
        image : numpy.ndarray
            RGB or mono image, shape ``(height, width[, channels])``.
            Mono images are converted to 3-channel automatically (SAM2
            expects an RGB-shaped input).
        box : numpy.ndarray
            Length-4 array ``[x1, y1, x2, y2]``, in pixels.

        Returns
        -------
        ParticleSegment

        Raises
        ------
        AIModelError
            If segmentation fails, or produces an empty mask.
        """
        rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        try:
            self._predictor.set_image(rgb)
            masks, scores, _ = self._predictor.predict(box=box, multimask_output=False)
        except Exception as exc:
            raise AIModelError(f"SAM2 segmentation failed: {exc}") from exc

        mask = np.asarray(masks[0], dtype=bool)
        score = float(np.asarray(scores).reshape(-1)[0])
        metrics = compute_shape_metrics(mask)
        return ParticleSegment(mask=mask, score=score, metrics=metrics)

    def segment_frame(
        self,
        image: NDArray[np.integer],
        detections: Sequence[Detection],
        *,
        box_margin_px: float = DEFAULT_BOX_MARGIN_PX,
    ) -> list[ParticleSegment]:
        """Segment every particle in one frame, prompted by its detections.

        Turns each :class:`~glas.analysis.tracking_utils.Detection`
        (from classical blob detection or
        :class:`~glas.ai.yolo_detector.YoloParticleDetector`) into a box
        prompt (centered on the detection, padded by ``box_margin_px``)
        and segments it individually -- SAM2's image predictor prompts
        one box per call, so this loops rather than batching.

        Parameters
        ----------
        image : numpy.ndarray
            See :meth:`segment`.
        detections : sequence of Detection
            One box prompt per detection.
        box_margin_px : float, default 4.0
            Padding added around each detection's radius when building
            its box prompt, so the box comfortably contains the whole
            particle even if the detection's radius slightly
            underestimates it.

        Returns
        -------
        list of ParticleSegment
            One entry per detection, same order.
        """
        rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        try:
            self._predictor.set_image(rgb)
        except Exception as exc:
            raise AIModelError(f"SAM2 segmentation failed: {exc}") from exc

        segments: list[ParticleSegment] = []
        for detection in detections:
            box = _box_from_detection(detection, margin_px=box_margin_px)
            try:
                masks, scores, _ = self._predictor.predict(box=box, multimask_output=False)
            except Exception as exc:
                raise AIModelError(f"SAM2 segmentation failed: {exc}") from exc
            mask = np.asarray(masks[0], dtype=bool)
            score = float(np.asarray(scores).reshape(-1)[0])
            segments.append(
                ParticleSegment(mask=mask, score=score, metrics=compute_shape_metrics(mask))
            )

        logger.debug("Segmented %d particle(s) with SAM2.", len(segments))
        return segments

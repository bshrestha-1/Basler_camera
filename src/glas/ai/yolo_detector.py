"""YOLO-based particle detection, classification, and intruder identification.

    Camera frame -> YoloParticleDetector.detect() -> list[YoloDetection] -> ParticleTracker

:class:`YoloDetection` is a :class:`~glas.analysis.tracking_utils.Detection`
subclass -- every ``YoloDetection`` *is* a ``Detection`` -- so YOLO output
plugs directly into the existing tracking pipeline
(:class:`~glas.analysis.particle_tracking.ParticleTracker`) with no
changes to it at all: :func:`track_dataset_yolo` mirrors
:func:`glas.analysis.particle_tracking.track_dataset` exactly, just
sourcing each frame's detections from a trained YOLO model instead of
classical blob thresholding. The resulting
:class:`~glas.analysis.particle_tracking.TrackedParticle` history carries
each detection's label, confidence, and intruder flag straight through,
so every downstream pipeline that already consumes tracking history
(Brazil nut, packing, segregation) sees them too.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from glas.ai.dependencies import import_ultralytics
from glas.analysis.particle_tracking import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_GAP,
    ParticleTracker,
    TrackedParticle,
)
from glas.analysis.tracking_utils import Detection
from glas.dataset import iter_frames
from glas.exceptions import AIModelError
from glas.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_INTRUDER_LABEL = "intruder"


class YoloDetection(Detection):
    """A particle detected and classified by a trained YOLO model.

    Extends :class:`~glas.analysis.tracking_utils.Detection` with the
    label, confidence, and intruder flag classical blob detection can't
    provide.

    Attributes
    ----------
    label : str
        Predicted class name, from the trained model's own class list
        (e.g. ``"glass_bead"``, ``"steel_ball"``, ``"brazil_nut"``,
        ``"contaminant"`` -- whatever it was trained on).
    confidence : float
        Model confidence for this detection, in ``[0, 1]``.
    is_intruder : bool
        ``True`` if ``label`` matches the detector's configured
        :attr:`~YoloParticleDetector.intruder_label` (case-insensitive).
    """

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: float = Field(ge=0, le=1)
    is_intruder: bool = False


def _to_numpy(value: Any) -> NDArray[np.float64]:
    """Convert a torch.Tensor (or anything array-like) to a float64 numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


class YoloParticleDetector:
    """Wraps a trained ``ultralytics`` YOLO model for particle detection and classification.

    Parameters
    ----------
    weights : str or pathlib.Path
        Path to a trained model's weights file (``.pt``), or an
        Ultralytics-recognized pretrained model name (e.g.
        ``"yolo11n.pt"``) to download automatically.
    confidence_threshold : float, default 0.25
        Minimum detection confidence to report, in ``[0, 1]``.
    iou_threshold : float, default 0.7
        IoU threshold for the model's own non-maximum suppression, in
        ``[0, 1]``.
    intruder_label : str, default "intruder"
        Class name (case-insensitive) treated as the Brazil-nut-style
        intruder -- see :attr:`YoloDetection.is_intruder`. A model with no
        class of this name simply never flags an intruder; nothing else
        changes.
    device : str, optional
        Inference device (e.g. ``"cpu"``, ``"cuda:0"``). ``None`` lets
        ``ultralytics`` choose automatically.

    Raises
    ------
    ValueError
        If ``confidence_threshold`` or ``iou_threshold`` is outside
        ``[0, 1]``.
    AIDependencyError
        If ``ultralytics`` is not installed.
    AIModelError
        If ``weights`` cannot be loaded.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        intruder_label: str = DEFAULT_INTRUDER_LABEL,
        device: str | None = None,
        _ultralytics: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}.")

        module = import_ultralytics(_ultralytics)
        try:
            self._model = module.YOLO(str(weights))
        except Exception as exc:
            raise AIModelError(f"Could not load YOLO weights from {weights!r}: {exc}") from exc

        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self.intruder_label = intruder_label
        self._device = device

    @property
    def class_names(self) -> dict[int, str]:
        """The trained model's class index -> name mapping."""
        return dict(self._model.names)

    def detect(self, image: NDArray[np.integer]) -> list[YoloDetection]:
        """Detect and classify every particle in one frame.

        Parameters
        ----------
        image : numpy.ndarray
            Mono or color image, as accepted by ``ultralytics``'s own
            ``predict()``.

        Returns
        -------
        list of YoloDetection
            One entry per detection above ``confidence_threshold``, in
            the order ``ultralytics`` returns them.

        Raises
        ------
        AIModelError
            If inference fails.
        """
        try:
            results = self._model.predict(
                source=image,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            raise AIModelError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = dict(self._model.names)
        xyxy = _to_numpy(boxes.xyxy)
        confidences = _to_numpy(boxes.conf)
        class_indices = _to_numpy(boxes.cls)

        detections: list[YoloDetection] = []
        for (x1, y1, x2, y2), confidence, class_index in zip(
            xyxy, confidences, class_indices, strict=True
        ):
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = width * height
            label = names.get(int(class_index), str(int(class_index)))
            detections.append(
                YoloDetection(
                    x=float((x1 + x2) / 2),
                    y=float((y1 + y2) / 2),
                    radius=math.sqrt(area / math.pi) if area > 0 else 0.0,
                    area=area,
                    label=label,
                    confidence=float(confidence),
                    is_intruder=label.lower() == self.intruder_label.lower(),
                )
            )
        return detections


def track_dataset_yolo(
    folder: Path,
    weights: str | Path,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_gap: int = DEFAULT_MAX_GAP,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    intruder_label: str = DEFAULT_INTRUDER_LABEL,
    device: str | None = None,
    _detector: YoloParticleDetector | None = None,
) -> dict[int, list[TrackedParticle]]:
    """Detect, classify, and track particles across a dataset using a trained YOLO model.

    The YOLO equivalent of :func:`glas.analysis.particle_tracking.track_dataset`
    -- same tracker, same return shape, just sourced from
    :class:`YoloParticleDetector` instead of
    :func:`~glas.analysis.tracking_utils.detect_particles`. Usable
    anywhere ``track_dataset`` is used today (tracking, Brazil nut,
    packing, segregation all consume the same
    ``dict[int, list[TrackedParticle]]`` shape).

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    weights : str or pathlib.Path
        See :class:`YoloParticleDetector`.
    max_distance, max_gap : see :class:`~glas.analysis.particle_tracking.ParticleTracker`.
    confidence_threshold, iou_threshold, intruder_label, device : see
        :class:`YoloParticleDetector`.

    Returns
    -------
    dict of int to list of TrackedParticle
        Every trajectory found, keyed by ``track_id`` -- each observation
        carries the detector's ``label``/``confidence``/``is_intruder``
        alongside the usual position and size.

    Raises
    ------
    AIDependencyError
        If ``ultralytics`` is not installed.
    AIModelError
        If ``weights`` cannot be loaded or inference fails.
    DatasetError, DatasetFormatError, DatasetIOError
        Propagated from :func:`glas.dataset.iter_frames` if the source
        dataset cannot be read.
    """
    detector = _detector or YoloParticleDetector(
        weights,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        intruder_label=intruder_label,
        device=device,
    )
    tracker = ParticleTracker(max_distance=max_distance, max_gap=max_gap)
    frame_count = 0
    for frame in iter_frames(folder):
        detections = detector.detect(frame.image)
        tracker.update(frame.frame_id, detections, frame.host_timestamp_ns)
        frame_count += 1

    logger.info(
        "YOLO-tracked %d particle(s) across %d frame(s) of %s.",
        len(tracker.history),
        frame_count,
        folder,
    )
    return tracker.history

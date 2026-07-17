"""GLAS's optional AI stack: YOLO particle detection/classification and SAM2 segmentation.

    Camera frame -> YoloParticleDetector.detect() -> ParticleTracker -> trajectories
    Camera frame + box -> Sam2Segmenter.segment() -> per-particle mask + shape metrics

Nothing outside :mod:`glas.ai` imports ``torch``, ``ultralytics``, or
``sam2`` -- every one of them is an optional dependency
(``pip install glas[ai]``), lazily imported inside this package (see
:mod:`glas.ai.dependencies`) so ``import glas``, the CLI, and the GUI all
work with none of them installed. Both YOLO and SAM2 ship a full
inference *and* training pipeline: :mod:`glas.ai.annotation` prepares a
labeled dataset, :mod:`glas.ai.yolo_train`/:mod:`glas.ai.sam2_train` train
and export a custom model, and :mod:`glas.ai.yolo_detector`/
:mod:`glas.ai.sam2_segmenter` run inference with either a custom or a
pretrained model -- a researcher who only wants inference never has to
touch the training modules at all.
"""

from __future__ import annotations

from glas.ai.annotation import (
    DEFAULT_LABEL,
    DEFAULT_VAL_FRACTION,
    FrameAnnotation,
    ParticleAnnotation,
    auto_annotate_dataset,
    collect_class_names,
    prepare_yolo_dataset,
)
from glas.ai.dependencies import (
    AI_EXTRA_INSTALL_HINT,
    describe_missing_ai_packages,
    missing_ai_packages,
)
from glas.ai.sam2_segmenter import (
    ParticleSegment,
    Sam2Segmenter,
    SegmentationSummary,
    ShapeMetrics,
    compute_contact_area,
    compute_segmentation_summary,
    compute_shape_metrics,
)
from glas.ai.sam2_train import (
    Sam2Example,
    Sam2TrainingConfig,
    Sam2TrainingResult,
    auto_annotate_masks,
    prepare_sam2_dataset,
    train_sam2,
)
from glas.ai.yolo_detector import YoloDetection, YoloParticleDetector, track_dataset_yolo
from glas.ai.yolo_train import (
    YoloTrainingConfig,
    YoloTrainingResult,
    export_yolo_model,
    train_yolo,
    validate_yolo,
)

__all__ = [
    "AI_EXTRA_INSTALL_HINT",
    "missing_ai_packages",
    "describe_missing_ai_packages",
    "ParticleAnnotation",
    "FrameAnnotation",
    "DEFAULT_LABEL",
    "DEFAULT_VAL_FRACTION",
    "auto_annotate_dataset",
    "collect_class_names",
    "prepare_yolo_dataset",
    "YoloDetection",
    "YoloParticleDetector",
    "track_dataset_yolo",
    "YoloTrainingConfig",
    "YoloTrainingResult",
    "train_yolo",
    "validate_yolo",
    "export_yolo_model",
    "ShapeMetrics",
    "ParticleSegment",
    "SegmentationSummary",
    "compute_shape_metrics",
    "compute_contact_area",
    "compute_segmentation_summary",
    "Sam2Segmenter",
    "Sam2Example",
    "Sam2TrainingConfig",
    "Sam2TrainingResult",
    "auto_annotate_masks",
    "prepare_sam2_dataset",
    "train_sam2",
]

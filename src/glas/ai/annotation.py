"""Dataset preparation and the YOLO-format annotation GLAS trains on.

    glas.dataset.iter_frames() -> auto_annotate_dataset() -> prepare_yolo_dataset() -> data.yaml

The annotation format is YOLO's own: an ``images/`` directory of frame
images, a parallel ``labels/`` directory holding one ``.txt`` file per
image (one line per particle -- ``class_id x_center y_center width
height``, all normalized to ``[0, 1]`` of the image's dimensions), and a
``data.yaml`` naming every class. Every GLAS-specific piece is
:class:`ParticleAnnotation` and :class:`FrameAnnotation`, a plain
in-memory representation that converts to/from that on-disk layout.

:func:`auto_annotate_dataset` bootstraps a first-pass label set straight
from GLAS's own classical blob detector
(:func:`glas.analysis.tracking_utils.detect_particles`) so a researcher
has something to hand-correct rather than labeling a whole recording from
a blank slate -- open the written images in any YOLO-format annotation
tool (e.g. CVAT, LabelImg, Roboflow), fix the boxes and class labels, then
build a fresh list of :class:`FrameAnnotation` from the corrected files
(or edit the ``annotations`` in place) before calling
:func:`prepare_yolo_dataset`, which performs the train/val split and
writes the final on-disk dataset :mod:`glas.ai.yolo_train` points
``ultralytics`` at.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import yaml
from pydantic import BaseModel, ConfigDict, Field

from glas.analysis.tracking_utils import DEFAULT_MIN_AREA, detect_particles
from glas.dataset import iter_frames
from glas.exceptions import AIDatasetError
from glas.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LABEL = "particle"
DEFAULT_VAL_FRACTION = 0.2
DATA_YAML_FILENAME = "data.yaml"


class ParticleAnnotation(BaseModel):
    """One labeled particle's bounding box, in pixel coordinates.

    Attributes
    ----------
    label : str
        Class name, e.g. ``"particle"``, ``"glass_bead"``, ``"steel_ball"``,
        ``"brazil_nut"``, ``"contaminant"``, ``"intruder"``. Any string is
        valid here -- the class list a model actually trains on is
        whatever :func:`prepare_yolo_dataset` is given (or infers via
        :func:`collect_class_names`).
    x_center, y_center : float
        Box center, in pixels.
    width, height : float
        Box size, in pixels. Both must be positive.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    x_center: float
    y_center: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class FrameAnnotation(BaseModel):
    """Every labeled particle in one image, plus the image's location and size.

    Attributes
    ----------
    image_path : pathlib.Path
        Path to the image file on disk.
    image_width, image_height : int
        Image dimensions, in pixels -- needed to normalize
        :class:`ParticleAnnotation` boxes into YOLO's ``[0, 1]`` format.
    annotations : list of ParticleAnnotation
        Every labeled particle in this image. May be empty (a frame with
        no particles is a valid, useful training example).
    """

    model_config = ConfigDict(frozen=True)

    image_path: Path
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    annotations: list[ParticleAnnotation] = Field(default_factory=list)


def auto_annotate_dataset(
    folder: Path,
    output_dir: Path,
    *,
    label: str = DEFAULT_LABEL,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
) -> list[FrameAnnotation]:
    """Bootstrap a first-pass label set for a dataset via classical blob detection.

    Writes every frame out as a PNG under ``output_dir`` and runs
    :func:`~glas.analysis.tracking_utils.detect_particles` on each one,
    turning every detected blob into a square :class:`ParticleAnnotation`
    (side length ``2 * radius``, the blob's equivalent diameter) labeled
    ``label``. This is a starting point for hand correction, not a
    finished label set: classical detection has the same false positives/
    negatives here (poor lighting, overlapping particles, reflections) it
    has everywhere else in GLAS, and every particle gets the same
    ``label`` since blob detection can't classify particle type or
    identify an intruder -- that's exactly the gap a trained YOLO model
    closes.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    output_dir : pathlib.Path
        Directory to write one ``frame_NNNNNN.png`` per frame into.
        Created if missing.
    label : str, default "particle"
        Class name applied to every detected blob.
    min_area, max_area, threshold, invert : see
        :func:`~glas.analysis.tracking_utils.detect_particles`.

    Returns
    -------
    list of FrameAnnotation
        One entry per frame, in dataset order.

    Raises
    ------
    AIDatasetError
        If an image cannot be written to ``output_dir``.
    DatasetError, DatasetFormatError, DatasetIOError
        Propagated from :func:`glas.dataset.iter_frames` if the source
        dataset cannot be read.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[FrameAnnotation] = []
    for index, frame in enumerate(iter_frames(folder)):
        image_path = output_dir / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(image_path), frame.image):
            raise AIDatasetError(f"Failed to write {image_path}.")

        detections = detect_particles(
            frame.image, min_area=min_area, max_area=max_area, threshold=threshold, invert=invert
        )
        height, width = frame.image.shape[:2]
        boxes = [
            ParticleAnnotation(
                label=label,
                x_center=detection.x,
                y_center=detection.y,
                width=detection.radius * 2,
                height=detection.radius * 2,
            )
            for detection in detections
        ]
        annotations.append(
            FrameAnnotation(
                image_path=image_path, image_width=width, image_height=height, annotations=boxes
            )
        )

    logger.info(
        "Auto-annotated %d frame(s) from %s into %s (%d total box(es)).",
        len(annotations),
        folder,
        output_dir,
        sum(len(frame_annotation.annotations) for frame_annotation in annotations),
    )
    return annotations


def collect_class_names(annotations: Sequence[FrameAnnotation]) -> list[str]:
    """Collect every distinct class label present in ``annotations``, sorted.

    A convenient default for :func:`prepare_yolo_dataset`'s ``class_names``
    when the caller hasn't fixed an explicit, ordered class list (e.g.
    right after :func:`auto_annotate_dataset`, before hand correction adds
    new labels).

    Parameters
    ----------
    annotations : sequence of FrameAnnotation

    Returns
    -------
    list of str
        Every distinct :attr:`ParticleAnnotation.label`, sorted
        alphabetically.
    """
    labels = {box.label for frame_annotation in annotations for box in frame_annotation.annotations}
    return sorted(labels)


def _yolo_label_line(
    annotation: ParticleAnnotation, class_index: int, width: int, height: int
) -> str:
    x = annotation.x_center / width
    y = annotation.y_center / height
    w = annotation.width / width
    h = annotation.height / height
    return f"{class_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def prepare_yolo_dataset(
    annotations: Sequence[FrameAnnotation],
    class_names: Sequence[str],
    output_dir: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
) -> Path:
    """Split, lay out, and write a YOLO-format training dataset.

    Copies every frame's image into ``output_dir/images/{train,val}/`` and
    writes a matching YOLO-format label ``.txt`` into
    ``output_dir/labels/{train,val}/``, then writes ``output_dir/data.yaml``
    naming ``class_names`` in order (index 0 is ``class_names[0]``, and so
    on -- this order becomes the trained model's class indices, so it must
    stay stable across a retraining run if checkpoints are expected to
    stay comparable).

    Parameters
    ----------
    annotations : sequence of FrameAnnotation
        The full labeled set, e.g. from :func:`auto_annotate_dataset`
        (optionally hand-corrected) or built directly.
    class_names : sequence of str
        Every class the model should learn, in a fixed order. Every
        :class:`ParticleAnnotation` in ``annotations`` must use a label
        from this list -- see :func:`collect_class_names` to build it
        automatically.
    output_dir : pathlib.Path
        Destination dataset root. Created if missing; ``images/``,
        ``labels/``, and ``data.yaml`` under it are overwritten if
        present.
    val_fraction : float, default 0.2
        Fraction of frames held out for validation, in ``(0, 1)``. The
        split is deterministic given ``seed`` and always reserves at
        least one frame for validation and at least one for training.
    seed : int, default 0
        Seed for the train/val shuffle, for a reproducible split.

    Returns
    -------
    pathlib.Path
        Path to the written ``data.yaml``, ready to pass to
        :func:`glas.ai.yolo_train.train_yolo`.

    Raises
    ------
    AIDatasetError
        If ``annotations`` is empty, ``class_names`` is empty,
        ``val_fraction`` is not in ``(0, 1)``, an annotation uses a label
        not present in ``class_names``, or there are too few frames to
        reserve at least one for each of train and val.
    """
    if not annotations:
        raise AIDatasetError("Cannot prepare a YOLO dataset from an empty annotation list.")
    if not class_names:
        raise AIDatasetError("class_names must not be empty.")
    if not 0.0 < val_fraction < 1.0:
        raise AIDatasetError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    class_index = {name: index for index, name in enumerate(class_names)}
    for frame_annotation in annotations:
        for box in frame_annotation.annotations:
            if box.label not in class_index:
                raise AIDatasetError(
                    f"Annotation label {box.label!r} ({frame_annotation.image_path}) is not in "
                    f"class_names {list(class_names)}."
                )

    count = len(annotations)
    val_count = max(1, round(count * val_fraction))
    train_count = count - val_count
    if train_count < 1:
        raise AIDatasetError(
            f"Only {count} frame(s) available; not enough to reserve at least one for training "
            f"and one for validation at val_fraction={val_fraction}."
        )

    order = list(range(count))
    random.Random(seed).shuffle(order)
    val_indices = set(order[:val_count])

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    for index, frame_annotation in enumerate(annotations):
        split = "val" if index in val_indices else "train"
        destination_image = images_dir / split / frame_annotation.image_path.name
        shutil.copy2(frame_annotation.image_path, destination_image)

        lines = [
            _yolo_label_line(
                box,
                class_index[box.label],
                frame_annotation.image_width,
                frame_annotation.image_height,
            )
            for box in frame_annotation.annotations
        ]
        label_path = labels_dir / split / f"{frame_annotation.image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml_path = output_dir / DATA_YAML_FILENAME
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(class_names)),
            },
            sort_keys=False,
        )
    )

    logger.info(
        "Prepared YOLO dataset at %s: %d train, %d val, %d class(es).",
        output_dir,
        train_count,
        val_count,
        len(class_names),
    )
    return data_yaml_path

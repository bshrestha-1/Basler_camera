"""Dataset preparation and lightweight fine-tuning for SAM2 particle segmentation.

    glas.dataset.iter_frames() -> auto_annotate_masks() -> prepare_sam2_dataset()
        -> manifest.json -> train_sam2() -> checkpoint.pt -> Sam2Segmenter

SAM2's own upstream training code is a full Hydra-driven, distributed
video-segmentation trainer -- much more machinery than adapting a
pretrained checkpoint to a specific particle material, lighting setup, or
background needs. GLAS instead trains the way most published SAM2
fine-tuning recipes for box-prompted, single-image segmentation do:
freeze the image encoder (the large, general-purpose component pretrained
on SA-1B) and fine-tune only the prompt encoder and mask decoder, using
each training example's box prompt and ground-truth mask. This reaches a
useful, domain-adapted model from far less data and far less compute than
training a segmentation model from scratch, while staying squarely within
:class:`~glas.ai.sam2_segmenter.Sam2Segmenter`'s existing box-prompted
inference API -- a fine-tuned checkpoint drops straight back into it.

Ground-truth masks are bootstrapped from GLAS's own classical blob
detector, the same way :func:`glas.ai.annotation.auto_annotate_dataset`
bootstraps YOLO boxes -- a starting point to hand-correct with an
external mask-annotation tool, not a finished label set.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from glas.ai.dependencies import import_build_sam2, import_sam2_image_predictor, import_torch
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA, to_uint8_mono
from glas.dataset import iter_frames
from glas.exceptions import AIDatasetError, AIModelError
from glas.logger import get_logger

logger = get_logger(__name__)

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-5
MANIFEST_FILENAME = "manifest.json"


class Sam2Example(BaseModel):
    """One box-prompted training example: an image, a box, and its ground-truth mask.

    Attributes
    ----------
    image_path : pathlib.Path
        Path to the source image (RGB or mono, as written to disk).
    mask_path : pathlib.Path
        Path to a single-channel PNG mask, same dimensions as the image
        -- nonzero pixels are the particle.
    box : tuple of (float, float, float, float)
        ``(x1, y1, x2, y2)`` prompt box, in pixels, tight around the same
        particle ``mask_path`` labels.
    """

    model_config = ConfigDict(frozen=True)

    image_path: Path
    mask_path: Path
    box: tuple[float, float, float, float]


def auto_annotate_masks(
    folder: Path,
    output_dir: Path,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
) -> list[Sam2Example]:
    """Bootstrap SAM2 mask ground truth via classical contour detection.

    Unlike :func:`glas.ai.annotation.auto_annotate_dataset` (which only
    needs a centroid and radius for a YOLO box), SAM2 fine-tuning needs an
    actual pixel mask per particle, so this reruns thresholding and
    contour extraction itself and rasterizes each surviving contour as a
    filled binary mask, rather than reusing
    :func:`~glas.analysis.tracking_utils.detect_particles`.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    output_dir : pathlib.Path
        Directory to write ``images/`` and ``masks/`` into. Created if
        missing.
    min_area, max_area, threshold, invert : see
        :func:`~glas.analysis.tracking_utils.detect_particles`.

    Returns
    -------
    list of Sam2Example
        One entry per surviving contour, across every frame.

    Raises
    ------
    AIDatasetError
        If an image or mask cannot be written to ``output_dir``.
    DatasetError, DatasetFormatError, DatasetIOError
        Propagated from :func:`glas.dataset.iter_frames` if the source
        dataset cannot be read.
    """
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    examples: list[Sam2Example] = []
    for index, frame in enumerate(iter_frames(folder)):
        mono = to_uint8_mono(frame.image)
        if threshold is None:
            _, binary = cv2.threshold(mono, 0, 255, threshold_type + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(mono, threshold, 255, threshold_type)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_path = images_dir / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(image_path), frame.image):
            raise AIDatasetError(f"Failed to write {image_path}.")

        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
            mask_path = masks_dir / f"frame_{index:06d}_{contour_index:03d}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise AIDatasetError(f"Failed to write {mask_path}.")

            examples.append(
                Sam2Example(
                    image_path=image_path,
                    mask_path=mask_path,
                    box=(float(x), float(y), float(x + width), float(y + height)),
                )
            )

    logger.info("Auto-annotated %d SAM2 mask example(s) from %s.", len(examples), folder)
    return examples


def prepare_sam2_dataset(
    examples: Sequence[Sam2Example],
    output_dir: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
) -> Path:
    """Split a labeled example set into train/val and write a training manifest.

    Parameters
    ----------
    examples : sequence of Sam2Example
        E.g. from :func:`auto_annotate_masks` (optionally hand-corrected).
    output_dir : pathlib.Path
        Where to write ``manifest.json``, listing every example's
        (already on-disk) ``image_path``/``mask_path``/``box``, split
        into ``"train"`` and ``"val"``.
    val_fraction : float, default 0.2
        Fraction of examples held out for validation, in ``(0, 1)``.
    seed : int, default 0
        Seed for the deterministic train/val shuffle.

    Returns
    -------
    pathlib.Path
        Path to the written ``manifest.json``.

    Raises
    ------
    AIDatasetError
        If ``examples`` is empty, ``val_fraction`` is not in ``(0, 1)``,
        or there are too few examples to reserve at least one for each of
        train and val.
    """
    if not examples:
        raise AIDatasetError("Cannot prepare a SAM2 dataset from an empty example list.")
    if not 0.0 < val_fraction < 1.0:
        raise AIDatasetError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    count = len(examples)
    val_count = max(1, round(count * val_fraction))
    train_count = count - val_count
    if train_count < 1:
        raise AIDatasetError(
            f"Only {count} example(s) available; not enough to reserve at least one for "
            f"training and one for validation at val_fraction={val_fraction}."
        )

    order = list(range(count))
    random.Random(seed).shuffle(order)
    val_indices = set(order[:val_count])

    manifest: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for index, example in enumerate(examples):
        split = "val" if index in val_indices else "train"
        manifest[split].append(
            {
                "image_path": str(example.image_path),
                "mask_path": str(example.mask_path),
                "box": list(example.box),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Prepared SAM2 dataset at %s: %d train, %d val.", output_dir, train_count, val_count
    )
    return manifest_path


class Sam2TrainingConfig(BaseModel):
    """Hyperparameters and checkpoint paths for one SAM2 fine-tuning run.

    Attributes
    ----------
    manifest_path : pathlib.Path
        Path to a ``manifest.json`` from :func:`prepare_sam2_dataset`.
    base_config_file : str
        SAM2 model config name (e.g.
        ``"configs/sam2.1/sam2.1_hiera_l.yaml"``), matching
        ``base_checkpoint_path``.
    base_checkpoint_path : pathlib.Path
        Pretrained checkpoint to fine-tune from.
    output_checkpoint_path : pathlib.Path
        Where to write the fine-tuned checkpoint.
    epochs : int, default 20
        Fine-tuning epochs. Must be positive.
    learning_rate : float, default 1e-5
        AdamW learning rate for the prompt encoder and mask decoder. Must
        be positive.
    device : str, optional
        Training device. ``None`` uses CUDA if available, else CPU.
    seed : int, default 0
        Random seed, for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    manifest_path: Path
    base_config_file: str
    base_checkpoint_path: Path
    output_checkpoint_path: Path
    epochs: int = Field(default=DEFAULT_EPOCHS, gt=0)
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0)
    device: str | None = None
    seed: int = 0


class Sam2TrainingResult(BaseModel):
    """Outcome of a successful :func:`train_sam2` call.

    Attributes
    ----------
    checkpoint_path : pathlib.Path
        The fine-tuned checkpoint, loadable via
        :class:`~glas.ai.sam2_segmenter.Sam2Segmenter`'s
        ``config_file``/``checkpoint_path`` constructor arguments.
    epochs : int
        Epochs actually trained.
    metrics : dict of str to float
        ``"train_loss"`` (final epoch's mean training loss) and, if the
        manifest's val split is non-empty, ``"val_loss"`` and
        ``"val_mean_iou"``.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_path: Path
    epochs: int
    metrics: dict[str, float]


def _segmentation_loss(torch_module: Any, logits: Any, target: Any) -> Any:
    """Combined BCE + soft-Dice loss between predicted mask logits and a binary target."""
    bce = torch_module.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch_module.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(-1, -2))
    union = probabilities.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0))
    return bce + dice_loss.mean()


def _mask_logits_for_box(
    predictor: Any, torch_module: Any, box_xyxy: np.ndarray, orig_hw: tuple[int, int]
) -> Any:
    """Forward a single box prompt through the (already ``set_image``'d) predictor's heads.

    Mirrors ``SAM2ImagePredictor._predict``'s body, minus its
    ``@torch.no_grad()`` decorator -- inference-only there, but gradients
    through the prompt encoder and mask decoder are exactly what
    fine-tuning needs. Only ``sam_prompt_encoder``/``sam_mask_decoder``
    (not the frozen image encoder) receive gradients here.
    """
    model = predictor.model
    box_tensor = torch_module.as_tensor(
        box_xyxy, dtype=torch_module.float32, device=predictor.device
    )
    unnorm_box = predictor._transforms.transform_boxes(box_tensor, normalize=True, orig_hw=orig_hw)
    box_coords = unnorm_box.reshape(-1, 2, 2)
    box_labels = torch_module.tensor([[2, 3]], dtype=torch_module.int, device=box_coords.device)
    box_labels = box_labels.repeat(box_coords.size(0), 1)

    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None
    )
    high_res_features = [
        feature_level[-1].unsqueeze(0) for feature_level in predictor._features["high_res_feats"]
    ]
    low_res_masks, _iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    return predictor._transforms.postprocess_masks(low_res_masks, orig_hw)


def _load_example(
    torch_module: Any, device: Any, entry: dict[str, Any]
) -> tuple[np.ndarray, Any, np.ndarray]:
    image = cv2.imread(entry["image_path"])
    if image is None:
        raise AIModelError(f"Could not read training image {entry['image_path']!r}.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise AIModelError(f"Could not read training mask {entry['mask_path']!r}.")
    target = torch_module.as_tensor((mask > 0).astype(np.float32), device=device)[None, None, :, :]

    box = np.asarray(entry["box"], dtype=np.float32)
    return image_rgb, target, box


def _run_split(
    predictor: Any,
    torch_module: Any,
    entries: list[dict[str, Any]],
    *,
    optimizer: Any | None,
) -> dict[str, float]:
    """Run one epoch/evaluation pass over ``entries``; trains if ``optimizer`` is given.

    When ``optimizer`` is ``None`` (validation), the forward pass and loss
    are computed under ``torch.no_grad()`` -- nothing is backpropagated,
    so building an autograd graph would only waste memory.
    """
    total_loss = 0.0
    total_iou = 0.0
    for entry in entries:
        image_rgb, target, box = _load_example(torch_module, predictor.device, entry)
        predictor.set_image(image_rgb)
        orig_hw = image_rgb.shape[:2]

        grad_context = (
            torch_module.enable_grad() if optimizer is not None else torch_module.no_grad()
        )
        with grad_context:
            logits = _mask_logits_for_box(predictor, torch_module, box, orig_hw)
            loss = _segmentation_loss(torch_module, logits, target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch_module.no_grad():
            predicted = torch_module.sigmoid(logits) > 0.5
            ground_truth = target > 0.5
            intersection = float((predicted & ground_truth).sum().item())
            union = float((predicted | ground_truth).sum().item())
            total_iou += intersection / union if union > 0 else 1.0
        total_loss += float(loss.item())

    count = len(entries)
    return {"loss": total_loss / count, "mean_iou": total_iou / count}


def train_sam2(
    config: Sam2TrainingConfig,
    *,
    _predictor: Any | None = None,
    _build_sam2: Any | None = None,
    _torch: Any | None = None,
) -> Sam2TrainingResult:
    """Fine-tune SAM2's prompt encoder and mask decoder on box-prompted particle masks.

    Freezes the image encoder and trains only the prompt encoder and mask
    decoder for ``config.epochs`` epochs, then evaluates on the
    manifest's val split (mean loss and mean IoU against the ground-truth
    masks) before saving the fine-tuned checkpoint.

    Parameters
    ----------
    config : Sam2TrainingConfig

    Returns
    -------
    Sam2TrainingResult

    Raises
    ------
    AIDependencyError
        If ``torch`` or ``sam2`` is not installed.
    AIDatasetError
        If the manifest has no training examples.
    AIModelError
        If the base model cannot be loaded, or a training image/mask
        cannot be read.
    """
    torch_module = import_torch(_torch)
    device = config.device or ("cuda" if torch_module.cuda.is_available() else "cpu")

    if _predictor is not None:
        predictor = _predictor
    else:
        build_sam2 = import_build_sam2(_build_sam2)
        predictor_cls = import_sam2_image_predictor()
        try:
            model = build_sam2(
                config.base_config_file, str(config.base_checkpoint_path), device=device
            )
            predictor = predictor_cls(model)
        except Exception as exc:
            raise AIModelError(f"Could not load base SAM2 model: {exc}") from exc

    manifest = json.loads(config.manifest_path.read_text())
    train_entries = manifest.get("train", [])
    val_entries = manifest.get("val", [])
    if not train_entries:
        raise AIDatasetError(f"Manifest {config.manifest_path} has no training examples.")

    model = predictor.model
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = list(model.sam_prompt_encoder.parameters()) + list(
        model.sam_mask_decoder.parameters()
    )
    for parameter in trainable_parameters:
        parameter.requires_grad_(True)
    optimizer = torch_module.optim.AdamW(trainable_parameters, lr=config.learning_rate)

    torch_module.manual_seed(config.seed)
    train_metrics: dict[str, float] = {"loss": 0.0, "mean_iou": 0.0}
    for epoch in range(config.epochs):
        train_metrics = _run_split(predictor, torch_module, train_entries, optimizer=optimizer)
        logger.info(
            "SAM2 fine-tune epoch %d/%d: loss=%.4f, mean_iou=%.4f",
            epoch + 1,
            config.epochs,
            train_metrics["loss"],
            train_metrics["mean_iou"],
        )

    metrics = {"train_loss": train_metrics["loss"]}
    if val_entries:
        val_metrics = _run_split(predictor, torch_module, val_entries, optimizer=None)
        metrics["val_loss"] = val_metrics["loss"]
        metrics["val_mean_iou"] = val_metrics["mean_iou"]

    config.output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(model.state_dict(), config.output_checkpoint_path)

    logger.info(
        "Fine-tuned SAM2 model for %d epoch(s); checkpoint at %s.",
        config.epochs,
        config.output_checkpoint_path,
    )
    return Sam2TrainingResult(
        checkpoint_path=config.output_checkpoint_path, epochs=config.epochs, metrics=metrics
    )

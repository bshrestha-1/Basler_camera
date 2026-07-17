"""Training, validation, checkpoint management, and export for custom YOLO particle detectors.

    prepare_yolo_dataset() -> data.yaml -> train_yolo() -> YoloTrainingResult -> detector

Wraps ``ultralytics``'s own training loop (`YOLO.train`/`YOLO.val`/
`YOLO.export`) rather than reimplementing it -- GLAS's job here is
config validation, dataset-path bookkeeping, and turning ``ultralytics``'s
own outputs into a small typed result researchers can act on
programmatically, not replacing a well-tested training loop with a
from-scratch one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from glas.ai.dependencies import import_ultralytics
from glas.exceptions import AIModelError
from glas.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_WEIGHTS = "yolo11n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_PATIENCE = 50


class YoloTrainingConfig(BaseModel):
    """Hyperparameters and bookkeeping for one YOLO training run.

    Attributes
    ----------
    data_yaml : pathlib.Path
        Path to a ``data.yaml`` written by
        :func:`glas.ai.annotation.prepare_yolo_dataset`.
    base_weights : str, default "yolo11n.pt"
        Starting point: an Ultralytics-recognized pretrained model name
        (downloaded automatically) for transfer learning, a local
        ``.pt`` checkpoint to resume/fine-tune from, or a ``.yaml`` model
        architecture to train from scratch.
    epochs : int, default 100
        Training epochs. Must be positive.
    image_size : int, default 640
        Input image size (pixels, square). Must be positive.
    batch_size : int, default 16
        Training batch size. Must be positive.
    patience : int, default 50
        Epochs with no validation improvement before early stopping.
        Must be non-negative; ``0`` disables early stopping.
    device : str, optional
        Training device (e.g. ``"cpu"``, ``"cuda:0"``, ``"0,1"`` for
        multi-GPU). ``None`` lets ``ultralytics`` choose automatically.
    project : pathlib.Path, optional
        Parent directory for run output. ``None`` uses ``ultralytics``'s
        own default (``runs/detect``).
    name : str, default "glas_yolo"
        Run name -- output lands in ``project/name`` (with a numeric
        suffix if it already exists).
    resume : bool, default False
        Resume an interrupted run from ``base_weights`` (which must then
        be a ``last.pt`` checkpoint from that run).
    seed : int, default 0
        Random seed, for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    data_yaml: Path
    base_weights: str = DEFAULT_BASE_WEIGHTS
    epochs: int = Field(default=DEFAULT_EPOCHS, gt=0)
    image_size: int = Field(default=DEFAULT_IMAGE_SIZE, gt=0)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    patience: int = Field(default=DEFAULT_PATIENCE, ge=0)
    device: str | None = None
    project: Path | None = None
    name: str = "glas_yolo"
    resume: bool = False
    seed: int = 0


class YoloTrainingResult(BaseModel):
    """Outcome of a successful :func:`train_yolo` call.

    Attributes
    ----------
    best_weights : pathlib.Path
        Checkpoint with the best validation performance seen during
        training -- the one to load for inference
        (:class:`~glas.ai.yolo_detector.YoloParticleDetector`).
    last_weights : pathlib.Path
        Checkpoint from the final epoch -- the one to pass back in as
        ``base_weights`` to resume training later.
    save_dir : pathlib.Path
        Directory ``ultralytics`` wrote every run artifact to (weights,
        plots, ``results.csv``, ``args.yaml``).
    metrics : dict of str to float
        Final validation metrics (e.g. ``"metrics/mAP50(B)"``,
        ``"metrics/mAP50-95(B)"``, ``"fitness"``), exactly as
        ``ultralytics`` reports them.
    """

    model_config = ConfigDict(frozen=True)

    best_weights: Path
    last_weights: Path
    save_dir: Path
    metrics: dict[str, float]


def _extract_scalar_metrics(metrics_object: Any) -> dict[str, float]:
    results_dict = getattr(metrics_object, "results_dict", None)
    if not results_dict:
        return {}
    extracted: dict[str, float] = {}
    for key, value in dict(results_dict).items():
        try:
            extracted[key] = float(value)
        except (TypeError, ValueError):
            continue
    return extracted


def train_yolo(
    config: YoloTrainingConfig, *, _ultralytics: Any | None = None
) -> YoloTrainingResult:
    """Train a YOLO particle detector on a dataset prepared by :mod:`glas.ai.annotation`.

    Parameters
    ----------
    config : YoloTrainingConfig
        Training hyperparameters and dataset/output paths.

    Returns
    -------
    YoloTrainingResult

    Raises
    ------
    AIDependencyError
        If ``ultralytics`` is not installed.
    AIModelError
        If training fails (e.g. ``config.data_yaml`` doesn't exist, or
        the requested device is unavailable).
    """
    module = import_ultralytics(_ultralytics)
    try:
        model = module.YOLO(config.base_weights)
        model.train(
            data=str(config.data_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            patience=config.patience,
            device=config.device,
            project=str(config.project) if config.project is not None else None,
            name=config.name,
            resume=config.resume,
            seed=config.seed,
            verbose=False,
        )
    except Exception as exc:
        raise AIModelError(f"YOLO training failed: {exc}") from exc

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    result = YoloTrainingResult(
        best_weights=Path(trainer.best),
        last_weights=Path(trainer.last),
        save_dir=save_dir,
        metrics=_extract_scalar_metrics(getattr(trainer, "metrics", None)) or {},
    )
    logger.info(
        "Trained YOLO model for %d epoch(s); best weights at %s.",
        config.epochs,
        result.best_weights,
    )
    return result


def validate_yolo(
    weights: str | Path,
    data_yaml: Path,
    *,
    device: str | None = None,
    _ultralytics: Any | None = None,
) -> dict[str, float]:
    """Run validation for a trained (or in-training) YOLO model against a labeled dataset.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained model weights (``.pt``).
    data_yaml : pathlib.Path
        Dataset ``data.yaml``, as produced by
        :func:`glas.ai.annotation.prepare_yolo_dataset`.
    device : str, optional
        See :class:`YoloTrainingConfig`.

    Returns
    -------
    dict of str to float
        Validation metrics, exactly as ``ultralytics`` reports them (see
        :attr:`YoloTrainingResult.metrics`).

    Raises
    ------
    AIDependencyError
        If ``ultralytics`` is not installed.
    AIModelError
        If validation fails.
    """
    module = import_ultralytics(_ultralytics)
    try:
        model = module.YOLO(str(weights))
        metrics_object = model.val(data=str(data_yaml), device=device, verbose=False)
    except Exception as exc:
        raise AIModelError(f"YOLO validation failed: {exc}") from exc
    return _extract_scalar_metrics(metrics_object)


def export_yolo_model(
    weights: str | Path, export_format: str = "onnx", *, _ultralytics: Any | None = None
) -> Path:
    """Export a trained YOLO model to a deployment format.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained model weights (``.pt``).
    export_format : str, default "onnx"
        Any format ``ultralytics``'s own ``YOLO.export`` accepts (e.g.
        ``"onnx"``, ``"torchscript"``, ``"engine"`` for TensorRT).

    Returns
    -------
    pathlib.Path
        Path to the exported model file.

    Raises
    ------
    AIDependencyError
        If ``ultralytics`` is not installed.
    AIModelError
        If export fails (e.g. an optional export backend isn't
        installed).
    """
    module = import_ultralytics(_ultralytics)
    try:
        model = module.YOLO(str(weights))
        exported_path = model.export(format=export_format)
    except Exception as exc:
        raise AIModelError(f"YOLO export to {export_format!r} failed: {exc}") from exc
    return Path(exported_path)

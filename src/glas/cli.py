"""GLAS command-line interface.

Provides subcommands for inspecting the installed version, managing
configuration files, recording experiments, browsing recorded
experiments, exporting them to common image/video formats, detecting and
tracking particles across a recording, analyzing the Brazil nut
(intruder) effect, measuring bulk convection via optical flow, measuring
packing fraction and segregation, importing/analyzing/synchronizing
accelerometer recordings, controlling lab hardware (camera triggers, a
waveform generator, an oscilloscope, a shaker, and DAQ devices), and
detecting/classifying/segmenting particles with trained YOLO and SAM2
models (the ``ai`` subcommand group, requires `pip install glas[ai]`).
"""

from __future__ import annotations

import csv
import time
from enum import Enum
from pathlib import Path

import typer
import yaml

from glas.accelerometer import (
    DEFAULT_SENSITIVITY_MV_PER_G,
    analyze_vibration,
    import_accelerometer_csv,
    synchronize_with_frames,
)
from glas.ai.annotation import DEFAULT_LABEL, auto_annotate_dataset, prepare_yolo_dataset
from glas.ai.annotation import DEFAULT_VAL_FRACTION as DEFAULT_ANNOTATION_VAL_FRACTION
from glas.ai.dependencies import describe_missing_ai_packages, missing_ai_packages
from glas.ai.sam2_segmenter import Sam2Segmenter, compute_segmentation_summary
from glas.ai.sam2_train import DEFAULT_EPOCHS as DEFAULT_SAM2_EPOCHS
from glas.ai.sam2_train import (
    DEFAULT_LEARNING_RATE,
    Sam2TrainingConfig,
    auto_annotate_masks,
    prepare_sam2_dataset,
    train_sam2,
)
from glas.ai.yolo_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INTRUDER_LABEL,
    DEFAULT_IOU_THRESHOLD,
    YoloParticleDetector,
    track_dataset_yolo,
)
from glas.ai.yolo_train import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
    YoloTrainingConfig,
    train_yolo,
)
from glas.ai.yolo_train import DEFAULT_EPOCHS as DEFAULT_YOLO_EPOCHS
from glas.analysis import export_tracks_csv, track_dataset
from glas.analysis.brazil_nut import DEFAULT_SETTLE_FRACTION, analyze_brazil_nut
from glas.analysis.convection import DEFAULT_GRID_SPACING, analyze_convection
from glas.analysis.packing import DEFAULT_GRID_SPACING as DEFAULT_PACKING_GRID_SPACING
from glas.analysis.packing import analyze_packing, plot_packing_summary
from glas.analysis.particle_tracking import DEFAULT_MAX_DISTANCE, DEFAULT_MAX_GAP
from glas.analysis.segregation import DEFAULT_GRID_SPACING as DEFAULT_SEGREGATION_GRID_SPACING
from glas.analysis.segregation import analyze_segregation
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA
from glas.camera import Camera
from glas.config import deep_merge, read_yaml_file
from glas.controller import RecorderController
from glas.dataset import iter_frames
from glas.exceptions import (
    AccelerometerError,
    AIDatasetError,
    AIDependencyError,
    AIModelError,
    BrazilNutError,
    CameraConfigurationError,
    CameraConnectionError,
    CameraDriverError,
    CameraFeatureUnavailableError,
    CameraNotFoundError,
    ConfigurationError,
    ConvectionError,
    DatasetError,
    DatasetFormatError,
    DatasetIOError,
    ExperimentNotFoundError,
    ExportError,
    HardwareError,
    JSONValidationError,
    PackingError,
    SegregationError,
)
from glas.experiment import ExperimentManager
from glas.export import export_dataset
from glas.hardware.daq import AnalogInputDAQ, LabJackDAQ, NiDAQ
from glas.hardware.oscilloscope import SCPIOscilloscope
from glas.hardware.scpi import SocketSCPITransport
from glas.hardware.shaker import ShakerCalibration, ShakerController
from glas.hardware.waveform_generator import SiglentSDG1032X
from glas.logger import configure_logging, get_logger
from glas.settings import DEFAULT_CONFIG, Settings
from glas.version import __version__

app = typer.Typer(
    name="glas",
    help="GLAS: Granular Lab Acquisition System command-line interface.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage GLAS configuration files.")
app.add_typer(config_app, name="config")
experiment_app = typer.Typer(help="Browse recorded experiments.")
app.add_typer(experiment_app, name="experiment")
accelerometer_app = typer.Typer(help="Import and analyze accelerometer recordings.")
app.add_typer(accelerometer_app, name="accelerometer")
trigger_app = typer.Typer(help="Configure camera hardware triggering.")
app.add_typer(trigger_app, name="trigger")
waveform_app = typer.Typer(help="Control a Siglent SDG1032X waveform generator.")
app.add_typer(waveform_app, name="waveform-gen")
oscilloscope_app = typer.Typer(help="Query a SCPI-compliant oscilloscope.")
app.add_typer(oscilloscope_app, name="oscilloscope")
shaker_app = typer.Typer(help="Drive a shaker (via its waveform generator) to a target Gamma.")
app.add_typer(shaker_app, name="shaker")
daq_app = typer.Typer(help="Read analog input channels from a DAQ device.")
app.add_typer(daq_app, name="daq")
ai_app = typer.Typer(
    help="YOLO particle detection and SAM2 segmentation (requires `pip install glas\\[ai]`)."
)
app.add_typer(ai_app, name="ai")

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "glas" / "config.yaml"
DEFAULT_POLL_INTERVAL_SECONDS = 0.2


class _ExportFormatChoice(str, Enum):
    """CLI-facing mirror of :data:`glas.export.ExportFormat`, for validated ``--format`` choices."""

    tiff = "tiff"
    png = "png"
    mp4 = "mp4"
    avi = "avi"
    gif = "gif"


class _HeatmapBackgroundChoice(str, Enum):
    """CLI-facing mirror of :data:`glas.analysis.convection.HeatmapBackground`."""

    speed = "speed"
    vorticity = "vorticity"


class _ValueUnitsChoice(str, Enum):
    """CLI-facing mirror of :data:`glas.accelerometer.ValueUnits`."""

    volts = "volts"
    g = "g"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"glas {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the GLAS version and exit.",
    ),
) -> None:
    """GLAS: Granular Lab Acquisition System."""


@config_app.command("init")
def config_init(
    path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--path", "-p", help="Where to write the new configuration file."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite the file if it already exists."
    ),
) -> None:
    """Write a default configuration file to PATH."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    typer.echo(f"Wrote default configuration to {path}")


@config_app.command("validate")
def config_validate(
    path: Path = typer.Argument(..., help="Configuration file to validate."),
) -> None:
    """Validate a configuration file against the GLAS schema."""
    try:
        file_data = read_yaml_file(path)
        merged = deep_merge(DEFAULT_CONFIG, file_data)
        Settings.from_dict(merged)
    except (ConfigurationError, JSONValidationError) as exc:
        typer.echo(f"Invalid configuration: {exc}", err=True)
        if isinstance(exc, JSONValidationError):
            for error in exc.errors:
                typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{path} is valid.")


@config_app.command("show")
def config_show(
    path: Path | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Configuration file to load instead of the default search path.",
    ),
) -> None:
    """Load configuration (defaults merged with a file, if found) and print it."""
    try:
        settings = Settings.load(config_path=path)
    except (ConfigurationError, JSONValidationError) as exc:
        typer.echo(f"Could not load configuration: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for field_name, value in settings.__dict__.items():
        typer.echo(f"{field_name}: {value}")


@app.command("record")
def record(
    base_dir: Path = typer.Argument(..., help="Directory experiment folders are created under."),
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        help="Recording duration, in seconds. Omit to record until Ctrl+C.",
    ),
    name: str = typer.Option("", "--name", help="Human-readable experiment name."),
    tag: list[str] = typer.Option([], "--tag", help="Tag to attach to the experiment. Repeatable."),
    notes: str = typer.Option("", "--notes", help="Free-text operator notes."),
    serial: str | None = typer.Option(
        None, "--serial", help="Serial number of the camera to connect to, if more than one."
    ),
    exposure_us: float | None = typer.Option(
        None, "--exposure-us", help="Exposure time to set before recording, in microseconds."
    ),
    gain_db: float | None = typer.Option(
        None, "--gain-db", help="Gain to set before recording, in dB."
    ),
    dataset_format: str = typer.Option(
        "auto", "--format", help="Dataset storage format: hdf5, raw_binary, or auto."
    ),
) -> None:
    """Record an experiment: connect the camera, record, and finalize the dataset."""
    controller = RecorderController(base_dir)
    try:
        info = controller.connect(serial_number=serial)
    except (CameraNotFoundError, CameraConnectionError, CameraDriverError) as exc:
        typer.echo(f"Could not connect to camera: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Connected to {info.model_name} (serial {info.serial_number}).")
    if exposure_us is not None:
        controller.camera.exposure_time_us = exposure_us
    if gain_db is not None:
        controller.camera.gain_db = gain_db

    recorder = None
    try:
        with controller.graceful_shutdown() as shutdown:
            recorder = controller.start_recording(
                notes=notes,
                name=name,
                tags=tag,
                dataset_format=dataset_format,
            )
            typer.echo(f"Recording to {recorder.dataset.folder} -- press Ctrl+C to stop.")
            deadline = None if duration is None else time.monotonic() + duration
            while not shutdown.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
    finally:
        controller.disconnect()

    if recorder is not None:
        typer.echo(
            f"Recorded {recorder.dataset.metadata.frame_count} frame(s) to "
            f"{recorder.dataset.folder}."
        )


@app.command("gui")
def gui(
    base_dir: Path = typer.Argument(..., help="Directory experiment folders are created under."),
) -> None:
    """Launch the GLAS desktop GUI.

    Requires the optional ``gui`` extra (``pip install glas\\[gui]``); the
    rest of GLAS, including every other CLI command, works without it.
    """
    try:
        from glas.gui.app import main as gui_main
    except ImportError as exc:
        typer.echo(
            "The GUI requires PySide6, which is not installed. "
            "Install it with: pip install glas[gui]",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    raise typer.Exit(code=gui_main(base_dir))


@experiment_app.command("list")
def experiment_list(
    base_dir: Path = typer.Argument(..., help="Directory experiment folders live under."),
    tag: str | None = typer.Option(None, "--tag", help="Only experiments with this tag."),
    name: str | None = typer.Option(
        None, "--name", help="Only experiments whose name contains this (case-insensitive)."
    ),
) -> None:
    """List recorded experiments under BASE_DIR."""
    manager = ExperimentManager(base_dir)
    summaries = manager.search_experiments(name_contains=name, tag=tag)
    if not summaries:
        typer.echo("No experiments found.")
        return

    for summary in summaries:
        tags = ", ".join(summary.tags) if summary.tags else "-"
        typer.echo(
            f"{summary.run_id}  name={summary.name or '(unnamed)'!r}  "
            f"frames={summary.frame_count}  tags=[{tags}]"
        )


@experiment_app.command("show")
def experiment_show(
    base_dir: Path = typer.Argument(..., help="Directory experiment folders live under."),
    run_id: str = typer.Argument(..., help="Experiment folder name, e.g. Run0001."),
) -> None:
    """Show details for one recorded experiment."""
    manager = ExperimentManager(base_dir)
    try:
        summary = manager.get_experiment(run_id)
    except ExperimentNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"run_id: {summary.run_id}")
    typer.echo(f"name: {summary.name or '(unnamed)'}")
    typer.echo(f"tags: {', '.join(summary.tags) or '(none)'}")
    typer.echo(f"notes: {summary.notes or '(none)'}")
    typer.echo(f"camera_model: {summary.camera_model}")
    typer.echo(f"frame_count: {summary.frame_count}")
    typer.echo(f"created_at_utc: {summary.created_at_utc}")
    typer.echo(f"folder: {summary.folder}")


@app.command("export")
def export(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to export."),
    output: Path = typer.Argument(
        ..., help="Destination: a directory for tiff/png, a file for mp4/avi/gif."
    ),
    format: _ExportFormatChoice = typer.Option(..., "--format", "-f", help="Output format."),
    fps: float = typer.Option(30.0, "--fps", help="Playback fps for mp4/avi/gif."),
    start_frame: int | None = typer.Option(
        None, "--start-frame", help="First frame to export (0-based, inclusive)."
    ),
    end_frame: int | None = typer.Option(
        None, "--end-frame", help="Last frame to export (0-based, exclusive)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-y", help="Replace an existing destination."
    ),
) -> None:
    """Export a recorded dataset to an image sequence, video, or GIF."""
    try:
        result = export_dataset(
            dataset_folder,
            output,
            format.value,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            overwrite=overwrite,
        )
    except (ExportError, DatasetError, DatasetFormatError, DatasetIOError) as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Exported {result.frame_count} frame(s) to {result.output_path}.")


@app.command("analyze")
def analyze(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    max_distance: float = typer.Option(
        DEFAULT_MAX_DISTANCE,
        "--max-distance",
        help="Max pixel distance to link a detection onto an existing track.",
    ),
    max_gap: int = typer.Option(
        DEFAULT_MAX_GAP,
        "--max-gap",
        help="Frames a track may go unmatched before it's retired.",
    ),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
    csv_output: Path | None = typer.Option(
        None, "--csv", help="Also write every tracked observation to this CSV file."
    ),
) -> None:
    """Detect and track particles across a recorded dataset."""
    try:
        history = track_dataset(
            dataset_folder,
            max_distance=max_distance,
            max_gap=max_gap,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
        )
    except (DatasetError, DatasetFormatError, DatasetIOError, JSONValidationError) as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not history:
        typer.echo("No particles detected.")
        return

    lengths = [len(observations) for observations in history.values()]
    typer.echo(f"Tracked {len(history)} particle(s) across the recording.")
    typer.echo(f"Mean track length: {sum(lengths) / len(lengths):.1f} frame(s).")
    typer.echo(f"Longest track: {max(lengths)} frame(s).")

    if csv_output is not None:
        row_count = export_tracks_csv(history, csv_output)
        typer.echo(f"Wrote {row_count} row(s) to {csv_output}.")


def _describe_ai_dependency_error(exc: AIDependencyError) -> None:
    missing = missing_ai_packages()
    if missing:
        typer.echo(describe_missing_ai_packages(missing), err=True)
    else:
        typer.echo(str(exc), err=True)


@ai_app.command("detect")
def ai_detect(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    weights: str = typer.Argument(
        ..., help="Path to trained YOLO weights (.pt), or a pretrained model name."
    ),
    max_distance: float = typer.Option(
        DEFAULT_MAX_DISTANCE,
        "--max-distance",
        help="Max pixel distance to link a detection onto an existing track.",
    ),
    max_gap: int = typer.Option(
        DEFAULT_MAX_GAP,
        "--max-gap",
        help="Frames a track may go unmatched before it's retired.",
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE_THRESHOLD, "--confidence", help="Minimum detection confidence, 0-1."
    ),
    iou: float = typer.Option(
        DEFAULT_IOU_THRESHOLD, "--iou", help="IoU threshold for the model's own NMS, 0-1."
    ),
    intruder_label: str = typer.Option(
        DEFAULT_INTRUDER_LABEL,
        "--intruder-label",
        help="Class name (case-insensitive) treated as the Brazil-nut-style intruder.",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Inference device (e.g. cpu, cuda:0). Omit to auto-select."
    ),
    csv_output: Path | None = typer.Option(
        None, "--csv", help="Also write every tracked observation to this CSV file."
    ),
) -> None:
    """Detect, classify, and track particles across a dataset using a trained YOLO model."""
    try:
        history = track_dataset_yolo(
            dataset_folder,
            weights,
            max_distance=max_distance,
            max_gap=max_gap,
            confidence_threshold=confidence,
            iou_threshold=iou,
            intruder_label=intruder_label,
            device=device,
        )
    except AIDependencyError as exc:
        _describe_ai_dependency_error(exc)
        raise typer.Exit(code=1) from exc
    except (
        AIModelError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Detection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not history:
        typer.echo("No particles detected.")
        return

    lengths = [len(observations) for observations in history.values()]
    intruder_count = sum(
        1 for observations in history.values() if any(o.is_intruder for o in observations)
    )
    label_counts: dict[str, int] = {}
    for observations in history.values():
        label = observations[-1].label
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1

    typer.echo(f"Tracked {len(history)} particle(s) across the recording.")
    typer.echo(f"Mean track length: {sum(lengths) / len(lengths):.1f} frame(s).")
    typer.echo(f"Longest track: {max(lengths)} frame(s).")
    if intruder_count:
        typer.echo(f"Intruder(s) detected: {intruder_count}.")
    for label, count in sorted(label_counts.items()):
        typer.echo(f"  {label}: {count}")

    if csv_output is not None:
        row_count = export_tracks_csv(history, csv_output)
        typer.echo(f"Wrote {row_count} row(s) to {csv_output}.")


@ai_app.command("prepare-yolo-dataset")
def ai_prepare_yolo_dataset(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to auto-annotate."),
    output_dir: Path = typer.Argument(..., help="Directory to write the prepared dataset into."),
    label: str = typer.Option(
        DEFAULT_LABEL, "--label", help="Class name applied to every auto-detected blob."
    ),
    val_fraction: float = typer.Option(
        DEFAULT_ANNOTATION_VAL_FRACTION,
        "--val-fraction",
        help="Fraction of frames held out for validation.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the deterministic train/val shuffle."),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
) -> None:
    """Bootstrap a YOLO training dataset from a recording via classical blob detection.

    Writes every frame's image plus a first-pass, single-class label set;
    hand-correct the boxes/labels with an external annotation tool before
    training if better-than-bootstrap accuracy is needed.
    """
    staging_dir = output_dir / "_staging"
    annotations = auto_annotate_dataset(
        dataset_folder,
        staging_dir,
        label=label,
        min_area=min_area,
        max_area=max_area,
        threshold=threshold,
        invert=invert,
    )
    try:
        data_yaml_path = prepare_yolo_dataset(
            annotations, [label], output_dir, val_fraction=val_fraction, seed=seed
        )
    except AIDatasetError as exc:
        typer.echo(f"Dataset preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {data_yaml_path}")


@ai_app.command("train-yolo")
def ai_train_yolo(
    data_yaml: Path = typer.Argument(..., help="Dataset data.yaml, from prepare-yolo-dataset."),
    base_weights: str = typer.Option(
        DEFAULT_BASE_WEIGHTS, "--base-weights", help="Pretrained weights (or .yaml) to start from."
    ),
    epochs: int = typer.Option(DEFAULT_YOLO_EPOCHS, "--epochs", help="Training epochs."),
    image_size: int = typer.Option(
        DEFAULT_IMAGE_SIZE, "--image-size", help="Input image size, pixels."
    ),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", help="Training batch size."),
    patience: int = typer.Option(
        DEFAULT_PATIENCE, "--patience", help="Epochs with no improvement before early stopping."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Training device (e.g. cpu, cuda:0). Omit to auto-select."
    ),
    project: Path | None = typer.Option(
        None, "--project", help="Parent directory for run output. Omit for ultralytics's default."
    ),
    name: str = typer.Option("glas_yolo", "--name", help="Run name."),
    resume: bool = typer.Option(False, "--resume", help="Resume an interrupted run."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Train a custom YOLO particle detector on a dataset from prepare-yolo-dataset."""
    config = YoloTrainingConfig(
        data_yaml=data_yaml,
        base_weights=base_weights,
        epochs=epochs,
        image_size=image_size,
        batch_size=batch_size,
        patience=patience,
        device=device,
        project=project,
        name=name,
        resume=resume,
        seed=seed,
    )
    try:
        result = train_yolo(config)
    except AIDependencyError as exc:
        _describe_ai_dependency_error(exc)
        raise typer.Exit(code=1) from exc
    except AIModelError as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Best weights: {result.best_weights}")
    typer.echo(f"Last weights: {result.last_weights}")
    for key, value in sorted(result.metrics.items()):
        typer.echo(f"  {key}: {value:.4f}")


@ai_app.command("segment")
def ai_segment(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to segment."),
    weights: str = typer.Argument(
        ..., help="Path to trained YOLO weights, used to prompt SAM2 with per-particle boxes."
    ),
    model_id: str | None = typer.Option(
        None, "--model-id", help="Pretrained SAM2 Hugging Face Hub model id."
    ),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Local SAM2 model config (with --checkpoint-path)."
    ),
    checkpoint_path: Path | None = typer.Option(
        None, "--checkpoint-path", help="Local SAM2 checkpoint (with --config-file)."
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE_THRESHOLD, "--confidence", help="Minimum YOLO detection confidence, 0-1."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Inference device (e.g. cpu, cuda:0). Omit to auto-select."
    ),
    csv_output: Path | None = typer.Option(
        None, "--csv", help="Write per-particle shape metrics for the last frame to this CSV file."
    ),
) -> None:
    """Segment every particle's exact outline in a dataset's last frame using YOLO + SAM2.

    YOLO locates each particle; SAM2 refines every box into a pixel-exact
    mask and reports area, perimeter, orientation, aspect ratio, packing
    fraction, and void fraction.
    """
    try:
        detector = YoloParticleDetector(weights, confidence_threshold=confidence, device=device)
        segmenter = Sam2Segmenter(
            model_id=model_id,
            config_file=config_file,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    except AIDependencyError as exc:
        _describe_ai_dependency_error(exc)
        raise typer.Exit(code=1) from exc
    except AIModelError as exc:
        typer.echo(f"Could not load model(s): {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        last_frame = None
        for frame in iter_frames(dataset_folder):
            last_frame = frame
        if last_frame is None:
            typer.echo("Dataset has no frames.")
            return

        detections = detector.detect(last_frame.image)
        segments = segmenter.segment_frame(last_frame.image, detections)
    except (
        AIModelError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Segmentation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not segments:
        typer.echo("No particles detected.")
        return

    summary = compute_segmentation_summary(segments, last_frame.image.shape[:2])
    typer.echo(f"Segmented {summary.particle_count} particle(s).")
    typer.echo(f"Packing fraction: {summary.packing_fraction:.3f}")
    typer.echo(f"Void fraction: {summary.void_fraction:.3f}")
    typer.echo(f"Contacts: {len(summary.contacts)}")

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "index",
                    "area_px",
                    "perimeter_px",
                    "centroid_x",
                    "centroid_y",
                    "orientation_deg",
                    "aspect_ratio",
                    "score",
                ]
            )
            for index, segment in enumerate(segments):
                metrics = segment.metrics
                writer.writerow(
                    [
                        index,
                        metrics.area_px,
                        metrics.perimeter_px,
                        metrics.centroid_x,
                        metrics.centroid_y,
                        metrics.orientation_deg,
                        metrics.aspect_ratio,
                        segment.score,
                    ]
                )
        typer.echo(f"Wrote {len(segments)} row(s) to {csv_output}.")


@ai_app.command("prepare-sam2-dataset")
def ai_prepare_sam2_dataset(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to auto-annotate."),
    output_dir: Path = typer.Argument(..., help="Directory to write the prepared dataset into."),
    val_fraction: float = typer.Option(
        DEFAULT_ANNOTATION_VAL_FRACTION,
        "--val-fraction",
        help="Fraction of examples held out for validation.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the deterministic train/val shuffle."),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
) -> None:
    """Bootstrap a SAM2 fine-tuning dataset from a recording via classical contour detection."""
    staging_dir = output_dir / "_staging"
    examples = auto_annotate_masks(
        dataset_folder,
        staging_dir,
        min_area=min_area,
        max_area=max_area,
        threshold=threshold,
        invert=invert,
    )
    try:
        manifest_path = prepare_sam2_dataset(
            examples, output_dir, val_fraction=val_fraction, seed=seed
        )
    except AIDatasetError as exc:
        typer.echo(f"Dataset preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {manifest_path}")


@ai_app.command("train-sam2")
def ai_train_sam2(
    manifest_path: Path = typer.Argument(
        ..., help="Dataset manifest.json, from prepare-sam2-dataset."
    ),
    base_config_file: str = typer.Argument(
        ..., help="SAM2 model config name matching the base checkpoint."
    ),
    base_checkpoint_path: Path = typer.Argument(
        ..., help="Pretrained SAM2 checkpoint to fine-tune from."
    ),
    output_checkpoint_path: Path = typer.Argument(
        ..., help="Where to write the fine-tuned checkpoint."
    ),
    epochs: int = typer.Option(DEFAULT_SAM2_EPOCHS, "--epochs", help="Fine-tuning epochs."),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE, "--learning-rate", help="AdamW learning rate."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Training device (e.g. cpu, cuda:0). Omit to auto-select."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Fine-tune SAM2's prompt encoder and mask decoder on a dataset from prepare-sam2-dataset."""
    config = Sam2TrainingConfig(
        manifest_path=manifest_path,
        base_config_file=base_config_file,
        base_checkpoint_path=base_checkpoint_path,
        output_checkpoint_path=output_checkpoint_path,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    try:
        result = train_sam2(config)
    except AIDependencyError as exc:
        _describe_ai_dependency_error(exc)
        raise typer.Exit(code=1) from exc
    except (AIDatasetError, AIModelError) as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Fine-tuned checkpoint: {result.checkpoint_path}")
    for key, value in sorted(result.metrics.items()):
        typer.echo(f"  {key}: {value:.4f}")


@app.command("brazil-nut")
def brazil_nut(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    track_id: int | None = typer.Option(
        None, "--track-id", help="Track to analyze. Omit to auto-identify the largest particle."
    ),
    settle_fraction: float = typer.Option(
        DEFAULT_SETTLE_FRACTION,
        "--settle-fraction",
        help="Fraction of the frame height counted as 'risen', for rise time.",
    ),
    plot: Path | None = typer.Option(
        None, "--plot", help="Save a height/velocity plot to this PNG file."
    ),
    max_distance: float = typer.Option(
        DEFAULT_MAX_DISTANCE,
        "--max-distance",
        help="Max pixel distance to link a detection onto an existing track.",
    ),
    max_gap: int = typer.Option(
        DEFAULT_MAX_GAP,
        "--max-gap",
        help="Frames a track may go unmatched before it's retired.",
    ),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
) -> None:
    """Analyze the Brazil nut (intruder) effect: height, rise time, and velocity."""
    try:
        trajectory = analyze_brazil_nut(
            dataset_folder,
            track_id=track_id,
            settle_fraction=settle_fraction,
            plot_path=plot,
            max_distance=max_distance,
            max_gap=max_gap,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
        )
    except (
        BrazilNutError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Brazil nut analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Brazil nut track: {trajectory.track_id}")
    rise_time = (
        f"{trajectory.rise_time_s:.2f}s" if trajectory.rise_time_s is not None else "not reached"
    )
    typer.echo(f"Rise time: {rise_time}")
    typer.echo(f"Mean rise velocity: {trajectory.mean_velocity_px_s:.1f} px/s")
    if plot is not None:
        typer.echo(f"Plot saved to {plot}")


@app.command("convection")
def convection(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    grid_spacing: int = typer.Option(
        DEFAULT_GRID_SPACING,
        "--grid-spacing",
        help="Pixel spacing between adjacent velocity samples.",
    ),
    heatmap_dir: Path | None = typer.Option(
        None, "--heatmap-dir", help="Save one heat map PNG per frame pair here."
    ),
    heatmap_background: _HeatmapBackgroundChoice = typer.Option(
        _HeatmapBackgroundChoice.speed,
        "--heatmap-background",
        help="Which field to render as the heat map background.",
    ),
) -> None:
    """Measure bulk convection via dense optical flow: velocity field and circulation."""
    try:
        summary = analyze_convection(
            dataset_folder,
            grid_spacing=grid_spacing,
            heatmap_dir=heatmap_dir,
            heatmap_background=heatmap_background.value,
        )
    except (
        ConvectionError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Convection analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    circulations = summary.circulations
    typer.echo(f"Analyzed {len(summary.fields)} frame pair(s).")
    typer.echo(f"Mean circulation: {sum(circulations) / len(circulations):.1f} px^2/s")
    typer.echo(f"Min/max circulation: {min(circulations):.1f} / {max(circulations):.1f} px^2/s")
    if heatmap_dir is not None:
        typer.echo(f"Saved {len(summary.fields)} heat map(s) to {heatmap_dir}")


@app.command("packing")
def packing(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    roi_area: float | None = typer.Option(
        None, "--roi-area", help="ROI area, in px^2. Omit to use the whole frame."
    ),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
    field_grid_spacing: int = typer.Option(
        DEFAULT_PACKING_GRID_SPACING,
        "--field-grid-spacing",
        help="Grid spacing for the spatial packing-fraction field (only used with --field-dir).",
    ),
    field_dir: Path | None = typer.Option(
        None,
        "--field-dir",
        help="If given, also save one packing-field heat map PNG per frame here.",
    ),
    plot: Path | None = typer.Option(
        None, "--plot", help="Save a packing-fraction-over-time plot to this PNG file."
    ),
) -> None:
    """Measure packing fraction, void fraction, and particle density across a recording."""
    try:
        summary = analyze_packing(
            dataset_folder,
            roi_area=roi_area,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
            field_grid_spacing=field_grid_spacing,
            field_dir=field_dir,
        )
    except (
        PackingError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Packing analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    fractions = [m.packing_fraction for m in summary.metrics]
    typer.echo(f"Analyzed {len(summary.metrics)} frame(s).")
    typer.echo(f"Mean packing fraction: {sum(fractions) / len(fractions):.4f}")
    typer.echo(f"Min/max packing fraction: {min(fractions):.4f} / {max(fractions):.4f}")
    if field_dir is not None:
        typer.echo(f"Saved {len(summary.metrics)} packing field heat map(s) to {field_dir}")
    if plot is not None:
        plot_packing_summary(summary, plot)
        typer.echo(f"Plot saved to {plot}")


@app.command("segregation")
def segregation(
    dataset_folder: Path = typer.Argument(..., help="Dataset folder to analyze."),
    size_threshold: float | None = typer.Option(
        None,
        "--size-threshold",
        help="Equivalent-radius threshold, in px, splitting large/small particles. Omit for "
        "the automatic median.",
    ),
    grid_spacing: int = typer.Option(
        DEFAULT_SEGREGATION_GRID_SPACING,
        "--grid-spacing",
        help="Pixel size of the composition-sampling grid.",
    ),
    min_area: float = typer.Option(
        DEFAULT_MIN_AREA, "--min-area", help="Minimum blob area, in pixels, to count as a particle."
    ),
    max_area: float | None = typer.Option(None, "--max-area", help="Maximum blob area, in pixels."),
    threshold: int | None = typer.Option(
        None, "--threshold", help="Explicit 0-255 threshold. Omit for automatic (Otsu)."
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Detect dark particles on a bright background."
    ),
    plot: Path | None = typer.Option(
        None, "--plot", help="Save a segregation/mixing/entropy plot to this PNG file."
    ),
) -> None:
    """Measure segregation index, mixing index, and mixing entropy across a recording."""
    try:
        summary = analyze_segregation(
            dataset_folder,
            size_threshold=size_threshold,
            grid_spacing=grid_spacing,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
            plot_path=plot,
        )
    except (
        SegregationError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
    ) as exc:
        typer.echo(f"Segregation analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    seg_idx = [m.segregation_index for m in summary.metrics]
    mix_idx = [m.mixing_index for m in summary.metrics]
    entropy = [m.mixing_entropy for m in summary.metrics]
    typer.echo(f"Analyzed {len(summary.metrics)} frame(s).")
    typer.echo(f"Mean segregation index: {sum(seg_idx) / len(seg_idx):.4f}")
    typer.echo(f"Mean mixing index: {sum(mix_idx) / len(mix_idx):.4f}")
    typer.echo(f"Mean mixing entropy: {sum(entropy) / len(entropy):.4f}")
    if plot is not None:
        typer.echo(f"Plot saved to {plot}")


@accelerometer_app.command("analyze")
def accelerometer_analyze(
    csv_path: Path = typer.Argument(..., help="Accelerometer CSV file to import."),
    time_column: str = typer.Option(
        "time_s", "--time-column", help="Header of the timestamp column, in seconds."
    ),
    value_column: str = typer.Option(
        "voltage_v", "--value-column", help="Header of the measured-value column."
    ),
    value_units: _ValueUnitsChoice = typer.Option(
        _ValueUnitsChoice.volts,
        "--value-units",
        help="Whether the value column already holds g, or raw sensor voltage.",
    ),
    sensitivity_mv_per_g: float = typer.Option(
        DEFAULT_SENSITIVITY_MV_PER_G,
        "--sensitivity-mv-per-g",
        help="Accelerometer sensitivity, in mV/g (from its calibration certificate). "
        "Only used with --value-units volts.",
    ),
    plot: Path | None = typer.Option(
        None, "--plot", help="Save a time-domain signal plot to this PNG file."
    ),
) -> None:
    """Compute vibration frequency, displacement amplitude, and Gamma from an accelerometer CSV."""
    try:
        metrics = analyze_vibration(
            csv_path,
            time_column=time_column,
            value_column=value_column,
            value_units=value_units.value,
            sensitivity_mv_per_g=sensitivity_mv_per_g,
            plot_path=plot,
        )
    except AccelerometerError as exc:
        typer.echo(f"Accelerometer analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Frequency: {metrics.frequency_hz:.2f} Hz")
    typer.echo(f"Amplitude: {metrics.amplitude_m * 1000:.4f} mm")
    typer.echo(f"Gamma: {metrics.gamma:.3f}")
    typer.echo(f"Peak acceleration: {metrics.peak_acceleration_g:.3f} g")
    if plot is not None:
        typer.echo(f"Plot saved to {plot}")


@accelerometer_app.command("sync")
def accelerometer_sync(
    csv_path: Path = typer.Argument(..., help="Accelerometer CSV file to import."),
    dataset_folder: Path = typer.Argument(
        ..., help="Dataset folder whose frames to synchronize with."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Where to save the per-frame synchronized acceleration, as CSV."
    ),
    offset_s: float = typer.Option(
        0.0,
        "--offset-s",
        help="Time offset, in seconds, to align the two recordings (positive if the "
        "accelerometer recording started before the camera).",
    ),
    time_column: str = typer.Option(
        "time_s", "--time-column", help="Header of the timestamp column, in seconds."
    ),
    value_column: str = typer.Option(
        "voltage_v", "--value-column", help="Header of the measured-value column."
    ),
    value_units: _ValueUnitsChoice = typer.Option(
        _ValueUnitsChoice.volts,
        "--value-units",
        help="Whether the value column already holds g, or raw sensor voltage.",
    ),
    sensitivity_mv_per_g: float = typer.Option(
        DEFAULT_SENSITIVITY_MV_PER_G,
        "--sensitivity-mv-per-g",
        help="Accelerometer sensitivity, in mV/g (from its calibration certificate). "
        "Only used with --value-units volts.",
    ),
) -> None:
    """Synchronize an accelerometer recording with a dataset's frames, one value per frame."""
    try:
        recording = import_accelerometer_csv(
            csv_path,
            time_column=time_column,
            value_column=value_column,
            value_units=value_units.value,
            sensitivity_mv_per_g=sensitivity_mv_per_g,
        )
        frames = list(iter_frames(dataset_folder))
        accelerations = synchronize_with_frames(recording, frames, offset_s=offset_s)
    except (
        AccelerometerError,
        DatasetError,
        DatasetFormatError,
        DatasetIOError,
        JSONValidationError,
        ValueError,
    ) as exc:
        typer.echo(f"Accelerometer synchronization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "host_timestamp_ns", "acceleration_g"])
        for frame, acceleration in zip(frames, accelerations, strict=True):
            writer.writerow([frame.frame_id, frame.host_timestamp_ns, acceleration])

    typer.echo(f"Synchronized {len(frames)} frame(s); wrote {output}")


@trigger_app.command("enable")
def trigger_enable(
    serial: str | None = typer.Option(
        None, "--serial", help="Serial number of the camera to connect to, if more than one."
    ),
    source: str = typer.Option(
        "Line1", "--source", help="Trigger source, e.g. a physical input line."
    ),
    activation: str = typer.Option(
        "RisingEdge", "--activation", help="Which edge or level of the signal fires the camera."
    ),
) -> None:
    """Configure the camera to wait for an external hardware trigger signal."""
    camera = Camera()
    try:
        info = camera.connect(serial_number=serial)
        camera.enable_hardware_trigger(source=source, activation=activation)
    except (
        CameraNotFoundError,
        CameraConnectionError,
        CameraDriverError,
        CameraConfigurationError,
        CameraFeatureUnavailableError,
    ) as exc:
        typer.echo(f"Could not enable hardware trigger: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        camera.disconnect()

    typer.echo(
        f"Hardware trigger enabled on {info.model_name} (source={source}, activation={activation})."
    )


@trigger_app.command("disable")
def trigger_disable(
    serial: str | None = typer.Option(
        None, "--serial", help="Serial number of the camera to connect to, if more than one."
    ),
) -> None:
    """Return the camera to free-running (untriggered) acquisition."""
    camera = Camera()
    try:
        info = camera.connect(serial_number=serial)
        camera.disable_hardware_trigger()
    except (
        CameraNotFoundError,
        CameraConnectionError,
        CameraDriverError,
        CameraFeatureUnavailableError,
    ) as exc:
        typer.echo(f"Could not disable hardware trigger: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        camera.disconnect()

    typer.echo(f"Hardware trigger disabled on {info.model_name}.")


@trigger_app.command("status")
def trigger_status(
    serial: str | None = typer.Option(
        None, "--serial", help="Serial number of the camera to connect to, if more than one."
    ),
) -> None:
    """Report whether the camera is currently configured for hardware triggering."""
    camera = Camera()
    try:
        info = camera.connect(serial_number=serial)
        triggered = camera.is_hardware_triggered()
    except (
        CameraNotFoundError,
        CameraConnectionError,
        CameraDriverError,
        CameraFeatureUnavailableError,
    ) as exc:
        typer.echo(f"Could not read trigger status: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        camera.disconnect()

    state = "enabled" if triggered else "disabled"
    typer.echo(f"Hardware trigger is {state} on {info.model_name}.")


@waveform_app.command("sine")
def waveform_gen_sine(
    host: str = typer.Argument(..., help="Waveform generator's IP address or hostname."),
    frequency_hz: float = typer.Option(..., "--frequency-hz", help="Frequency, in Hz."),
    amplitude_vpp: float = typer.Option(
        ..., "--amplitude-vpp", help="Peak-to-peak amplitude, in volts."
    ),
    channel: int = typer.Option(1, "--channel", help="Output channel, 1 or 2."),
    offset_v: float = typer.Option(0.0, "--offset-v", help="DC offset, in volts."),
    port: int = typer.Option(5025, "--port", help="SCPI-raw TCP port."),
    enable: bool = typer.Option(False, "--enable", help="Also turn the channel's output on."),
) -> None:
    """Configure a Siglent SDG1032X channel to output a sine wave."""
    try:
        transport = SocketSCPITransport(host, port=port)
    except HardwareError as exc:
        typer.echo(f"Could not connect to waveform generator: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    generator = SiglentSDG1032X(transport)
    try:
        generator.set_sine_wave(
            channel, frequency_hz=frequency_hz, amplitude_vpp=amplitude_vpp, offset_v=offset_v
        )
        if enable:
            generator.enable_output(channel)
    except (ValueError, HardwareError) as exc:
        typer.echo(f"Could not configure waveform generator: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        generator.close()

    typer.echo(
        f"Channel {channel} set to {frequency_hz} Hz, {amplitude_vpp} Vpp"
        f"{' (output enabled)' if enable else ''}."
    )


@oscilloscope_app.command("query")
def oscilloscope_query(
    host: str = typer.Argument(..., help="Oscilloscope's IP address or hostname."),
    command: str = typer.Argument(..., help="Raw SCPI query to send, e.g. '*IDN?'."),
    port: int = typer.Option(5025, "--port", help="SCPI-raw TCP port."),
) -> None:
    """Send a raw SCPI query to an oscilloscope and print its response."""
    try:
        transport = SocketSCPITransport(host, port=port)
    except HardwareError as exc:
        typer.echo(f"Could not connect to oscilloscope: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    scope = SCPIOscilloscope(transport)
    try:
        response = scope.query(command)
    except HardwareError as exc:
        typer.echo(f"Query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        scope.close()

    typer.echo(response)


@shaker_app.command("set-gamma")
def shaker_set_gamma(
    host: str = typer.Argument(..., help="Waveform generator's IP address or hostname."),
    gamma: float = typer.Argument(..., help="Target dimensionless vibration intensity."),
    volts_per_g: float = typer.Option(
        ..., "--volts-per-g", help="Calibrated drive voltage per g of peak acceleration."
    ),
    calibration_frequency_hz: float = typer.Option(
        ..., "--calibration-frequency-hz", help="Frequency the calibration was measured at."
    ),
    frequency_hz: float | None = typer.Option(
        None,
        "--frequency-hz",
        help="Drive frequency, in Hz. Defaults to the calibration frequency.",
    ),
    channel: int = typer.Option(1, "--channel", help="Output channel wired to the amplifier."),
    port: int = typer.Option(5025, "--port", help="SCPI-raw TCP port."),
    start: bool = typer.Option(False, "--start", help="Also turn the channel's output on."),
) -> None:
    """Drive a shaker (via its waveform generator) to a target Gamma."""
    try:
        transport = SocketSCPITransport(host, port=port)
    except HardwareError as exc:
        typer.echo(f"Could not connect to waveform generator: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    generator = SiglentSDG1032X(transport)
    controller = ShakerController(
        generator,
        ShakerCalibration(volts_per_g=volts_per_g, frequency_hz=calibration_frequency_hz),
        channel=channel,
    )
    try:
        voltage = controller.set_target_gamma(gamma, frequency_hz=frequency_hz)
        if start:
            controller.start()
    except (ValueError, HardwareError) as exc:
        typer.echo(f"Could not set target Gamma: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        generator.close()

    typer.echo(
        f"Gamma={gamma} -> drive voltage {voltage:.4f} Vpp{' (output enabled)' if start else ''}."
    )


@daq_app.command("read")
def daq_read(
    backend: str = typer.Argument(..., help="DAQ backend: 'labjack' or 'ni'."),
    channel: int = typer.Option(0, "--channel", help="Analog input channel to read."),
    device_type: str = typer.Option("ANY", "--device-type", help="LabJack device type."),
    connection_type: str = typer.Option(
        "ANY", "--connection-type", help="LabJack connection type."
    ),
    identifier: str = typer.Option("ANY", "--identifier", help="LabJack device identifier."),
    device_name: str = typer.Option("Dev1", "--device-name", help="NI device name (e.g. Dev1)."),
) -> None:
    """Read a single analog input channel from a LabJack or National Instruments DAQ."""
    daq: AnalogInputDAQ
    if backend == "labjack":
        daq = LabJackDAQ(
            device_type=device_type, connection_type=connection_type, identifier=identifier
        )
    elif backend == "ni":
        daq = NiDAQ(device_name)
    else:
        typer.echo(f"Unknown DAQ backend {backend!r}; expected 'labjack' or 'ni'.", err=True)
        raise typer.Exit(code=1)

    try:
        daq.connect()
        value = daq.read_channel(channel)
    except HardwareError as exc:
        typer.echo(f"Could not read from DAQ: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        daq.close()

    typer.echo(f"Channel {channel}: {value:.6f} V")


def run() -> None:
    """Entry point used by the ``glas`` console script."""
    configure_logging()
    app()


if __name__ == "__main__":
    run()

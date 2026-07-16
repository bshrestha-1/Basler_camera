"""GLAS command-line interface.

Provides subcommands for inspecting the installed version, managing
configuration files, recording experiments, browsing recorded
experiments, exporting them to common image/video formats, detecting and
tracking particles across a recording, analyzing the Brazil nut
(intruder) effect, measuring bulk convection via optical flow, and
measuring packing fraction and segregation.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path

import typer
import yaml

from glas.analysis import track_dataset
from glas.analysis.brazil_nut import DEFAULT_SETTLE_FRACTION, analyze_brazil_nut
from glas.analysis.convection import DEFAULT_GRID_SPACING, analyze_convection
from glas.analysis.packing import DEFAULT_GRID_SPACING as DEFAULT_PACKING_GRID_SPACING
from glas.analysis.packing import analyze_packing, plot_packing_summary
from glas.analysis.particle_tracking import DEFAULT_MAX_DISTANCE, DEFAULT_MAX_GAP
from glas.analysis.segregation import DEFAULT_GRID_SPACING as DEFAULT_SEGREGATION_GRID_SPACING
from glas.analysis.segregation import analyze_segregation
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA
from glas.config import deep_merge, read_yaml_file
from glas.controller import RecorderController
from glas.exceptions import (
    BrazilNutError,
    CameraConnectionError,
    CameraDriverError,
    CameraNotFoundError,
    ConfigurationError,
    ConvectionError,
    DatasetError,
    DatasetFormatError,
    DatasetIOError,
    ExperimentNotFoundError,
    ExportError,
    JSONValidationError,
    PackingError,
    SegregationError,
)
from glas.experiment import ExperimentManager
from glas.export import export_dataset
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


def run() -> None:
    """Entry point used by the ``glas`` console script."""
    configure_logging()
    app()


if __name__ == "__main__":
    run()

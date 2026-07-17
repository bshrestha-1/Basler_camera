"""Tests for glas.cli."""

from __future__ import annotations

import contextlib
import csv
import math
import socket
import threading
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from glas.cli import app
from glas.dataset import Dataset
from glas.experiment import build_experiment_extra
from glas.frame import Frame
from glas.metadata import DatasetMetadata
from glas.version import __version__

runner = CliRunner()


def _make_dataset(
    base_dir: Path,
    *,
    name: str = "",
    tags: list[str] | None = None,
    frame_count: int = 3,
    width: int = 8,
    height: int = 4,
) -> Path:
    from glas.dataset import create_experiment_folder

    folder = create_experiment_folder(base_dir)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=width,
        height=height,
        created_at_utc="2026-07-13T00:00:00+00:00",
        extra=build_experiment_extra(name=name, tags=tags),
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=np.full((height, width), (i * 10) % 256, dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=i * 1000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _make_moving_blob_dataset(
    base_dir: Path,
    *,
    frame_count: int = 5,
    width: int = 100,
    height: int = 100,
    draw_blob: bool = True,
) -> Path:
    from glas.dataset import create_experiment_folder

    folder = create_experiment_folder(base_dir)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=width,
        height=height,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        image = np.zeros((height, width), dtype=np.uint8)
        if draw_blob:
            cv2.circle(image, (20 + i * 5, height // 2), 5, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 1000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _make_rising_blob_dataset(
    base_dir: Path, *, frame_count: int = 10, width: int = 100, height: int = 200
) -> Path:
    from glas.dataset import create_experiment_folder

    folder = create_experiment_folder(base_dir)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=width,
        height=height,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        image = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(image, (width // 2, height - 20 - i * 15), 12, 255, -1)  # big, rising
        cv2.circle(image, (20, 20), 3, 255, -1)  # small, static
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 33_000_000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _make_shifting_texture_dataset(
    base_dir: Path, *, frame_count: int = 5, size: int = 200, shift_per_frame: int = 3
) -> Path:
    from glas.dataset import create_experiment_folder

    folder = create_experiment_folder(base_dir)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=size,
        height=size,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    pad = shift_per_frame * frame_count + 20
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(size + pad, size + pad), dtype=np.uint8)
    for i in range(frame_count):
        shift = i * shift_per_frame
        image = base[0:size, shift : shift + size].copy()
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 100_000_000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _make_bidisperse_dataset(base_dir: Path, *, frame_count: int = 3, size: int = 200) -> Path:
    """Two populations of particles (radius 10 and radius 4), 4 of each per frame.

    Placed two-per-grid-cell (grid_spacing=100 -> a 2x2 grid), large
    confined to the top row of cells and small to the bottom row --
    clearly segregated. See ``TestAnalyzeSegregation`` in
    ``tests/test_segregation.py`` for why two particles per cell matters.
    """
    from glas.dataset import create_experiment_folder

    folder = create_experiment_folder(base_dir)
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=size,
        height=size,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    large_centers = [(30, 30), (70, 30), (130, 30), (170, 30)]
    small_centers = [(30, 170), (70, 170), (130, 170), (170, 170)]
    for i in range(frame_count):
        image = np.zeros((size, size), dtype=np.uint8)
        for cx, cy in large_centers:
            cv2.circle(image, (cx, cy), 10, 255, -1)
        for cx, cy in small_centers:
            cv2.circle(image, (cx, cy), 4, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 100_000_000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _write_sinusoidal_accelerometer_csv(
    path: Path,
    *,
    frequency_hz: float = 60.0,
    amplitude_m: float = 1e-4,
    sample_rate_hz: float = 2000.0,
    duration_s: float = 0.5,
) -> None:
    omega = 2 * math.pi * frequency_hz
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "acceleration_g"])
        n = int(duration_s * sample_rate_hz)
        for i in range(n):
            t = i / sample_rate_hz
            acceleration_m_s2 = amplitude_m * omega**2 * math.sin(omega * t)
            writer.writerow([t, acceleration_m_s2 / 9.80665])


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_init_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    result = runner.invoke(app, ["config", "init", "--path", str(target)])
    assert result.exit_code == 0
    assert target.exists()


def test_config_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("existing: true")
    result = runner.invoke(app, ["config", "init", "--path", str(target)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_config_init_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("existing: true")
    result = runner.invoke(app, ["config", "init", "--path", str(target), "--force"])
    assert result.exit_code == 0
    assert "existing" not in target.read_text()


def test_config_validate_accepts_generated_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    runner.invoke(app, ["config", "init", "--path", str(target)])
    result = runner.invoke(app, ["config", "validate", str(target)])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_config_validate_rejects_bad_file(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text("logging:\n  level: NOT_A_LEVEL\n")
    result = runner.invoke(app, ["config", "validate", str(target)])
    assert result.exit_code == 1


def test_config_show_prints_settings(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    runner.invoke(app, ["config", "init", "--path", str(target)])
    result = runner.invoke(app, ["config", "show", "--path", str(target)])
    assert result.exit_code == 0
    assert "log_level" in result.output


class TestExperimentList:
    def test_lists_finalized_experiments(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path, name="first")
        _make_dataset(tmp_path, name="second")

        result = runner.invoke(app, ["experiment", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "Run0001" in result.output
        assert "first" in result.output
        assert "Run0002" in result.output
        assert "second" in result.output

    def test_reports_when_nothing_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["experiment", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "No experiments found" in result.output

    def test_filters_by_tag(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path, name="a", tags=["keep"])
        _make_dataset(tmp_path, name="b", tags=["drop"])

        result = runner.invoke(app, ["experiment", "list", str(tmp_path), "--tag", "keep"])
        assert result.exit_code == 0
        assert "a" in result.output
        assert "Run0002" not in result.output

    def test_filters_by_name(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path, name="Shaker Sweep")
        _make_dataset(tmp_path, name="Convection Test")

        result = runner.invoke(app, ["experiment", "list", str(tmp_path), "--name", "shaker"])
        assert result.exit_code == 0
        assert "Shaker Sweep" in result.output
        assert "Convection Test" not in result.output


class TestExperimentShow:
    def test_shows_experiment_details(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path, name="only one", tags=["x", "y"], frame_count=5)

        result = runner.invoke(app, ["experiment", "show", str(tmp_path), "Run0001"])
        assert result.exit_code == 0
        assert "name: only one" in result.output
        assert "tags: x, y" in result.output
        assert "frame_count: 5" in result.output

    def test_missing_run_id_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["experiment", "show", str(tmp_path), "Run9999"])
        assert result.exit_code == 1


class TestExport:
    def test_exports_to_png_sequence(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=4)
        output = tmp_path / "frames_out"

        result = runner.invoke(app, ["export", str(dataset_folder), str(output), "--format", "png"])
        assert result.exit_code == 0
        assert "Exported 4 frame(s)" in result.output
        assert len(list(output.glob("*.png"))) == 4

    def test_exports_to_mp4(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=3)
        output = tmp_path / "out.mp4"

        result = runner.invoke(
            app, ["export", str(dataset_folder), str(output), "--format", "mp4", "--fps", "10"]
        )
        assert result.exit_code == 0
        assert output.is_file()

    def test_rejects_unknown_format(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=1)
        output = tmp_path / "out.bmp"

        result = runner.invoke(app, ["export", str(dataset_folder), str(output), "--format", "bmp"])
        assert result.exit_code != 0

    def test_existing_destination_without_overwrite_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=1)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"not a real video")

        result = runner.invoke(app, ["export", str(dataset_folder), str(output), "--format", "mp4"])
        assert result.exit_code == 1
        assert "Export failed" in result.output

    def test_overwrite_flag_replaces_existing_destination(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=1)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"not a real video")

        result = runner.invoke(
            app, ["export", str(dataset_folder), str(output), "--format", "mp4", "--overwrite"]
        )
        assert result.exit_code == 0


class TestAnalyze:
    def test_reports_a_tracked_particle(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path)

        result = runner.invoke(app, ["analyze", str(dataset_folder)])
        assert result.exit_code == 0
        assert "Tracked 1 particle(s)" in result.output
        assert "Mean track length: 5.0 frame(s)." in result.output
        assert "Longest track: 5 frame(s)." in result.output

    def test_reports_no_particles_on_a_blank_recording(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=3, draw_blob=False)

        result = runner.invoke(app, ["analyze", str(dataset_folder)])
        assert result.exit_code == 0
        assert "No particles detected" in result.output

    def test_max_distance_flag_affects_track_count(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path)

        result = runner.invoke(app, ["analyze", str(dataset_folder), "--max-distance", "1.0"])
        assert result.exit_code == 0
        assert "Tracked 5 particle(s)" in result.output

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["analyze", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Analysis failed" in result.output


class TestBrazilNut:
    def test_reports_track_and_velocity(self, tmp_path: Path) -> None:
        dataset_folder = _make_rising_blob_dataset(tmp_path)

        result = runner.invoke(app, ["brazil-nut", str(dataset_folder)])
        assert result.exit_code == 0
        assert "Brazil nut track: 0" in result.output
        assert "Mean rise velocity:" in result.output
        assert "px/s" in result.output

    def test_reports_rise_time_when_reached(self, tmp_path: Path) -> None:
        dataset_folder = _make_rising_blob_dataset(tmp_path, frame_count=15)

        result = runner.invoke(app, ["brazil-nut", str(dataset_folder), "--settle-fraction", "0.5"])
        assert result.exit_code == 0
        assert "Rise time:" in result.output
        assert "not reached" not in result.output

    def test_reports_not_reached_when_threshold_never_hit(self, tmp_path: Path) -> None:
        dataset_folder = _make_rising_blob_dataset(tmp_path, frame_count=3)

        result = runner.invoke(
            app, ["brazil-nut", str(dataset_folder), "--settle-fraction", "0.99"]
        )
        assert result.exit_code == 0
        assert "Rise time: not reached" in result.output

    def test_plot_flag_writes_a_png(self, tmp_path: Path) -> None:
        dataset_folder = _make_rising_blob_dataset(tmp_path)
        plot_path = tmp_path / "plot.png"

        result = runner.invoke(app, ["brazil-nut", str(dataset_folder), "--plot", str(plot_path)])
        assert result.exit_code == 0
        assert f"Plot saved to {plot_path}" in result.output
        assert plot_path.is_file()

    def test_explicit_track_id_is_respected(self, tmp_path: Path) -> None:
        dataset_folder = _make_rising_blob_dataset(tmp_path)

        result = runner.invoke(app, ["brazil-nut", str(dataset_folder), "--track-id", "1"])
        assert result.exit_code == 0
        assert "Brazil nut track: 1" in result.output

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["brazil-nut", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Brazil nut analysis failed" in result.output

    def test_no_particles_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=3, draw_blob=False)

        result = runner.invoke(app, ["brazil-nut", str(dataset_folder)])
        assert result.exit_code == 1
        assert "Brazil nut analysis failed" in result.output


class TestConvection:
    def test_reports_circulation_summary(self, tmp_path: Path) -> None:
        dataset_folder = _make_shifting_texture_dataset(tmp_path)

        result = runner.invoke(app, ["convection", str(dataset_folder)])
        assert result.exit_code == 0
        assert "Analyzed 4 frame pair(s)." in result.output
        assert "Mean circulation:" in result.output
        assert "Min/max circulation:" in result.output

    def test_heatmap_dir_writes_pngs(self, tmp_path: Path) -> None:
        dataset_folder = _make_shifting_texture_dataset(tmp_path)
        heatmap_dir = tmp_path / "heatmaps"

        result = runner.invoke(
            app, ["convection", str(dataset_folder), "--heatmap-dir", str(heatmap_dir)]
        )
        assert result.exit_code == 0
        assert f"Saved 4 heat map(s) to {heatmap_dir}" in result.output
        assert len(list(heatmap_dir.glob("*.png"))) == 4

    def test_vorticity_background_is_accepted(self, tmp_path: Path) -> None:
        dataset_folder = _make_shifting_texture_dataset(tmp_path)
        heatmap_dir = tmp_path / "heatmaps"

        result = runner.invoke(
            app,
            [
                "convection",
                str(dataset_folder),
                "--heatmap-dir",
                str(heatmap_dir),
                "--heatmap-background",
                "vorticity",
            ],
        )
        assert result.exit_code == 0
        assert len(list(heatmap_dir.glob("*.png"))) == 4

    def test_grid_spacing_flag_is_respected(self, tmp_path: Path) -> None:
        dataset_folder = _make_shifting_texture_dataset(tmp_path)

        result = runner.invoke(app, ["convection", str(dataset_folder), "--grid-spacing", "50"])
        assert result.exit_code == 0

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["convection", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Convection analysis failed" in result.output

    def test_single_frame_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=1)

        result = runner.invoke(app, ["convection", str(dataset_folder)])
        assert result.exit_code == 1
        assert "Convection analysis failed" in result.output


class TestPacking:
    def test_reports_packing_fraction_summary(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path)

        result = runner.invoke(app, ["packing", str(dataset_folder)])
        assert result.exit_code == 0
        assert "Analyzed 5 frame(s)." in result.output
        assert "Mean packing fraction:" in result.output
        assert "Min/max packing fraction:" in result.output

    def test_roi_area_flag_is_respected(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=2)

        result = runner.invoke(app, ["packing", str(dataset_folder), "--roi-area", "1000"])
        assert result.exit_code == 0

    def test_field_dir_writes_pngs(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=3)
        field_dir = tmp_path / "fields"

        result = runner.invoke(app, ["packing", str(dataset_folder), "--field-dir", str(field_dir)])
        assert result.exit_code == 0
        assert f"Saved 3 packing field heat map(s) to {field_dir}" in result.output
        assert len(list(field_dir.glob("*.png"))) == 3

    def test_plot_flag_writes_a_png(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=3)
        plot_path = tmp_path / "plot.png"

        result = runner.invoke(app, ["packing", str(dataset_folder), "--plot", str(plot_path)])
        assert result.exit_code == 0
        assert f"Plot saved to {plot_path}" in result.output
        assert plot_path.is_file()

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["packing", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Packing analysis failed" in result.output

    def test_empty_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=0)

        result = runner.invoke(app, ["packing", str(dataset_folder)])
        assert result.exit_code == 1
        assert "Packing analysis failed" in result.output


class TestSegregation:
    def test_reports_segregation_summary(self, tmp_path: Path) -> None:
        dataset_folder = _make_bidisperse_dataset(tmp_path)

        result = runner.invoke(app, ["segregation", str(dataset_folder), "--grid-spacing", "100"])
        assert result.exit_code == 0
        assert "Analyzed 3 frame(s)." in result.output
        assert "Mean segregation index:" in result.output
        assert "Mean mixing index:" in result.output
        assert "Mean mixing entropy:" in result.output

    def test_explicit_size_threshold_is_respected(self, tmp_path: Path) -> None:
        dataset_folder = _make_bidisperse_dataset(tmp_path, frame_count=2)

        result = runner.invoke(
            app,
            [
                "segregation",
                str(dataset_folder),
                "--grid-spacing",
                "100",
                "--size-threshold",
                "7",
            ],
        )
        assert result.exit_code == 0

    def test_plot_flag_writes_a_png(self, tmp_path: Path) -> None:
        dataset_folder = _make_bidisperse_dataset(tmp_path, frame_count=2)
        plot_path = tmp_path / "plot.png"

        result = runner.invoke(
            app,
            [
                "segregation",
                str(dataset_folder),
                "--grid-spacing",
                "100",
                "--plot",
                str(plot_path),
            ],
        )
        assert result.exit_code == 0
        assert f"Plot saved to {plot_path}" in result.output
        assert plot_path.is_file()

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["segregation", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Segregation analysis failed" in result.output

    def test_empty_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, frame_count=0)

        result = runner.invoke(app, ["segregation", str(dataset_folder)])
        assert result.exit_code == 1
        assert "Segregation analysis failed" in result.output

    def test_default_grid_spacing_too_fine_fails_cleanly(self, tmp_path: Path) -> None:
        # The default grid_spacing (32) produces mostly single-particle
        # cells for this small, sparse dataset -- exercises the
        # SegregationError path through the CLI, not just ValueError.
        dataset_folder = _make_bidisperse_dataset(tmp_path, frame_count=1)

        result = runner.invoke(app, ["segregation", str(dataset_folder)])
        assert result.exit_code == 1
        assert "Segregation analysis failed" in result.output


class TestAccelerometerAnalyze:
    def test_reports_frequency_amplitude_and_gamma(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        _write_sinusoidal_accelerometer_csv(csv_path, frequency_hz=60.0, amplitude_m=1e-4)

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "analyze",
                str(csv_path),
                "--value-column",
                "acceleration_g",
                "--value-units",
                "g",
            ],
        )
        assert result.exit_code == 0
        assert "Frequency: 60.0" in result.output
        assert "Amplitude:" in result.output
        assert "Gamma:" in result.output
        assert "Peak acceleration:" in result.output

    def test_plot_flag_writes_a_png(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        _write_sinusoidal_accelerometer_csv(csv_path)
        plot_path = tmp_path / "plot.png"

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "analyze",
                str(csv_path),
                "--value-column",
                "acceleration_g",
                "--value-units",
                "g",
                "--plot",
                str(plot_path),
            ],
        )
        assert result.exit_code == 0
        assert f"Plot saved to {plot_path}" in result.output
        assert plot_path.is_file()

    def test_volts_conversion_via_sensitivity(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "voltage_v"])
            for i in range(8):
                writer.writerow([i * 0.001, 0.01 if i % 2 == 0 else -0.01])

        result = runner.invoke(
            app,
            ["accelerometer", "analyze", str(csv_path), "--sensitivity-mv-per-g", "10"],
        )
        assert result.exit_code == 0

    def test_missing_file_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["accelerometer", "analyze", str(tmp_path / "does_not_exist.csv")]
        )
        assert result.exit_code == 1
        assert "Accelerometer analysis failed" in result.output


class TestAccelerometerSync:
    def test_writes_a_synchronized_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        _write_sinusoidal_accelerometer_csv(csv_path)
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=3)
        output = tmp_path / "synced.csv"

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "sync",
                str(csv_path),
                str(dataset_folder),
                "--value-column",
                "acceleration_g",
                "--value-units",
                "g",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert "Synchronized 3 frame(s)" in result.output
        assert output.is_file()

        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert rows[0]["frame_id"] == "0"

    def test_offset_flag_is_accepted(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        _write_sinusoidal_accelerometer_csv(csv_path)
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=2)
        output = tmp_path / "synced.csv"

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "sync",
                str(csv_path),
                str(dataset_folder),
                "--value-column",
                "acceleration_g",
                "--value-units",
                "g",
                "--output",
                str(output),
                "--offset-s",
                "0.01",
            ],
        )
        assert result.exit_code == 0

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "accel.csv"
        _write_sinusoidal_accelerometer_csv(csv_path)

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "sync",
                str(csv_path),
                str(tmp_path / "does_not_exist"),
                "--value-column",
                "acceleration_g",
                "--value-units",
                "g",
                "--output",
                str(tmp_path / "synced.csv"),
            ],
        )
        assert result.exit_code == 1
        assert "Accelerometer synchronization failed" in result.output

    def test_missing_accelerometer_file_fails_cleanly(self, tmp_path: Path) -> None:
        dataset_folder = _make_moving_blob_dataset(tmp_path, frame_count=2)

        result = runner.invoke(
            app,
            [
                "accelerometer",
                "sync",
                str(tmp_path / "does_not_exist.csv"),
                str(dataset_folder),
                "--output",
                str(tmp_path / "synced.csv"),
            ],
        )
        assert result.exit_code == 1
        assert "Accelerometer synchronization failed" in result.output


def _start_stub_scpi_server(
    response: bytes | None = None,
) -> tuple[str, int, list[bytes], socket.socket, threading.Thread]:
    """Start a background TCP server that records what it receives.

    If ``response`` is given, it's sent to the client immediately on
    accept (queued in the socket's send buffer, so the client can read it
    whenever it queries -- no second thread needed to time a reply, same
    approach as ``tests/test_scpi.py``).
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5.0)  # bound accept() too, so a client that never
    # connects (e.g. a CLI invocation that fails argument parsing before
    # opening a socket) can't hang this thread -- and the process -- forever.
    host, port = server.getsockname()
    received: list[bytes] = []

    def handle() -> None:
        try:
            connection, _ = server.accept()
        except OSError:
            return
        connection.settimeout(2.0)
        if response is not None:
            connection.sendall(response)
        with contextlib.suppress(OSError):
            received.append(connection.recv(4096))
        connection.close()

    thread = threading.Thread(target=handle, daemon=True)
    thread.start()
    return host, port, received, server, thread


class TestWaveformGenSine:
    def test_sends_expected_scpi_command(self) -> None:
        host, port, received, server, thread = _start_stub_scpi_server()
        try:
            result = runner.invoke(
                app,
                [
                    "waveform-gen",
                    "sine",
                    host,
                    "--port",
                    str(port),
                    "--frequency-hz",
                    "100",
                    "--amplitude-vpp",
                    "2.0",
                ],
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 0
        assert "100.0 Hz" in result.output
        assert received == [b"C1:BSWV WVTP,SINE,FRQ,100.0HZ,AMP,2.0V,OFST,0.0V,PHSE,0.0\n"]

    def test_enable_flag_also_turns_output_on(self) -> None:
        host, port, received, server, thread = _start_stub_scpi_server()
        try:
            result = runner.invoke(
                app,
                [
                    "waveform-gen",
                    "sine",
                    host,
                    "--port",
                    str(port),
                    "--frequency-hz",
                    "50",
                    "--amplitude-vpp",
                    "1.0",
                    "--enable",
                ],
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 0
        assert "output enabled" in result.output

    def test_connection_failure_fails_cleanly(self) -> None:
        result = runner.invoke(
            app,
            [
                "waveform-gen",
                "sine",
                "127.0.0.1",
                "--port",
                "1",
                "--frequency-hz",
                "100",
                "--amplitude-vpp",
                "2.0",
            ],
        )
        assert result.exit_code == 1
        assert "Could not connect to waveform generator" in result.output

    def test_invalid_channel_fails_cleanly(self) -> None:
        host, port, _received, server, thread = _start_stub_scpi_server()
        try:
            result = runner.invoke(
                app,
                [
                    "waveform-gen",
                    "sine",
                    host,
                    "--port",
                    str(port),
                    "--frequency-hz",
                    "100",
                    "--amplitude-vpp",
                    "2.0",
                    "--channel",
                    "5",
                ],
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 1
        assert "Could not configure waveform generator" in result.output


class TestOscilloscopeQuery:
    def test_prints_response(self) -> None:
        host, port, received, server, thread = _start_stub_scpi_server(
            response=b"FAKE,SCOPE,SN,1.0\n"
        )
        try:
            result = runner.invoke(
                app, ["oscilloscope", "query", host, "*IDN?", "--port", str(port)]
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 0
        assert result.output.strip() == "FAKE,SCOPE,SN,1.0"
        assert received == [b"*IDN?\n"]

    def test_connection_failure_fails_cleanly(self) -> None:
        result = runner.invoke(app, ["oscilloscope", "query", "127.0.0.1", "*IDN?", "--port", "1"])
        assert result.exit_code == 1
        assert "Could not connect to oscilloscope" in result.output


class TestShakerSetGamma:
    def test_sends_computed_drive_voltage(self) -> None:
        host, port, received, server, thread = _start_stub_scpi_server()
        try:
            result = runner.invoke(
                app,
                [
                    "shaker",
                    "set-gamma",
                    host,
                    "2.0",
                    "--port",
                    str(port),
                    "--volts-per-g",
                    "0.5",
                    "--calibration-frequency-hz",
                    "60",
                ],
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 0
        assert "1.0000 Vpp" in result.output
        assert received == [b"C1:BSWV WVTP,SINE,FRQ,60.0HZ,AMP,1.0V,OFST,0.0V,PHSE,0.0\n"]

    def test_connection_failure_fails_cleanly(self) -> None:
        result = runner.invoke(
            app,
            [
                "shaker",
                "set-gamma",
                "127.0.0.1",
                "2.0",
                "--port",
                "1",
                "--volts-per-g",
                "0.5",
                "--calibration-frequency-hz",
                "60",
            ],
        )
        assert result.exit_code == 1
        assert "Could not connect to waveform generator" in result.output

    def test_rejects_non_positive_gamma(self) -> None:
        host, port, _received, server, thread = _start_stub_scpi_server()
        try:
            # "0" rather than a negative number: a leading "-" gets
            # misparsed by Click as an option flag rather than the
            # positional GAMMA argument, which would fail argument
            # parsing before the command ever opens a connection.
            result = runner.invoke(
                app,
                [
                    "shaker",
                    "set-gamma",
                    host,
                    "0",
                    "--port",
                    str(port),
                    "--volts-per-g",
                    "0.5",
                    "--calibration-frequency-hz",
                    "60",
                ],
            )
        finally:
            thread.join(timeout=3.0)
            server.close()

        assert result.exit_code == 1
        assert "Could not set target Gamma" in result.output


class TestDaqRead:
    def test_labjack_backend_missing_sdk_fails_cleanly(self) -> None:
        result = runner.invoke(app, ["daq", "read", "labjack", "--channel", "0"])
        assert result.exit_code == 1
        assert "Could not read from DAQ" in result.output
        assert "labjack-ljm" in result.output

    def test_ni_backend_missing_sdk_fails_cleanly(self) -> None:
        result = runner.invoke(
            app, ["daq", "read", "ni", "--device-name", "Dev1", "--channel", "0"]
        )
        assert result.exit_code == 1
        assert "Could not read from DAQ" in result.output
        assert "nidaqmx" in result.output

    def test_unknown_backend_fails_cleanly(self) -> None:
        result = runner.invoke(app, ["daq", "read", "bogus"])
        assert result.exit_code == 1
        assert "Unknown DAQ backend" in result.output

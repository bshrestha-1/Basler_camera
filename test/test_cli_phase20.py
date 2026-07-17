"""Tests for the Phase 20 CLI commands: doctor, qa, report, compare, calibrate."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from glas.cli import app
from glas.dataset import Dataset, create_experiment_folder
from glas.experiment import PhysicalParameters, build_physical_parameters_extra
from glas.frame import Frame
from glas.metadata import DatasetMetadata

runner = CliRunner()


def _make_rising_blob_dataset(
    base_dir: Path, *, frame_count: int = 10, width: int = 100, height: int = 200
) -> Path:
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
        cv2.circle(image, (width // 2, height - 20 - i * 15), 12, 255, -1)
        cv2.circle(image, (20, 20), 3, 255, -1)
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


def _make_checkerboard_image(
    path: Path, *, squares: tuple[int, int] = (8, 8), square_size_px: int = 40, margin_px: int = 40
) -> None:
    cols, rows = squares
    board_w = cols * square_size_px
    board_h = rows * square_size_px
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[
                    r * square_size_px : (r + 1) * square_size_px,
                    c * square_size_px : (c + 1) * square_size_px,
                ] = 255
    image = np.full((board_h + 2 * margin_px, board_w + 2 * margin_px), 255, dtype=np.uint8)
    image[margin_px : margin_px + board_h, margin_px : margin_px + board_w] = board
    cv2.imwrite(str(path), image)


class TestCalibrateTwoPoint:
    def test_computes_and_writes_calibration(self, tmp_path: Path) -> None:
        output_path = tmp_path / "calibration.json"
        result = runner.invoke(
            app,
            ["calibrate", "two-point", "0", "0", "100", "0", "50.0", "--output", str(output_path)],
        )
        assert result.exit_code == 0
        assert "mm_per_pixel = 0.500000" in result.output
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["mm_per_pixel"] == 0.5

    def test_invalid_distance_fails_cleanly(self, tmp_path: Path) -> None:
        output_path = tmp_path / "calibration.json"
        result = runner.invoke(
            app,
            ["calibrate", "two-point", "0", "0", "100", "0", "0", "--output", str(output_path)],
        )
        assert result.exit_code == 1
        assert "Calibration failed" in result.output


class TestCalibrateCheckerboard:
    def test_computes_and_writes_calibration(self, tmp_path: Path) -> None:
        image_path = tmp_path / "checkerboard.png"
        _make_checkerboard_image(image_path)
        output_path = tmp_path / "calibration.json"
        result = runner.invoke(
            app,
            [
                "calibrate",
                "checkerboard",
                str(image_path),
                "7",
                "7",
                "25.0",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["mm_per_pixel"] == pytest.approx(25.0 / 40.0, rel=1e-3)

    def test_missing_image_fails_cleanly(self, tmp_path: Path) -> None:
        output_path = tmp_path / "calibration.json"
        result = runner.invoke(
            app,
            [
                "calibrate",
                "checkerboard",
                str(tmp_path / "does_not_exist.png"),
                "7",
                "7",
                "25.0",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 1
        assert "Could not read image" in result.output

    def test_pattern_not_found_fails_cleanly(self, tmp_path: Path) -> None:
        image_path = tmp_path / "blank.png"
        cv2.imwrite(str(image_path), np.full((200, 200), 128, dtype=np.uint8))
        output_path = tmp_path / "calibration.json"
        result = runner.invoke(
            app,
            [
                "calibrate",
                "checkerboard",
                str(image_path),
                "7",
                "7",
                "25.0",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 1
        assert "Calibration failed" in result.output


class TestDoctor:
    def test_reports_disk_space_and_no_camera(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path)])
        assert "disk_space" in result.output

    def test_min_disk_free_gb_absurdly_high_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path), "--min-disk-free-gb", "1000000000"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestQa:
    def test_clean_recording_reports_no_issues(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        result = runner.invoke(app, ["qa", str(folder)])
        assert result.exit_code == 0
        assert "No issues found." in result.output

    def test_expected_fps_mismatch_reports_warning(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        result = runner.invoke(app, ["qa", str(folder), "--expected-fps", "1000"])
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_strict_flag_exits_nonzero_on_warning(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        result = runner.invoke(app, ["qa", str(folder), "--expected-fps", "1000", "--strict"])
        assert result.exit_code == 1

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["qa", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "Quality assessment failed" in result.output


class TestReport:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        result = runner.invoke(app, ["report", str(folder), str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()
        assert "Report saved to" in result.output

    def test_missing_dataset_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["report", str(tmp_path / "does_not_exist"), str(tmp_path / "out.html")]
        )
        assert result.exit_code == 1
        assert "Report generation failed" in result.output


class TestCompare:
    def _make_multi_run_dataset(self, base_dir: Path) -> None:
        for gamma, blob_count in [(1.0, 5), (1.0, 6), (2.0, 7), (2.0, 8), (3.0, 9)]:
            folder = create_experiment_folder(base_dir)
            extra = build_physical_parameters_extra(PhysicalParameters(target_acceleration_g=gamma))
            metadata = DatasetMetadata(
                dataset_format="hdf5",
                camera_model="acA640-750um",
                camera_serial="12345678",
                pixel_format="Mono8",
                width=100,
                height=100,
                created_at_utc="2026-07-13T00:00:00+00:00",
                extra=extra,
            )
            dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
            for i in range(5):
                image = np.zeros((100, 100), dtype=np.uint8)
                for j in range(blob_count):
                    cv2.circle(image, (10 + j * 8, 50), 3, 255, -1)
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

    def test_reports_points_and_writes_outputs(self, tmp_path: Path) -> None:
        self._make_multi_run_dataset(tmp_path)
        plot_path = tmp_path / "sweep.png"
        csv_path = tmp_path / "sweep.csv"
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_path),
                "--parameter",
                "target-acceleration-g",
                "--metric",
                "packing-fraction",
                "--plot",
                str(plot_path),
                "--csv",
                str(csv_path),
            ],
        )
        assert result.exit_code == 0
        assert "target-acceleration-g=1.0" in result.output
        assert plot_path.exists()
        assert csv_path.exists()

    def test_unknown_parameter_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_path),
                "--parameter",
                "bogus-parameter",
                "--metric",
                "packing-fraction",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown --parameter" in result.output

    def test_unknown_metric_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_path),
                "--parameter",
                "target-acceleration-g",
                "--metric",
                "bogus-metric",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown --metric" in result.output

    def test_no_matching_runs_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_path),
                "--parameter",
                "target-acceleration-g",
                "--metric",
                "packing-fraction",
            ],
        )
        assert result.exit_code == 1
        assert "Comparison failed" in result.output

    def test_no_fit_flag_omits_fit_line(self, tmp_path: Path) -> None:
        self._make_multi_run_dataset(tmp_path)
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_path),
                "--parameter",
                "target-acceleration-g",
                "--metric",
                "packing-fraction",
                "--no-fit",
            ],
        )
        assert result.exit_code == 0
        assert "Linear fit" not in result.output

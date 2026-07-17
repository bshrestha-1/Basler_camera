"""Tests for glas.report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from glas.dataset import Dataset
from glas.exceptions import ReportError
from glas.experiment import (
    NAME_KEY,
    PhysicalParameters,
    build_experiment_extra,
    build_physical_parameters_extra,
)
from glas.frame import Frame
from glas.metadata import DatasetMetadata
from glas.report import generate_report


def _make_rising_blob_dataset(
    tmp_path: Path,
    *,
    frame_count: int = 10,
    width: int = 100,
    height: int = 200,
    extra: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=width,
        height=height,
        created_at_utc="2026-07-13T00:00:00+00:00",
        notes=notes,
        extra=extra or {},
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


def _make_blank_single_frame_dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=50,
        height=50,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    dataset.append_frame(
        Frame(
            frame_id=0,
            image=np.zeros((50, 50), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
    )
    dataset.finalize()
    return folder


class TestGenerateReport:
    def test_returns_output_path(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        assert generate_report(folder, output_path) == output_path

    def test_writes_a_file(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        assert output_path.exists()

    def test_includes_every_analysis_section(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        for section in ["Tracking", "Brazil Nut", "Convection", "Packing", "Segregation"]:
            assert f"<h2>{section}</h2>" in text

    def test_embeds_images_as_base64(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "data:image/png;base64," in text

    def test_is_valid_html_document(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert text.startswith("<!doctype html>")
        assert "<html>" in text
        assert "</html>" in text

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2)
        output_path = tmp_path / "nested" / "dir" / "report.html"
        generate_report(folder, output_path)
        assert output_path.exists()

    def test_escapes_html_special_characters_in_notes(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(
            tmp_path, frame_count=2, notes="<script>alert(1)</script>"
        )
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text

    def test_includes_experiment_name_in_title(self, tmp_path: Path) -> None:
        extra = build_experiment_extra(name="shaker sweep")
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2, extra=extra)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "shaker sweep" in text
        assert extra[NAME_KEY] == "shaker sweep"

    def test_includes_physical_parameters(self, tmp_path: Path) -> None:
        extra = build_physical_parameters_extra(
            PhysicalParameters(material="glass beads", target_acceleration_g=2.0)
        )
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2, extra=extra)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "glass beads" in text

    def test_metadata_table_includes_camera_info(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "acA640-750um" in text
        assert "12345678" in text

    def test_metadata_table_includes_reproducibility_fields(self, tmp_path: Path) -> None:
        """The report is the publishable artifact -- it must show enough
        for a reader to reproduce how the recording was captured, not just
        exposure/gain."""
        folder = tmp_path / "dataset"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=100,
            height=200,
            created_at_utc="2026-07-13T00:00:00+00:00",
            exposure_time_us=5000.0,
            gain_db=6.0,
            frame_rate_hz=100.0,
            roi_offset_x=32,
            roi_offset_y=16,
            camera_settings={"gamma": 1.0, "reverse_x": False},
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        for i in range(2):
            image = np.zeros((200, 100), dtype=np.uint8)
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

        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "100.00 Hz" in text
        assert "(32, 16)" in text
        assert "gamma=1.0" in text
        assert "reverse_x=False" in text

    def test_partial_failure_shows_skipped_sections_not_crash(self, tmp_path: Path) -> None:
        folder = _make_blank_single_frame_dataset(tmp_path)
        output_path = tmp_path / "report.html"
        result = generate_report(folder, output_path)
        assert result == output_path
        text = output_path.read_text()
        assert "Skipped:" in text

    def test_missing_metadata_raises_report_error(self, tmp_path: Path) -> None:
        missing_folder = tmp_path / "does_not_exist"
        with pytest.raises(ReportError):
            generate_report(missing_folder, tmp_path / "report.html")

    def test_respects_min_area_parameter(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path, frame_count=3)
        output_path = tmp_path / "report.html"
        # An absurdly large min_area filters out every particle -- tracking
        # should report "no particles" rather than crash.
        generate_report(folder, output_path, min_area=1_000_000)
        text = output_path.read_text()
        assert "No particles detected" in text

    def test_vibration_section_absent_without_accelerometer_csv(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2)
        output_path = tmp_path / "report.html"
        generate_report(folder, output_path)
        text = output_path.read_text()
        assert "<h2>Vibration</h2>" not in text

    def test_vibration_section_present_with_accelerometer_csv(self, tmp_path: Path) -> None:
        folder = _make_rising_blob_dataset(tmp_path, frame_count=2)
        csv_path = tmp_path / "accel.csv"
        rows = ["time_s,voltage_v"]
        for i in range(200):
            t = i / 1000.0
            rows.append(f"{t},{0.5 * np.sin(2 * np.pi * 60 * t)}")
        csv_path.write_text("\n".join(rows))

        output_path = tmp_path / "report.html"
        generate_report(folder, output_path, accelerometer_csv=csv_path)
        text = output_path.read_text()
        assert "<h2>Vibration</h2>" in text

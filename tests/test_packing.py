"""Tests for glas.analysis.packing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from glas.analysis.packing import (
    PackingField,
    PackingMetrics,
    PackingSummary,
    analyze_packing,
    compute_packing_field,
    compute_packing_metrics,
    plot_packing_heatmap,
    plot_packing_summary,
)
from glas.analysis.tracking_utils import Detection
from glas.dataset import Dataset
from glas.exceptions import PackingError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _detection(x: float, y: float, area: float) -> Detection:
    return Detection(x=x, y=y, radius=(area / 3.141592653589793) ** 0.5, area=area)


class TestComputePackingMetrics:
    def test_computes_exact_packing_fraction(self) -> None:
        detections = [_detection(10, 10, 12.0), _detection(50, 50, 28.0)]
        metrics = compute_packing_metrics(detections, roi_area=100.0, frame_id=5)

        assert metrics.frame_id == 5
        assert metrics.particle_count == 2
        assert metrics.packing_fraction == pytest.approx(0.4)
        assert metrics.void_fraction == pytest.approx(0.6)
        assert metrics.number_density == pytest.approx(0.02)

    def test_no_detections_yields_zero_fraction_and_full_void(self) -> None:
        metrics = compute_packing_metrics([], roi_area=100.0)
        assert metrics.particle_count == 0
        assert metrics.packing_fraction == 0.0
        assert metrics.void_fraction == 1.0
        assert metrics.number_density == 0.0

    def test_does_not_clamp_packing_fraction_above_one(self) -> None:
        # Overlapping/merged blob detections can legitimately sum to more
        # than the ROI area -- this is a valid computed result, not an
        # error condition.
        detections = [_detection(10, 10, 80.0), _detection(12, 12, 60.0)]
        metrics = compute_packing_metrics(detections, roi_area=100.0)
        assert metrics.packing_fraction == pytest.approx(1.4)
        assert metrics.void_fraction == pytest.approx(-0.4)

    def test_rejects_non_positive_roi_area(self) -> None:
        with pytest.raises(ValueError):
            compute_packing_metrics([], roi_area=0.0)
        with pytest.raises(ValueError):
            compute_packing_metrics([], roi_area=-5.0)

    def test_is_frozen(self) -> None:
        metrics = compute_packing_metrics([], roi_area=1.0)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            metrics.particle_count = 5  # type: ignore[misc]


class TestComputePackingField:
    def test_bins_detections_into_correct_grid_cells(self) -> None:
        # 64x64 image, grid_spacing=32 -> 2x2 grid, each interior cell is
        # exactly 32*32 = 1024 px^2.
        detections = [_detection(10, 10, 16.0), _detection(50, 50, 64.0)]
        field = compute_packing_field(
            detections, image_width=64, image_height=64, grid_spacing=32, frame_id=1
        )

        assert isinstance(field, PackingField)
        assert field.frame_id == 1
        assert field.packing_fraction.shape == (2, 2)
        assert field.packing_fraction[0, 0] == pytest.approx(16.0 / 1024)
        assert field.packing_fraction[0, 1] == pytest.approx(0.0)
        assert field.packing_fraction[1, 0] == pytest.approx(0.0)
        assert field.packing_fraction[1, 1] == pytest.approx(64.0 / 1024)

    def test_multiple_detections_in_same_cell_are_summed(self) -> None:
        detections = [_detection(5, 5, 10.0), _detection(8, 8, 6.0)]
        field = compute_packing_field(detections, image_width=32, image_height=32, grid_spacing=32)
        assert field.packing_fraction[0, 0] == pytest.approx(16.0 / (32 * 32))

    def test_boundary_cells_are_clipped_to_actual_coverage(self) -> None:
        # 50x50 image, grid_spacing=32 -> 2x2 grid; last row/col are only
        # 18px wide/tall (50 - 32 = 18), not the full 32.
        detections = [_detection(45, 45, 18.0)]
        field = compute_packing_field(detections, image_width=50, image_height=50, grid_spacing=32)
        assert field.packing_fraction[1, 1] == pytest.approx(18.0 / (18 * 18))

    def test_grid_coordinates_are_cell_centers(self) -> None:
        field = compute_packing_field([], image_width=64, image_height=64, grid_spacing=32)
        assert field.x[0, 0] == pytest.approx(16.0)
        assert field.x[0, 1] == pytest.approx(48.0)
        assert field.y[0, 0] == pytest.approx(16.0)
        assert field.y[1, 0] == pytest.approx(48.0)

    def test_out_of_bounds_centroid_is_clamped_not_wrapped(self) -> None:
        # Detection.x/y are unconstrained floats; a negative or
        # over-large centroid must clamp into the grid, not silently
        # wrap around via numpy's negative indexing.
        detections = [_detection(-5.0, -5.0, 10.0), _detection(1000.0, 1000.0, 20.0)]
        field = compute_packing_field(detections, image_width=64, image_height=64, grid_spacing=32)
        assert field.packing_fraction[0, 0] == pytest.approx(10.0 / 1024)
        assert field.packing_fraction[1, 1] == pytest.approx(20.0 / 1024)

    def test_rejects_non_positive_grid_spacing(self) -> None:
        with pytest.raises(ValueError):
            compute_packing_field([], image_width=10, image_height=10, grid_spacing=0)

    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError):
            compute_packing_field([], image_width=0, image_height=10, grid_spacing=1)
        with pytest.raises(ValueError):
            compute_packing_field([], image_width=10, image_height=0, grid_spacing=1)


class TestPlotPackingHeatmap:
    def test_produces_a_valid_png(self, tmp_path: Path) -> None:
        field = compute_packing_field(
            [_detection(10, 10, 16.0)], image_width=64, image_height=64, grid_spacing=32
        )
        output = tmp_path / "heatmap.png"

        result = plot_packing_heatmap(field, output)

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        field = compute_packing_field([], image_width=32, image_height=32, grid_spacing=32)
        output = tmp_path / "nested" / "dir" / "heatmap.png"

        plot_packing_heatmap(field, output)
        assert output.is_file()


class TestPlotPackingSummary:
    def test_produces_a_valid_png(self, tmp_path: Path) -> None:
        summary = PackingSummary(
            frame_ids=[0, 1, 2],
            times_s=[0.0, 0.1, 0.2],
            metrics=[
                PackingMetrics(
                    frame_id=i,
                    particle_count=i + 1,
                    packing_fraction=0.1 * i,
                    void_fraction=1 - 0.1 * i,
                    number_density=0.01 * (i + 1),
                )
                for i in range(3)
            ],
        )
        output = tmp_path / "summary.png"

        result = plot_packing_summary(summary, output)

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"


class TestAnalyzePacking:
    def _make_dataset(
        self, tmp_path: Path, *, size: int = 200, frame_count: int = 3, radius: int = 8
    ) -> tuple[Path, float]:
        folder = tmp_path / "dataset"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=size,
            height=size,
            created_at_utc="2026-07-16T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")

        # Two well-separated circles of known radius per frame.
        centers = [(50, 50), (150, 150)]
        particle_area = 0.0
        for i in range(frame_count):
            image = np.zeros((size, size), dtype=np.uint8)
            for cx, cy in centers:
                cv2.circle(image, (cx, cy), radius, 255, -1)
            if i == 0:
                # cv2.contourArea on a filled circle differs slightly from
                # pi*r^2; measure it directly from a detection instead of
                # assuming the analytic formula.
                from glas.analysis.tracking_utils import detect_particles

                measured = detect_particles(image)
                particle_area = sum(d.area for d in measured)
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
        return folder, particle_area

    def test_computes_expected_packing_fraction(self, tmp_path: Path) -> None:
        size = 200
        folder, total_particle_area = self._make_dataset(tmp_path, size=size, frame_count=2)

        summary = analyze_packing(folder)

        assert isinstance(summary, PackingSummary)
        assert summary.frame_ids == [0, 1]
        assert len(summary.metrics) == 2
        expected_fraction = total_particle_area / (size * size)
        for metrics in summary.metrics:
            assert metrics.particle_count == 2
            assert metrics.packing_fraction == pytest.approx(expected_fraction, rel=0.05)

    def test_roi_area_override_changes_fraction(self, tmp_path: Path) -> None:
        folder, _ = self._make_dataset(tmp_path, frame_count=1)

        default_summary = analyze_packing(folder)
        overridden_summary = analyze_packing(folder, roi_area=1000.0)

        assert overridden_summary.metrics[0].packing_fraction == pytest.approx(
            default_summary.metrics[0].packing_fraction * (200 * 200) / 1000.0
        )

    def test_times_s_reflect_real_elapsed_time(self, tmp_path: Path) -> None:
        folder, _ = self._make_dataset(tmp_path, frame_count=3)
        summary = analyze_packing(folder)
        assert summary.times_s == pytest.approx([0.0, 0.1, 0.2])

    def test_optionally_writes_field_heatmaps(self, tmp_path: Path) -> None:
        folder, _ = self._make_dataset(tmp_path, frame_count=3)
        field_dir = tmp_path / "fields"

        analyze_packing(folder, field_grid_spacing=32, field_dir=field_dir)

        pngs = sorted(field_dir.glob("*.png"))
        assert len(pngs) == 3

    def test_no_field_heatmaps_written_when_omitted(self, tmp_path: Path) -> None:
        folder, _ = self._make_dataset(tmp_path, frame_count=2)
        analyze_packing(folder)
        assert not (tmp_path / "fields").exists()

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "empty"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=10,
            height=10,
            created_at_utc="2026-07-16T00:00:00+00:00",
        )
        Dataset.create(folder, metadata, dataset_format="hdf5").finalize()

        with pytest.raises(PackingError):
            analyze_packing(folder)

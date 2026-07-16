"""Tests for glas.analysis.segregation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from glas.analysis.segregation import (
    SegregationMetrics,
    SegregationSummary,
    analyze_segregation,
    compute_segregation_metrics,
    plot_segregation_summary,
)
from glas.analysis.tracking_utils import Detection
from glas.dataset import Dataset
from glas.exceptions import SegregationError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _det(x: float, y: float, radius: float = 1.0) -> Detection:
    return Detection(x=x, y=y, radius=radius, area=math.pi * radius**2)


class TestComputeSegregationMetrics:
    def test_fully_segregated_grid_gives_zero_mixing(self) -> None:
        # 64x64 image, grid_spacing=32 -> 2x2 grid. Top row cells are all
        # large, bottom row cells are all small: perfectly segregated.
        large = [_det(5, 5), _det(6, 6), _det(40, 5), _det(41, 6)]
        small = [_det(5, 40), _det(6, 41), _det(40, 40), _det(41, 41)]

        metrics = compute_segregation_metrics(
            large, small, image_width=64, image_height=64, grid_spacing=32, frame_id=7
        )

        assert isinstance(metrics, SegregationMetrics)
        assert metrics.frame_id == 7
        assert metrics.large_count == 4
        assert metrics.small_count == 4
        assert metrics.mixing_index == pytest.approx(0.0, abs=1e-9)
        assert metrics.segregation_index == pytest.approx(1.0, abs=1e-9)

    def test_fully_mixed_grid_gives_full_mixing_and_entropy(self) -> None:
        # Every cell has exactly 1 large + 1 small -> local concentration
        # matches the global concentration (0.5) everywhere.
        large = [_det(5, 5), _det(40, 5), _det(5, 40), _det(40, 40)]
        small = [_det(6, 6), _det(41, 6), _det(6, 41), _det(41, 41)]

        metrics = compute_segregation_metrics(
            large, small, image_width=64, image_height=64, grid_spacing=32
        )

        assert metrics.mixing_index == pytest.approx(1.0, abs=1e-9)
        assert metrics.segregation_index == pytest.approx(0.0, abs=1e-9)
        assert metrics.mixing_entropy == pytest.approx(1.0, abs=1e-6)

    def test_indices_are_complementary(self) -> None:
        large = [_det(5, 5), _det(6, 6), _det(40, 5)]
        small = [_det(5, 40), _det(40, 40), _det(41, 41)]
        metrics = compute_segregation_metrics(
            large, small, image_width=64, image_height=64, grid_spacing=32
        )
        assert metrics.segregation_index == pytest.approx(1.0 - metrics.mixing_index)

    def test_only_large_population_present(self) -> None:
        large = [_det(5, 5), _det(6, 6)]
        metrics = compute_segregation_metrics(
            large, [], image_width=64, image_height=64, grid_spacing=32
        )
        assert metrics.large_count == 2
        assert metrics.small_count == 0
        assert metrics.segregation_index == 0.0
        assert metrics.mixing_index == 1.0
        assert metrics.mixing_entropy == 0.0

    def test_only_small_population_present(self) -> None:
        small = [_det(5, 5), _det(6, 6)]
        metrics = compute_segregation_metrics(
            [], small, image_width=64, image_height=64, grid_spacing=32
        )
        assert metrics.large_count == 0
        assert metrics.small_count == 2
        assert metrics.segregation_index == 0.0
        assert metrics.mixing_index == 1.0
        assert metrics.mixing_entropy == 0.0

    def test_no_particles_at_all_raises(self) -> None:
        with pytest.raises(SegregationError):
            compute_segregation_metrics([], [], image_width=64, image_height=64, grid_spacing=32)

    def test_single_particle_per_occupied_cell_raises(self) -> None:
        # Every occupied cell has exactly 1 particle -- segregation and
        # randomness are statistically indistinguishable at this
        # resolution.
        large = [_det(5, 5)]
        small = [_det(40, 40)]
        with pytest.raises(SegregationError):
            compute_segregation_metrics(
                large, small, image_width=64, image_height=64, grid_spacing=32
            )

    def test_rejects_non_positive_grid_spacing(self) -> None:
        with pytest.raises(ValueError):
            compute_segregation_metrics(
                [_det(1, 1)], [_det(2, 2)], image_width=10, image_height=10, grid_spacing=0
            )

    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError):
            compute_segregation_metrics(
                [_det(1, 1)], [_det(2, 2)], image_width=0, image_height=10, grid_spacing=1
            )
        with pytest.raises(ValueError):
            compute_segregation_metrics(
                [_det(1, 1)], [_det(2, 2)], image_width=10, image_height=0, grid_spacing=1
            )

    def test_is_frozen(self) -> None:
        large = [_det(5, 5), _det(6, 6)]
        small = [_det(5, 40), _det(40, 40)]
        metrics = compute_segregation_metrics(
            large, small, image_width=64, image_height=64, grid_spacing=32
        )
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            metrics.large_count = 99  # type: ignore[misc]

    def test_no_runtime_warnings_from_pure_cells(self) -> None:
        # A frame with some spatially pure cells and some mixed cells
        # must not emit numpy log(0) warnings while computing entropy.
        import warnings

        large = [_det(5, 5), _det(6, 6), _det(40, 5)]
        small = [_det(5, 40), _det(40, 40), _det(41, 41)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_segregation_metrics(
                large, small, image_width=64, image_height=64, grid_spacing=32
            )


class TestPlotSegregationSummary:
    def test_produces_a_valid_png(self, tmp_path: Path) -> None:
        summary = SegregationSummary(
            frame_ids=[0, 1, 2],
            times_s=[0.0, 0.1, 0.2],
            metrics=[
                SegregationMetrics(
                    frame_id=i,
                    large_count=4,
                    small_count=4,
                    segregation_index=1.0 - 0.3 * i,
                    mixing_index=0.3 * i,
                    mixing_entropy=0.2 * i,
                )
                for i in range(3)
            ],
        )
        output = tmp_path / "summary.png"

        result = plot_segregation_summary(summary, output)

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        summary = SegregationSummary(
            frame_ids=[0],
            times_s=[0.0],
            metrics=[
                SegregationMetrics(
                    frame_id=0,
                    large_count=1,
                    small_count=1,
                    segregation_index=0.5,
                    mixing_index=0.5,
                    mixing_entropy=0.5,
                )
            ],
        )
        output = tmp_path / "nested" / "dir" / "summary.png"

        plot_segregation_summary(summary, output)
        assert output.is_file()


class TestAnalyzeSegregation:
    def _make_dataset(
        self, tmp_path: Path, *, size: int = 200, frame_count: int = 3, segregated: bool = True
    ) -> Path:
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

        large_radius = 10
        small_radius = 4
        # Equal counts of each population so the median radius (used as
        # the automatic size_threshold) falls strictly between the two
        # radii, rather than colliding with one of them. Two particles
        # per grid cell (grid_spacing=100 -> a 2x2 grid for size=200), so
        # each occupied cell samples more than one particle -- otherwise
        # compute_segregation_metrics() cannot distinguish segregation
        # from randomness (see its docstring).
        if segregated:
            # Every cell's two particles are the same population --
            # large confined to the top row of cells, small to the
            # bottom row: clearly segregated.
            large_centers = [(30, 30), (70, 30), (130, 30), (170, 30)]
            small_centers = [(30, 170), (70, 170), (130, 170), (170, 170)]
        else:
            # Every cell has one large and one small particle --
            # locally matches the global 50/50 composition everywhere:
            # well mixed.
            large_centers = [(30, 30), (130, 30), (30, 170), (130, 170)]
            small_centers = [(70, 30), (170, 30), (70, 170), (170, 170)]

        for i in range(frame_count):
            image = np.zeros((size, size), dtype=np.uint8)
            for cx, cy in large_centers:
                cv2.circle(image, (cx, cy), large_radius, 255, -1)
            for cx, cy in small_centers:
                cv2.circle(image, (cx, cy), small_radius, 255, -1)
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

    def test_returns_metrics_per_frame(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=3)
        summary = analyze_segregation(folder, grid_spacing=100)

        assert isinstance(summary, SegregationSummary)
        assert summary.frame_ids == [0, 1, 2]
        assert len(summary.metrics) == 3
        for metrics in summary.metrics:
            assert metrics.large_count == 4
            assert metrics.small_count == 4

    def test_segregated_dataset_has_high_segregation_index(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=2, segregated=True)
        summary = analyze_segregation(folder, grid_spacing=100)
        assert summary.metrics[0].segregation_index > 0.5

    def test_mixed_dataset_has_low_segregation_index(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=2, segregated=False)
        summary = analyze_segregation(folder, grid_spacing=100)
        assert summary.metrics[0].segregation_index < 0.5

    def test_times_s_reflect_real_elapsed_time(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=3)
        summary = analyze_segregation(folder, grid_spacing=100)
        assert summary.times_s == pytest.approx([0.0, 0.1, 0.2])

    def test_explicit_size_threshold_is_respected(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=1)
        # A very high threshold should classify everything as "small".
        summary = analyze_segregation(folder, size_threshold=1000.0, grid_spacing=100)
        assert summary.metrics[0].large_count == 0
        assert summary.metrics[0].small_count == 8

    def test_optionally_writes_a_plot(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path, frame_count=2)
        plot_path = tmp_path / "plot.png"

        analyze_segregation(folder, grid_spacing=100, plot_path=plot_path)
        assert plot_path.is_file()

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

        with pytest.raises(SegregationError):
            analyze_segregation(folder)

    def test_no_particles_detected_without_explicit_threshold_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "blank"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=20,
            height=20,
            created_at_utc="2026-07-16T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        dataset.append_frame(
            Frame(
                frame_id=0,
                image=np.zeros((20, 20), dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=0,
                device_timestamp_ticks=0,
            )
        )
        dataset.finalize()

        with pytest.raises(SegregationError):
            analyze_segregation(folder)

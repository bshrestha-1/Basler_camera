"""Tests for glas.analysis.convection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glas.analysis.convection import (
    ConvectionSummary,
    VelocityField,
    analyze_convection,
    compute_optical_flow,
    compute_vorticity,
    plot_velocity_heatmap,
    total_circulation,
)
from glas.dataset import Dataset
from glas.exceptions import ConvectionError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _textured_image(size: int = 220, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def _rotation_field(domain_size: int = 20, grid_spacing: int = 1) -> VelocityField:
    """An exact solid-body rotation field: vx=-y, vy=x has constant vorticity=2.

    ``x``/``y`` are real coordinate values spaced ``grid_spacing`` apart
    (matching how :func:`compute_optical_flow` itself builds a sample
    grid), covering the same ``[0, domain_size)`` physical region
    regardless of ``grid_spacing`` -- so vorticity stays exactly 2
    everywhere and total circulation stays consistent across different
    sampling densities (Stokes' theorem: the same physical region's
    circulation shouldn't depend on how coarsely it's sampled). Both vx
    and vy are linear functions of the real coordinates, so a finite
    difference recovers the exact analytic derivative (zero truncation
    error) -- this makes for an exact, not approximate, expected value.
    """
    xs = np.arange(0, domain_size, grid_spacing)
    ys = np.arange(0, domain_size, grid_spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    vx = -grid_y.astype(np.float64)
    vy = grid_x.astype(np.float64)
    return VelocityField(
        frame_id=0,
        elapsed_s=1.0,
        grid_spacing=grid_spacing,
        x=grid_x,
        y=grid_y,
        vx=vx,
        vy=vy,
    )


class TestComputeOpticalFlow:
    def test_detects_a_uniform_rightward_shift(self) -> None:
        base = _textured_image()
        shift = 6
        shifted = np.zeros_like(base)
        shifted[:, shift:] = base[:, :-shift]

        field = compute_optical_flow(base, shifted, elapsed_s=1.0)

        assert field.vx.mean() == pytest.approx(shift, abs=1.0)
        assert field.vy.mean() == pytest.approx(0.0, abs=1.0)

    def test_detects_a_uniform_downward_shift(self) -> None:
        base = _textured_image()
        shift = 4
        shifted = np.zeros_like(base)
        shifted[shift:, :] = base[:-shift, :]

        field = compute_optical_flow(base, shifted, elapsed_s=1.0)

        assert field.vy.mean() == pytest.approx(shift, abs=1.0)
        assert field.vx.mean() == pytest.approx(0.0, abs=1.0)

    def test_elapsed_s_scales_velocity(self) -> None:
        base = _textured_image()
        shift = 6
        shifted = np.zeros_like(base)
        shifted[:, shift:] = base[:, :-shift]

        field_1s = compute_optical_flow(base, shifted, elapsed_s=1.0)
        field_half_s = compute_optical_flow(base, shifted, elapsed_s=0.5)

        assert field_half_s.vx.mean() == pytest.approx(field_1s.vx.mean() * 2, rel=0.05)

    def test_grid_spacing_controls_sample_count(self) -> None:
        base = _textured_image(size=100)
        field_coarse = compute_optical_flow(base, base, grid_spacing=50)
        field_fine = compute_optical_flow(base, base, grid_spacing=10)

        assert field_coarse.vx.size < field_fine.vx.size

    def test_frame_id_is_recorded(self) -> None:
        base = _textured_image()
        field = compute_optical_flow(base, base, frame_id=42)
        assert field.frame_id == 42

    def test_speed_is_hypot_of_components(self) -> None:
        base = _textured_image()
        field = compute_optical_flow(base, base)
        assert np.array_equal(field.speed, np.hypot(field.vx, field.vy))

    def test_rejects_non_positive_elapsed_s(self) -> None:
        base = _textured_image()
        with pytest.raises(ValueError):
            compute_optical_flow(base, base, elapsed_s=0.0)
        with pytest.raises(ValueError):
            compute_optical_flow(base, base, elapsed_s=-1.0)

    def test_rejects_non_positive_grid_spacing(self) -> None:
        base = _textured_image()
        with pytest.raises(ValueError):
            compute_optical_flow(base, base, grid_spacing=0)

    def test_16_bit_images_are_scaled_before_flow(self) -> None:
        base = (_textured_image().astype(np.uint16)) * 200
        field = compute_optical_flow(base, base)  # must not raise
        assert field.vx.shape == field.vy.shape


class TestComputeVorticity:
    def test_solid_body_rotation_has_exact_constant_vorticity(self) -> None:
        field = _rotation_field()
        vorticity = compute_vorticity(field)
        assert vorticity == pytest.approx(2.0)

    def test_uniform_translation_has_zero_vorticity(self) -> None:
        size = 10
        xs = np.arange(size)
        ys = np.arange(size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        field = VelocityField(
            frame_id=0,
            elapsed_s=1.0,
            grid_spacing=1,
            x=grid_x,
            y=grid_y,
            vx=np.full((size, size), 5.0),
            vy=np.full((size, size), -3.0),
        )
        vorticity = compute_vorticity(field)
        assert vorticity == pytest.approx(0.0, abs=1e-9)

    def test_output_shape_matches_input(self) -> None:
        field = _rotation_field(domain_size=15)
        assert compute_vorticity(field).shape == field.vx.shape


class TestTotalCirculation:
    def test_matches_analytic_value_for_solid_body_rotation(self) -> None:
        domain_size = 20
        field = _rotation_field(domain_size=domain_size, grid_spacing=1)
        # vorticity == 2 everywhere, grid_spacing=1 -> circulation = 2 * n_cells.
        assert total_circulation(field) == pytest.approx(2.0 * domain_size * domain_size)

    def test_consistent_across_grid_spacing(self) -> None:
        # Stokes' theorem: circulation over the same physical region should
        # not depend on how coarsely that region is sampled.
        field_1 = _rotation_field(domain_size=20, grid_spacing=1)
        field_2 = _rotation_field(domain_size=20, grid_spacing=2)
        assert total_circulation(field_2) == pytest.approx(total_circulation(field_1))

    def test_zero_for_uniform_translation(self) -> None:
        size = 10
        xs = np.arange(size)
        ys = np.arange(size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        field = VelocityField(
            frame_id=0,
            elapsed_s=1.0,
            grid_spacing=1,
            x=grid_x,
            y=grid_y,
            vx=np.full((size, size), 5.0),
            vy=np.full((size, size), -3.0),
        )
        assert total_circulation(field) == pytest.approx(0.0, abs=1e-6)


class TestPlotVelocityHeatmap:
    def test_produces_a_valid_png_with_speed_background(self, tmp_path: Path) -> None:
        field = _rotation_field()
        output = tmp_path / "heatmap.png"

        result = plot_velocity_heatmap(field, output, background="speed")

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width > 0

    def test_produces_a_valid_png_with_vorticity_background(self, tmp_path: Path) -> None:
        field = _rotation_field()
        output = tmp_path / "heatmap.png"

        plot_velocity_heatmap(field, output, background="vorticity")
        assert output.is_file()

    def test_works_without_quiver(self, tmp_path: Path) -> None:
        field = _rotation_field()
        output = tmp_path / "heatmap.png"

        plot_velocity_heatmap(field, output, show_quiver=False)
        assert output.is_file()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        field = _rotation_field()
        output = tmp_path / "nested" / "dir" / "heatmap.png"

        plot_velocity_heatmap(field, output)
        assert output.is_file()

    def test_unknown_background_raises(self, tmp_path: Path) -> None:
        field = _rotation_field()
        with pytest.raises(ValueError):
            plot_velocity_heatmap(field, tmp_path / "x.png", background="bogus")  # type: ignore[arg-type]


class TestAnalyzeConvection:
    def _make_shifting_dataset(
        self, tmp_path: Path, frame_count: int = 5, shift_per_frame: int = 3
    ) -> Path:
        folder = tmp_path / "dataset"
        size = 200
        pad = shift_per_frame * frame_count + 20
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
        base = _textured_image(size=size + pad)
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

    def test_returns_a_field_per_consecutive_pair(self, tmp_path: Path) -> None:
        folder = self._make_shifting_dataset(tmp_path, frame_count=5)
        summary = analyze_convection(folder)

        assert isinstance(summary, ConvectionSummary)
        assert summary.frame_ids == [1, 2, 3, 4]
        assert len(summary.fields) == 4
        assert len(summary.circulations) == 4

    def test_times_s_reflect_real_elapsed_time(self, tmp_path: Path) -> None:
        folder = self._make_shifting_dataset(tmp_path, frame_count=3)
        summary = analyze_convection(folder)
        assert summary.times_s == pytest.approx([0.1, 0.2])

    def test_detects_expected_shift_direction(self, tmp_path: Path) -> None:
        folder = self._make_shifting_dataset(tmp_path, frame_count=3, shift_per_frame=3)
        summary = analyze_convection(folder)
        # Content moves left within the frame as the sampling window
        # shifts right through the source texture.
        assert summary.fields[0].vx.mean() < 0

    def test_optionally_writes_heatmaps(self, tmp_path: Path) -> None:
        folder = self._make_shifting_dataset(tmp_path, frame_count=4)
        heatmap_dir = tmp_path / "heatmaps"

        analyze_convection(folder, heatmap_dir=heatmap_dir)

        pngs = sorted(heatmap_dir.glob("*.png"))
        assert len(pngs) == 3

    def test_no_heatmaps_written_when_dir_omitted(self, tmp_path: Path) -> None:
        folder = self._make_shifting_dataset(tmp_path, frame_count=3)
        analyze_convection(folder)
        assert not (tmp_path / "heatmaps").exists()

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "empty"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=10,
            height=10,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        Dataset.create(folder, metadata, dataset_format="hdf5").finalize()

        with pytest.raises(ConvectionError):
            analyze_convection(folder)

    def test_single_frame_dataset_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "one"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=10,
            height=10,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        dataset.append_frame(
            Frame(
                frame_id=0,
                image=np.zeros((10, 10), dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=0,
                device_timestamp_ticks=0,
            )
        )
        dataset.finalize()

        with pytest.raises(ConvectionError):
            analyze_convection(folder)

    def test_non_increasing_timestamps_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_ts"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=10,
            height=10,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        for i in range(2):
            dataset.append_frame(
                Frame(
                    frame_id=i,
                    image=np.zeros((10, 10), dtype=np.uint8),
                    pixel_format="Mono8",
                    host_timestamp_ns=1000,  # same timestamp for both frames
                    device_timestamp_ticks=i,
                )
            )
        dataset.finalize()

        with pytest.raises(ConvectionError):
            analyze_convection(folder)

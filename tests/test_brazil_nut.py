"""Tests for glas.analysis.brazil_nut."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from glas.analysis.brazil_nut import (
    BrazilNutTrajectory,
    analyze_brazil_nut,
    compute_brazil_nut_trajectory,
    identify_brazil_nut,
    plot_brazil_nut_trajectory,
)
from glas.analysis.particle_tracking import TrackedParticle
from glas.dataset import Dataset
from glas.exceptions import BrazilNutError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _particle(
    track_id: int, frame_id: int, y: float, radius: float = 10.0, host_timestamp_ns: int = 0
) -> TrackedParticle:
    return TrackedParticle(
        track_id=track_id,
        frame_id=frame_id,
        x=50.0,
        y=y,
        radius=radius,
        area=radius**2,
        host_timestamp_ns=host_timestamp_ns,
    )


def _rising_history(
    *,
    frame_count: int = 10,
    start_y: float = 180.0,
    step: float = 20.0,
    ns_per_frame: int = 100_000_000,
) -> dict[int, list[TrackedParticle]]:
    return {
        0: [
            _particle(0, i, start_y - i * step, radius=12.0, host_timestamp_ns=i * ns_per_frame)
            for i in range(frame_count)
        ]
    }


class TestIdentifyBrazilNut:
    def test_largest_mean_radius_wins(self) -> None:
        history = {
            0: [_particle(0, 0, y=10.0, radius=3.0)],
            1: [_particle(1, 0, y=20.0, radius=12.0)],
            2: [_particle(2, 0, y=30.0, radius=5.0)],
        }
        assert identify_brazil_nut(history) == 1

    def test_uses_mean_radius_across_the_whole_trajectory(self) -> None:
        history = {
            0: [_particle(0, i, y=10.0, radius=20.0) for i in range(1)],  # mean 20, one obs
            1: [
                _particle(1, 0, y=10.0, radius=5.0),
                _particle(1, 1, y=10.0, radius=25.0),
            ],  # mean 15
        }
        assert identify_brazil_nut(history) == 0

    def test_empty_history_raises(self) -> None:
        with pytest.raises(BrazilNutError):
            identify_brazil_nut({})


class TestComputeBrazilNutTrajectory:
    def test_heights_are_inverted_from_image_y(self) -> None:
        history = _rising_history()
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)

        assert trajectory.heights_px == pytest.approx([20.0 + i * 20.0 for i in range(10)])

    def test_times_derived_from_host_timestamp_ns(self) -> None:
        history = _rising_history(ns_per_frame=50_000_000)  # 50ms/frame
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)

        assert trajectory.times_s == pytest.approx([i * 0.05 for i in range(10)])

    def test_velocities_are_finite_differences(self) -> None:
        history = _rising_history(step=20.0, ns_per_frame=100_000_000)  # 20px / 0.1s = 200 px/s
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)

        assert len(trajectory.velocities_px_s) == len(trajectory.heights_px) - 1
        assert trajectory.velocities_px_s == pytest.approx([200.0] * 9)

    def test_mean_velocity_is_overall_average(self) -> None:
        history = _rising_history(step=20.0, ns_per_frame=100_000_000)
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)

        assert trajectory.mean_velocity_px_s == pytest.approx(200.0)

    def test_rise_time_found_when_threshold_is_crossed(self) -> None:
        history = _rising_history(start_y=180.0, step=20.0, ns_per_frame=100_000_000)
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200, settle_fraction=0.9)

        # height reaches 180 (0.9 * 200) at frame 8 -> t = 0.8s
        assert trajectory.rise_time_s == pytest.approx(0.8)

    def test_rise_time_is_none_when_never_reached(self) -> None:
        history = _rising_history(start_y=180.0, step=1.0)  # barely moves
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200, settle_fraction=0.9)

        assert trajectory.rise_time_s is None

    def test_auto_identifies_track_when_track_id_omitted(self) -> None:
        history = {
            0: [_particle(0, i, y=100.0, radius=3.0, host_timestamp_ns=i * 10) for i in range(3)],
            1: [
                _particle(1, i, y=100.0 - i * 10, radius=15.0, host_timestamp_ns=i * 10)
                for i in range(3)
            ],
        }
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)
        assert trajectory.track_id == 1

    def test_explicit_track_id_is_respected(self) -> None:
        history = {
            0: [_particle(0, i, y=100.0, radius=3.0, host_timestamp_ns=i * 10) for i in range(3)],
            1: [
                _particle(1, i, y=100.0 - i * 10, radius=15.0, host_timestamp_ns=i * 10)
                for i in range(3)
            ],
        }
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200, track_id=0)
        assert trajectory.track_id == 0

    def test_non_increasing_timestamps_raises(self) -> None:
        history = {
            0: [
                _particle(0, 0, y=100.0, host_timestamp_ns=1000),
                _particle(0, 1, y=90.0, host_timestamp_ns=1000),  # same timestamp as frame 0
            ]
        }
        with pytest.raises(BrazilNutError):
            compute_brazil_nut_trajectory(history, frame_height=200)

    def test_decreasing_timestamps_raises(self) -> None:
        history = {
            0: [
                _particle(0, 0, y=100.0, host_timestamp_ns=2000),
                _particle(0, 1, y=90.0, host_timestamp_ns=1000),  # earlier than frame 0
            ]
        }
        with pytest.raises(BrazilNutError):
            compute_brazil_nut_trajectory(history, frame_height=200)

    def test_unknown_track_id_raises(self) -> None:
        history = _rising_history()
        with pytest.raises(BrazilNutError):
            compute_brazil_nut_trajectory(history, frame_height=200, track_id=99)

    def test_empty_history_raises(self) -> None:
        with pytest.raises(BrazilNutError):
            compute_brazil_nut_trajectory({}, frame_height=200)

    def test_single_observation_track_raises(self) -> None:
        history = {0: [_particle(0, 0, y=100.0)]}
        with pytest.raises(BrazilNutError):
            compute_brazil_nut_trajectory(history, frame_height=200)

    @pytest.mark.parametrize("settle_fraction", [0.0, -0.5, 1.1])
    def test_invalid_settle_fraction_raises(self, settle_fraction: float) -> None:
        history = _rising_history()
        with pytest.raises(ValueError):
            compute_brazil_nut_trajectory(
                history, frame_height=200, settle_fraction=settle_fraction
            )

    def test_settle_fraction_of_one_is_valid(self) -> None:
        history = _rising_history()
        compute_brazil_nut_trajectory(history, frame_height=200, settle_fraction=1.0)  # no raise

    def test_is_frozen(self) -> None:
        history = _rising_history()
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            trajectory.track_id = 99  # type: ignore[misc]


class TestPlotBrazilNutTrajectory:
    def test_produces_a_valid_png(self, tmp_path: Path) -> None:
        history = _rising_history()
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)
        output = tmp_path / "plot.png"

        result = plot_brazil_nut_trajectory(trajectory, output)

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        history = _rising_history()
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200)
        output = tmp_path / "nested" / "dir" / "plot.png"

        plot_brazil_nut_trajectory(trajectory, output)
        assert output.is_file()

    def test_works_without_a_rise_time(self, tmp_path: Path) -> None:
        history = _rising_history(start_y=180.0, step=1.0)  # never crosses threshold
        trajectory = compute_brazil_nut_trajectory(history, frame_height=200, settle_fraction=0.9)
        assert trajectory.rise_time_s is None

        output = tmp_path / "plot.png"
        plot_brazil_nut_trajectory(trajectory, output)  # must not raise
        assert output.is_file()


class TestAnalyzeBrazilNut:
    def _make_dataset(self, tmp_path: Path, frame_count: int = 8) -> Path:
        folder = tmp_path / "dataset"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=100,
            height=200,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        for i in range(frame_count):
            image = np.zeros((200, 100), dtype=np.uint8)
            cv2.circle(image, (50, 180 - i * 15), 12, 255, -1)  # big intruder, rising
            cv2.circle(image, (20, 20), 3, 255, -1)  # small static particle
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

    def test_identifies_and_analyzes_the_intruder(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path)
        trajectory = analyze_brazil_nut(folder)

        assert isinstance(trajectory, BrazilNutTrajectory)
        assert len(trajectory.heights_px) == 8
        # Rising: heights should be strictly increasing.
        assert trajectory.heights_px == sorted(trajectory.heights_px)
        assert trajectory.mean_velocity_px_s > 0

    def test_optionally_writes_a_plot(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path)
        plot_path = tmp_path / "brazil_nut.png"

        analyze_brazil_nut(folder, plot_path=plot_path)
        assert plot_path.is_file()

    def test_no_plot_written_when_plot_path_omitted(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path)
        analyze_brazil_nut(folder)
        assert list(tmp_path.glob("*.png")) == []

    def test_respects_explicit_track_id(self, tmp_path: Path) -> None:
        folder = self._make_dataset(tmp_path)
        from glas.analysis import track_dataset

        history = track_dataset(folder)
        small_track_id = min(history, key=lambda tid: history[tid][0].radius)

        trajectory = analyze_brazil_nut(folder, track_id=small_track_id)
        assert trajectory.track_id == small_track_id

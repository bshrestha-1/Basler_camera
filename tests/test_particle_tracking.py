"""Tests for glas.analysis.particle_tracking."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from glas.analysis.particle_tracking import ParticleTracker, TrackedParticle, track_dataset
from glas.analysis.tracking_utils import Detection
from glas.dataset import Dataset
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _detection(x: float, y: float) -> Detection:
    return Detection(x=x, y=y, radius=5.0, area=78.5)


class TestParticleTrackerConstruction:
    def test_rejects_non_positive_max_distance(self) -> None:
        with pytest.raises(ValueError):
            ParticleTracker(max_distance=0)
        with pytest.raises(ValueError):
            ParticleTracker(max_distance=-1)

    def test_rejects_negative_max_gap(self) -> None:
        with pytest.raises(ValueError):
            ParticleTracker(max_gap=-1)

    def test_starts_with_no_active_tracks(self) -> None:
        tracker = ParticleTracker()
        assert tracker.active_track_count == 0
        assert tracker.history == {}


class TestParticleTrackerUpdate:
    def test_first_frame_spawns_a_new_track_per_detection(self) -> None:
        tracker = ParticleTracker()
        observed = tracker.update(0, [_detection(10, 10), _detection(50, 50)])

        assert len(observed) == 2
        assert {p.track_id for p in observed} == {0, 1}
        assert tracker.active_track_count == 2

    def test_a_moving_particle_keeps_the_same_track_id(self) -> None:
        tracker = ParticleTracker(max_distance=10.0)
        tracker.update(0, [_detection(10, 10)])
        tracker.update(1, [_detection(12, 11)])
        observed = tracker.update(2, [_detection(14, 12)])

        assert len(observed) == 1
        assert observed[0].track_id == 0
        assert tracker.active_track_count == 1

    def test_a_far_detection_spawns_a_new_track_instead_of_continuing(self) -> None:
        tracker = ParticleTracker(max_distance=5.0, max_gap=0)
        tracker.update(0, [_detection(10, 10)])
        observed = tracker.update(1, [_detection(500, 500)])

        assert observed[0].track_id == 1
        # The original track went unmatched this frame and max_gap=0, so
        # it's retired immediately -- only the new track remains active.
        assert tracker.active_track_count == 1
        assert set(tracker.history.keys()) == {0, 1}

    def test_track_ids_are_never_reused(self) -> None:
        tracker = ParticleTracker(max_distance=5.0, max_gap=0)
        tracker.update(0, [_detection(10, 10)])  # spawns track_id=0
        tracker.update(1, [])  # track 0 goes unmatched, retired (max_gap=0)
        observed = tracker.update(2, [_detection(10, 10)])  # spawns a new track

        assert observed[0].track_id == 1  # not reusing the retired track_id 0
        assert set(tracker.history.keys()) == {0, 1}

    def test_track_retired_after_max_gap_frames_unmatched(self) -> None:
        tracker = ParticleTracker(max_distance=5.0, max_gap=1)
        tracker.update(0, [_detection(10, 10)])
        tracker.update(1, [])  # missed once -- still within max_gap=1
        assert tracker.active_track_count == 1

        tracker.update(2, [])  # missed twice -- gap now 2 > max_gap=1
        assert tracker.active_track_count == 0

    def test_track_resumes_within_max_gap_window(self) -> None:
        tracker = ParticleTracker(max_distance=5.0, max_gap=2)
        tracker.update(0, [_detection(10, 10)])
        tracker.update(1, [])  # occluded
        observed = tracker.update(2, [_detection(11, 11)])  # reappears

        assert observed[0].track_id == 0  # same track, not a new one
        assert tracker.active_track_count == 1

    def test_history_accumulates_every_observation_in_order(self) -> None:
        tracker = ParticleTracker(max_distance=10.0)
        tracker.update(0, [_detection(0, 0)])
        tracker.update(1, [_detection(5, 5)])
        tracker.update(2, [_detection(10, 10)])

        history = tracker.history
        assert list(history.keys()) == [0]
        assert [p.frame_id for p in history[0]] == [0, 1, 2]
        assert [p.x for p in history[0]] == [0, 5, 10]

    def test_history_returns_a_copy_not_a_live_view(self) -> None:
        tracker = ParticleTracker()
        tracker.update(0, [_detection(0, 0)])
        history = tracker.history
        history[0].append(TrackedParticle(track_id=0, frame_id=99, x=0, y=0, radius=1, area=1))
        assert len(tracker.history[0]) == 1  # unaffected by the mutation above

    def test_returned_particles_match_detection_fields(self) -> None:
        tracker = ParticleTracker()
        detection = Detection(x=12.5, y=34.5, radius=3.0, area=28.3)
        observed = tracker.update(0, [detection])

        particle = observed[0]
        assert particle.x == detection.x
        assert particle.y == detection.y
        assert particle.radius == detection.radius
        assert particle.area == detection.area
        assert particle.frame_id == 0

    def test_host_timestamp_ns_defaults_to_zero(self) -> None:
        tracker = ParticleTracker()
        observed = tracker.update(0, [_detection(0, 0)])
        assert observed[0].host_timestamp_ns == 0

    def test_host_timestamp_ns_is_recorded_when_given(self) -> None:
        tracker = ParticleTracker(max_distance=10.0)
        tracker.update(0, [_detection(0, 0)], host_timestamp_ns=1_000_000_000)
        observed = tracker.update(1, [_detection(1, 1)], host_timestamp_ns=1_100_000_000)

        assert observed[0].host_timestamp_ns == 1_100_000_000
        assert tracker.history[0][0].host_timestamp_ns == 1_000_000_000
        assert tracker.history[0][1].host_timestamp_ns == 1_100_000_000


class TestTrackDataset:
    def _make_moving_blob_dataset(self, tmp_path: Path, frame_count: int = 5) -> Path:
        folder = tmp_path / "dataset"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=100,
            height=100,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
        for i in range(frame_count):
            image = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(image, (20 + i * 5, 50), 5, 255, -1)
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

    def test_single_moving_particle_resolves_to_one_track(self, tmp_path: Path) -> None:
        folder = self._make_moving_blob_dataset(tmp_path)
        history = track_dataset(folder)

        assert len(history) == 1
        observations = next(iter(history.values()))
        assert len(observations) == 5
        assert [p.frame_id for p in observations] == [0, 1, 2, 3, 4]
        # Moving monotonically to the right, 5px per frame.
        xs = [p.x for p in observations]
        assert xs == sorted(xs)
        # Each Frame's host_timestamp_ns (set to frame_id * 1000 above)
        # flows through onto the corresponding TrackedParticle.
        assert [p.host_timestamp_ns for p in observations] == [0, 1000, 2000, 3000, 4000]

    def test_empty_dataset_yields_no_tracks(self, tmp_path: Path) -> None:
        folder = tmp_path / "dataset"
        metadata = DatasetMetadata(
            dataset_format="hdf5",
            camera_model="acA640-750um",
            camera_serial="12345678",
            pixel_format="Mono8",
            width=100,
            height=100,
            created_at_utc="2026-07-13T00:00:00+00:00",
        )
        Dataset.create(folder, metadata, dataset_format="hdf5").finalize()

        assert track_dataset(folder) == {}

    def test_respects_tracking_parameters(self, tmp_path: Path) -> None:
        folder = self._make_moving_blob_dataset(tmp_path)
        # A tiny max_distance can't keep up with a 5px/frame jump -- each
        # frame's detection should spawn a new track instead of one
        # continuous trajectory.
        history = track_dataset(folder, max_distance=1.0)
        assert len(history) == 5

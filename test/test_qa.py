"""Tests for glas.qa."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from glas.dataset import Dataset
from glas.frame import Frame
from glas.metadata import DatasetMetadata
from glas.qa import (
    HealthCheckResult,
    RecordingQualityReport,
    assess_recording_quality,
    run_preflight_checks,
)


def _make_dataset(
    tmp_path: Path,
    *,
    frame_count: int = 10,
    interval_ns: int = 33_000_000,
    draw_particle: bool = True,
    gap_at: int | None = None,
) -> Path:
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
        if gap_at is not None and i == gap_at:
            continue
        image = np.zeros((100, 100), dtype=np.uint8)
        if draw_particle:
            cv2.circle(image, (20 + i, 50), 6, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * interval_ns,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


class TestHealthCheckResult:
    def test_all_passed_true_when_every_item_passes(self) -> None:
        from glas.qa import HealthCheckItem

        result = HealthCheckResult(
            items=[
                HealthCheckItem(name="a", passed=True, message="ok"),
                HealthCheckItem(name="b", passed=True, message="ok"),
            ]
        )
        assert result.all_passed is True

    def test_all_passed_false_when_any_item_fails(self) -> None:
        from glas.qa import HealthCheckItem

        result = HealthCheckResult(
            items=[
                HealthCheckItem(name="a", passed=True, message="ok"),
                HealthCheckItem(name="b", passed=False, message="bad"),
            ]
        )
        assert result.all_passed is False

    def test_all_passed_true_for_empty_items(self) -> None:
        assert HealthCheckResult(items=[]).all_passed is True


pypylon = pytest.importorskip("pypylon")

from glas.camera import Camera  # noqa: E402
from glas.camera_info import detect_cameras  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )


@pytest.fixture
def camera() -> Iterator[Camera]:
    cam = Camera()
    yield cam
    cam.disconnect()


class TestRunPreflightChecks:
    def test_disk_space_check_always_present(self, camera: Camera, tmp_path: Path) -> None:
        result = run_preflight_checks(camera, tmp_path, min_sharpness=0.0)
        names = [item.name for item in result.items]
        assert "disk_space" in names

    def test_disk_space_fails_when_threshold_absurdly_high(
        self, camera: Camera, tmp_path: Path
    ) -> None:
        result = run_preflight_checks(camera, tmp_path, min_disk_free_gb=1e9)
        disk_item = next(item for item in result.items if item.name == "disk_space")
        assert disk_item.passed is False

    def test_camera_not_connected_reports_failure(self, camera: Camera, tmp_path: Path) -> None:
        result = run_preflight_checks(camera, tmp_path)
        connected_item = next(item for item in result.items if item.name == "camera_connected")
        assert connected_item.passed is False

    def test_camera_not_connected_skips_frame_based_checks(
        self, camera: Camera, tmp_path: Path
    ) -> None:
        result = run_preflight_checks(camera, tmp_path)
        names = {item.name for item in result.items}
        assert "focus" not in names
        assert "exposure_level" not in names

    def test_connected_camera_runs_frame_based_checks(self, camera: Camera, tmp_path: Path) -> None:
        camera.connect()
        result = run_preflight_checks(camera, tmp_path, min_sharpness=0.0)
        names = {item.name for item in result.items}
        assert "focus" in names
        assert "exposure_level" in names
        assert "exposure_sanity" in names
        assert "gain_sanity" in names

    def test_focus_check_uses_min_sharpness_threshold(self, camera: Camera, tmp_path: Path) -> None:
        camera.connect()
        result = run_preflight_checks(camera, tmp_path, min_sharpness=1e12)
        focus_item = next(item for item in result.items if item.name == "focus")
        assert focus_item.passed is False

    def test_calibration_check_absent_when_not_requested(
        self, camera: Camera, tmp_path: Path
    ) -> None:
        result = run_preflight_checks(camera, tmp_path)
        names = {item.name for item in result.items}
        assert "calibration_present" not in names

    def test_calibration_check_fails_when_file_missing(
        self, camera: Camera, tmp_path: Path
    ) -> None:
        result = run_preflight_checks(
            camera, tmp_path, calibration_path=tmp_path / "does_not_exist.json"
        )
        calibration_item = next(item for item in result.items if item.name == "calibration_present")
        assert calibration_item.passed is False

    def test_calibration_check_passes_when_file_present(
        self, camera: Camera, tmp_path: Path
    ) -> None:
        calibration_path = tmp_path / "calibration.json"
        calibration_path.write_text("{}")
        result = run_preflight_checks(camera, tmp_path, calibration_path=calibration_path)
        calibration_item = next(item for item in result.items if item.name == "calibration_present")
        assert calibration_item.passed is True

    def test_returns_health_check_result(self, camera: Camera, tmp_path: Path) -> None:
        assert isinstance(run_preflight_checks(camera, tmp_path), HealthCheckResult)


class TestAssessRecordingQuality:
    def test_clean_recording_has_no_warnings(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path)
        report = assess_recording_quality(folder, expected_fps=30.0)
        assert report.is_clean
        assert report.warnings == []

    def test_frame_count_matches_dataset(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=15)
        report = assess_recording_quality(folder)
        assert report.frame_count == 15

    def test_detects_dropped_frames(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=10, gap_at=5)
        report = assess_recording_quality(folder)
        assert report.dropped_frame_count == 1
        assert report.frame_id_gaps == [(5, 5)]
        assert any("dropped" in w for w in report.warnings)

    def test_mean_fps_matches_known_interval(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, interval_ns=20_000_000)  # 50 fps
        report = assess_recording_quality(folder)
        assert report.mean_fps == pytest.approx(50.0, rel=0.01)

    def test_regular_interval_has_near_zero_jitter(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path)
        report = assess_recording_quality(folder)
        assert report.fps_jitter_percent == pytest.approx(0.0, abs=0.1)

    def test_expected_fps_mismatch_produces_warning(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, interval_ns=33_000_000)  # ~30 fps
        report = assess_recording_quality(folder, expected_fps=100.0)
        assert any("deviates" in w for w in report.warnings)

    def test_expected_fps_within_tolerance_no_warning(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, interval_ns=33_000_000)  # ~30.3 fps
        report = assess_recording_quality(folder, expected_fps=30.0)
        assert not any("deviates" in w for w in report.warnings)

    def test_particle_counts_are_correct_for_single_particle_frames(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=5)
        report = assess_recording_quality(folder)
        assert report.mean_particle_count == pytest.approx(1.0)
        assert report.min_particle_count == 1
        assert report.max_particle_count == 1

    def test_frames_with_no_particles_detected_and_warned(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=5, draw_particle=False)
        report = assess_recording_quality(folder)
        assert report.frames_with_no_particles == 5
        assert any("no detected particles" in w for w in report.warnings)

    def test_sampled_frame_count_limited_by_max_sample_frames(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=20)
        report = assess_recording_quality(folder, max_sample_frames=5)
        assert report.sampled_frame_count <= 5 + 1  # stride rounding may include one extra

    def test_too_few_frames_for_fps_produces_warning(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=1)
        report = assess_recording_quality(folder)
        assert report.mean_fps == 0.0
        assert any("Too few frames" in w for w in report.warnings)

    def test_returns_recording_quality_report(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=3)
        assert isinstance(assess_recording_quality(folder), RecordingQualityReport)

"""Data-taking quality assurance: catch a bad recording before it starts, or before it's trusted.

Two independent checks, covering the two moments a bad recording is
cheapest to catch:

    glas.camera.Camera (connected) -> run_preflight_checks() -> HealthCheckResult
    glas.dataset.iter_frames()     -> assess_recording_quality() -> RecordingQualityReport

:func:`run_preflight_checks` runs *before* recording starts: disk space,
camera connectivity, exposure/gain not pinned at their device limits, a
grabbed frame's focus (variance of Laplacian, a standard sharpness proxy)
and exposure sanity, and whether a spatial calibration file exists. None
of these guarantee a good recording, but each one catches a specific,
common way a lab session gets wasted (out of disk space, camera asleep,
lens cap on, badly out of focus).

:func:`assess_recording_quality` runs *after* recording finishes, on top
of :func:`glas.dataset.validate_dataset`'s structural/checksum
validation: dropped frames and frame-rate jitter (via
:class:`glas.timestamps.TimestampLog`, replayed from the finalized
frames rather than duplicating its gap-detection logic), and per-frame
particle-count sanity (via :func:`glas.analysis.tracking_utils.detect_particles`
on a subsample, for large recordings) -- catching a recording that's
structurally fine but scientifically useless (e.g. the camera was
pointed at a wall, or lost focus partway through).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from glas.analysis.tracking_utils import DEFAULT_MIN_AREA, detect_particles, to_uint8_mono
from glas.dataset import iter_frames, validate_dataset
from glas.timestamps import TimestampLog

DEFAULT_MIN_DISK_FREE_GB = 1.0
DEFAULT_MIN_SHARPNESS = 50.0
DEFAULT_DARK_MEAN_THRESHOLD = 5.0
DEFAULT_SATURATED_MEAN_THRESHOLD = 250.0
DEFAULT_GRAB_TIMEOUT_MS = 2000
DEFAULT_MAX_SAMPLE_FRAMES = 200
DEFAULT_FPS_JITTER_WARNING_PERCENT = 20.0
DEFAULT_EXPECTED_FPS_TOLERANCE_PERCENT = 10.0


class HealthCheckItem(BaseModel):
    """One pass/fail preflight check.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"disk_space"``, ``"focus"``.
    passed : bool
    message : str
        Human-readable detail (the measured value and threshold, or why
        the check couldn't run).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    message: str


class HealthCheckResult(BaseModel):
    """Every preflight check run by :func:`run_preflight_checks`.

    Attributes
    ----------
    items : list of HealthCheckItem
    """

    model_config = ConfigDict(frozen=True)

    items: list[HealthCheckItem]

    @property
    def all_passed(self) -> bool:
        """``True`` if every check in :attr:`items` passed."""
        return all(item.passed for item in self.items)


def _grab_one_frame(camera: Any, *, timeout_ms: int) -> Any:
    camera.start_grabbing()
    try:
        return camera.retrieve_frame(frame_id=0, timeout_ms=timeout_ms)
    finally:
        camera.stop_grabbing()


def run_preflight_checks(
    camera: Any,
    data_dir: Path,
    *,
    calibration_path: Path | None = None,
    min_disk_free_gb: float = DEFAULT_MIN_DISK_FREE_GB,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    dark_mean_threshold: float = DEFAULT_DARK_MEAN_THRESHOLD,
    saturated_mean_threshold: float = DEFAULT_SATURATED_MEAN_THRESHOLD,
    grab_timeout_ms: int = DEFAULT_GRAB_TIMEOUT_MS,
) -> HealthCheckResult:
    """Run every preflight check that's applicable given the camera's current state.

    Disk space is always checked. Camera connectivity is always checked.
    Exposure/gain sanity and a grabbed frame's focus/exposure are only
    checked if ``camera.is_connected`` -- there is nothing meaningful to
    check about a camera that isn't connected yet, so those checks are
    simply absent from the result rather than reported as failures.

    Parameters
    ----------
    camera : glas.camera.Camera
        Typed as ``Any`` so this module has no hard dependency on
        pypylon being importable -- matches
        :func:`glas.exceptions`'s general policy of keeping camera-driver
        dependencies out of modules that don't need to touch the driver
        directly. Pass an already-connected (or not-yet-connected)
        :class:`~glas.camera.Camera`.
    data_dir : pathlib.Path
        Directory recordings will be written under -- checked for free
        disk space.
    calibration_path : pathlib.Path, optional
        If given, checked for existence (not validity -- use
        :func:`glas.calibration.load_calibration` for that).
    min_disk_free_gb : float, default 1.0
        Minimum acceptable free disk space, in GB.
    min_sharpness : float, default 50.0
        Minimum acceptable focus score (variance of the Laplacian of a
        grabbed frame) -- scene-dependent, so tune this for the actual
        setup rather than trusting the default blindly.
    dark_mean_threshold, saturated_mean_threshold : float, default 5.0, 250.0
        A grabbed frame's mean pixel intensity (0-255 scale) must fall
        between these to avoid flagging a lens cap left on
        (``dark_mean_threshold``) or a badly overexposed scene
        (``saturated_mean_threshold``).
    grab_timeout_ms : int, default 2000
        Timeout for the single test frame grab.

    Returns
    -------
    HealthCheckResult
    """
    items: list[HealthCheckItem] = []

    disk_usage = shutil.disk_usage(data_dir if data_dir.exists() else data_dir.parent)
    disk_free_gb = disk_usage.free / 1e9
    items.append(
        HealthCheckItem(
            name="disk_space",
            passed=disk_free_gb >= min_disk_free_gb,
            message=f"{disk_free_gb:.2f} GB free (minimum {min_disk_free_gb:.2f} GB).",
        )
    )

    items.append(
        HealthCheckItem(
            name="camera_connected",
            passed=camera.is_connected,
            message="Connected." if camera.is_connected else "Camera is not connected.",
        )
    )

    if camera.is_connected:
        exposure_bounds = camera.exposure_time_bounds_us()
        exposure_pinned = camera.exposure_time_us in (
            exposure_bounds.minimum,
            exposure_bounds.maximum,
        )
        items.append(
            HealthCheckItem(
                name="exposure_sanity",
                passed=not exposure_pinned,
                message=(
                    f"exposure_time_us={camera.exposure_time_us:.1f} (device range "
                    f"[{exposure_bounds.minimum:.1f}, {exposure_bounds.maximum:.1f}])."
                ),
            )
        )

        gain_bounds = camera.gain_bounds_db()
        gain_pinned = camera.gain_db in (gain_bounds.minimum, gain_bounds.maximum)
        items.append(
            HealthCheckItem(
                name="gain_sanity",
                passed=not gain_pinned,
                message=(
                    f"gain_db={camera.gain_db:.2f} "
                    f"(device range [{gain_bounds.minimum:.2f}, {gain_bounds.maximum:.2f}])."
                ),
            )
        )

        frame = _grab_one_frame(camera, timeout_ms=grab_timeout_ms)
        if frame is None:
            items.append(
                HealthCheckItem(
                    name="frame_grab", passed=False, message="Timed out grabbing a test frame."
                )
            )
        else:
            mono = to_uint8_mono(frame.image)
            mean_intensity = float(np.mean(mono))
            items.append(
                HealthCheckItem(
                    name="exposure_level",
                    passed=dark_mean_threshold <= mean_intensity <= saturated_mean_threshold,
                    message=f"Mean pixel intensity {mean_intensity:.1f} (0-255 scale).",
                )
            )

            sharpness = float(cv2.Laplacian(mono, cv2.CV_64F).var())
            items.append(
                HealthCheckItem(
                    name="focus",
                    passed=sharpness >= min_sharpness,
                    message=f"Focus score {sharpness:.1f} (minimum {min_sharpness:.1f}).",
                )
            )

    if calibration_path is not None:
        items.append(
            HealthCheckItem(
                name="calibration_present",
                passed=calibration_path.is_file(),
                message=(
                    f"Found {calibration_path}."
                    if calibration_path.is_file()
                    else f"No calibration file at {calibration_path}."
                ),
            )
        )

    return HealthCheckResult(items=items)


class RecordingQualityReport(BaseModel):
    """A post-recording assessment of whether a dataset is scientifically trustworthy.

    Attributes
    ----------
    frame_count : int
        Frames actually present in the recording.
    dropped_frame_count : int
        Total frame IDs missing from the sequence (see
        :meth:`glas.timestamps.TimestampLog.dropped_frame_count`).
    frame_id_gaps : list of (int, int)
        Missing frame-ID ranges (see
        :meth:`glas.timestamps.TimestampLog.frame_id_gaps`).
    mean_fps : float
        Mean frame rate, from host timestamps. ``0.0`` if fewer than 2
        frames.
    fps_jitter_percent : float
        Standard deviation of inter-frame intervals, as a percentage of
        the mean interval -- ``0.0`` for a perfectly regular frame rate,
        larger for an inconsistent one.
    mean_particle_count, min_particle_count, max_particle_count : float, int, int
        Particle-count statistics from classical blob detection
        (:func:`~glas.analysis.tracking_utils.detect_particles`) across
        the sampled frames.
    sampled_frame_count : int
        Number of frames particle counts were actually computed from
        (every frame for a short recording, a subsample for a long one).
    frames_with_no_particles : int
        Sampled frames with zero detections -- a recording that's mostly
        this is very likely pointed at the wrong thing.
    warnings : list of str
        Every problem found. Empty for a clean recording.
    """

    model_config = ConfigDict(frozen=True)

    frame_count: int = Field(ge=0)
    dropped_frame_count: int = Field(ge=0)
    frame_id_gaps: list[tuple[int, int]]
    mean_fps: float = Field(ge=0)
    fps_jitter_percent: float = Field(ge=0)
    mean_particle_count: float = Field(ge=0)
    min_particle_count: int = Field(ge=0)
    max_particle_count: int = Field(ge=0)
    sampled_frame_count: int = Field(ge=0)
    frames_with_no_particles: int = Field(ge=0)
    warnings: list[str]

    @property
    def is_clean(self) -> bool:
        """``True`` if no problems were found."""
        return not self.warnings


def assess_recording_quality(
    folder: Path,
    *,
    expected_fps: float | None = None,
    min_area: float = DEFAULT_MIN_AREA,
    max_sample_frames: int = DEFAULT_MAX_SAMPLE_FRAMES,
    fps_jitter_warning_percent: float = DEFAULT_FPS_JITTER_WARNING_PERCENT,
    expected_fps_tolerance_percent: float = DEFAULT_EXPECTED_FPS_TOLERANCE_PERCENT,
) -> RecordingQualityReport:
    """Assess whether a finalized recording is structurally sound and scientifically usable.

    Runs :func:`glas.dataset.validate_dataset` first (structural/checksum
    integrity); any errors it finds are folded into :attr:`RecordingQualityReport.warnings`
    rather than raised, so a checksum mismatch doesn't prevent the rest
    of this assessment from running if the frames are still readable.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder.
    expected_fps : float, optional
        If given, :attr:`RecordingQualityReport.mean_fps` more than
        ``expected_fps_tolerance_percent`` away from this triggers a
        warning.
    min_area : float, see :func:`~glas.analysis.tracking_utils.detect_particles`.
    max_sample_frames : int, default 200
        Particle counts are computed for at most this many frames,
        evenly spaced across the recording -- keeps this fast for very
        long recordings without needing every frame.
    fps_jitter_warning_percent : float, default 20.0
        :attr:`RecordingQualityReport.fps_jitter_percent` above this
        triggers a warning.
    expected_fps_tolerance_percent : float, default 10.0
        See ``expected_fps``.

    Returns
    -------
    RecordingQualityReport

    Raises
    ------
    DatasetError, DatasetFormatError, DatasetIOError
        Propagated from :func:`glas.dataset.iter_frames` if the frames
        themselves cannot be read at all.
    """
    warnings: list[str] = []
    validation = validate_dataset(folder)
    if not validation.valid:
        warnings.extend(f"Dataset integrity: {error}" for error in validation.errors)

    timestamp_log = TimestampLog()
    particle_counts: list[int] = []

    frames = list(iter_frames(folder))
    frame_count = len(frames)
    stride = max(1, frame_count // max_sample_frames) if frame_count else 1

    for index, frame in enumerate(frames):
        timestamp_log.append(frame)
        if index % stride == 0:
            particle_counts.append(len(detect_particles(frame.image, min_area=min_area)))

    dropped_frame_count = timestamp_log.dropped_frame_count()
    frame_id_gaps = timestamp_log.frame_id_gaps()
    if dropped_frame_count > 0:
        warnings.append(
            f"{dropped_frame_count} frame(s) dropped across {len(frame_id_gaps)} gap(s)."
        )

    intervals_ns = timestamp_log.intervals_ns()
    if len(intervals_ns) > 0:
        intervals_s = intervals_ns.astype(np.float64) / 1e9
        mean_interval_s = float(np.mean(intervals_s))
        mean_fps = 1.0 / mean_interval_s if mean_interval_s > 0 else 0.0
        jitter_percent = (
            float(np.std(intervals_s)) / mean_interval_s * 100.0 if mean_interval_s > 0 else 0.0
        )
    else:
        mean_fps = 0.0
        jitter_percent = 0.0
        warnings.append("Too few frames to compute a frame rate.")

    if jitter_percent > fps_jitter_warning_percent:
        warnings.append(f"High frame-rate jitter: {jitter_percent:.1f}%.")

    if expected_fps is not None and mean_fps > 0:
        deviation_percent = abs(mean_fps - expected_fps) / expected_fps * 100.0
        if deviation_percent > expected_fps_tolerance_percent:
            warnings.append(
                f"Mean frame rate {mean_fps:.2f} fps deviates {deviation_percent:.1f}% from "
                f"expected {expected_fps:.2f} fps."
            )

    frames_with_no_particles = sum(1 for count in particle_counts if count == 0)
    if frames_with_no_particles > 0:
        warnings.append(
            f"{frames_with_no_particles} of {len(particle_counts)} sampled frame(s) had no "
            "detected particles."
        )

    return RecordingQualityReport(
        frame_count=frame_count,
        dropped_frame_count=dropped_frame_count,
        frame_id_gaps=frame_id_gaps,
        mean_fps=mean_fps,
        fps_jitter_percent=jitter_percent,
        mean_particle_count=float(np.mean(particle_counts)) if particle_counts else 0.0,
        min_particle_count=min(particle_counts) if particle_counts else 0,
        max_particle_count=max(particle_counts) if particle_counts else 0,
        sampled_frame_count=len(particle_counts),
        frames_with_no_particles=frames_with_no_particles,
        warnings=warnings,
    )

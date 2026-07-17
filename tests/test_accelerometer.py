"""Tests for glas.accelerometer."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glas.accelerometer import (
    GRAVITY_M_S2,
    AccelerometerRecording,
    VibrationMetrics,
    analyze_vibration,
    compute_gamma,
    compute_vibration_amplitude,
    compute_vibration_frequency,
    import_accelerometer_csv,
    plot_vibration_signal,
    synchronize_with_frames,
)
from glas.exceptions import AccelerometerError
from glas.frame import Frame


def _sinusoidal_recording(
    *,
    frequency_hz: float = 50.0,
    amplitude_m: float = 1e-4,
    sample_rate_hz: float = 5000.0,
    duration_s: float = 0.5,
) -> AccelerometerRecording:
    omega = 2 * math.pi * frequency_hz
    times_s = np.arange(0, duration_s, 1 / sample_rate_hz)
    acceleration_m_s2 = amplitude_m * omega**2 * np.sin(omega * times_s)
    return AccelerometerRecording(times_s=times_s, acceleration_g=acceleration_m_s2 / GRAVITY_M_S2)


def _write_csv(
    path: Path,
    rows: list[tuple[float, float]],
    *,
    time_column: str = "time_s",
    value_column: str = "voltage_v",
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([time_column, value_column])
        for time_value, measurement in rows:
            writer.writerow([time_value, measurement])


class TestImportAccelerometerCsv:
    def test_converts_volts_to_g_using_sensitivity(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        # Sensitivity 10 mV/g -> 0.01V == 1g.
        _write_csv(path, [(0.0, 0.0), (0.001, 0.01), (0.002, -0.01)])

        recording = import_accelerometer_csv(path, sensitivity_mv_per_g=10.0)

        assert isinstance(recording, AccelerometerRecording)
        assert recording.acceleration_g == pytest.approx([0.0, 1.0, -1.0])
        assert recording.times_s == pytest.approx([0.0, 0.001, 0.002])

    def test_g_units_pass_through_unconverted(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.5), (0.001, -0.5)], value_column="acceleration_g")

        recording = import_accelerometer_csv(path, value_column="acceleration_g", value_units="g")

        assert recording.acceleration_g == pytest.approx([0.5, -0.5])

    def test_custom_column_names(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.01), (0.001, 0.02)], time_column="t", value_column="v")

        recording = import_accelerometer_csv(path, time_column="t", value_column="v")
        assert recording.times_s == pytest.approx([0.0, 0.001])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(tmp_path / "does_not_exist.csv")

    def test_missing_time_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.0), (0.001, 0.0)])
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(path, time_column="not_a_column")

    def test_missing_value_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.0), (0.001, 0.0)])
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(path, value_column="not_a_column")

    def test_non_numeric_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "voltage_v"])
            writer.writerow([0.0, "not-a-number"])
            writer.writerow([0.001, 0.01])
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(path)

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.0)])
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(path)

    def test_non_increasing_timestamps_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.0), (0.0, 0.01), (0.001, 0.0)])
        with pytest.raises(AccelerometerError):
            import_accelerometer_csv(path)

    def test_rejects_non_positive_sensitivity(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        _write_csv(path, [(0.0, 0.0), (0.001, 0.01)])
        with pytest.raises(ValueError):
            import_accelerometer_csv(path, sensitivity_mv_per_g=0.0)


class TestComputeVibrationFrequency:
    def test_recovers_known_frequency(self) -> None:
        recording = _sinusoidal_recording(frequency_hz=60.0)
        frequency = compute_vibration_frequency(recording)
        assert frequency == pytest.approx(60.0, abs=2.0)

    def test_too_few_samples_raises(self) -> None:
        recording = AccelerometerRecording(
            times_s=np.array([0.0, 0.001, 0.002]),
            acceleration_g=np.array([0.0, 1.0, -1.0]),
        )
        with pytest.raises(AccelerometerError):
            compute_vibration_frequency(recording)


class TestComputeGamma:
    def test_matches_analytic_peak_acceleration(self) -> None:
        frequency_hz = 50.0
        amplitude_m = 2e-4
        recording = _sinusoidal_recording(frequency_hz=frequency_hz, amplitude_m=amplitude_m)

        omega = 2 * math.pi * frequency_hz
        expected_gamma = amplitude_m * omega**2 / GRAVITY_M_S2

        assert compute_gamma(recording) == pytest.approx(expected_gamma, rel=1e-6)

    def test_zero_signal_gives_zero_gamma(self) -> None:
        recording = AccelerometerRecording(
            times_s=np.array([0.0, 0.001, 0.002, 0.003]),
            acceleration_g=np.zeros(4),
        )
        assert compute_gamma(recording) == 0.0


class TestComputeVibrationAmplitude:
    def test_recovers_known_amplitude(self) -> None:
        frequency_hz = 50.0
        amplitude_m = 1.5e-4
        recording = _sinusoidal_recording(frequency_hz=frequency_hz, amplitude_m=amplitude_m)

        recovered = compute_vibration_amplitude(recording, frequency_hz)
        assert recovered == pytest.approx(amplitude_m, rel=1e-6)

    def test_rejects_non_positive_frequency(self) -> None:
        recording = _sinusoidal_recording()
        with pytest.raises(ValueError):
            compute_vibration_amplitude(recording, 0.0)
        with pytest.raises(ValueError):
            compute_vibration_amplitude(recording, -1.0)


class TestPlotVibrationSignal:
    def test_produces_a_valid_png(self, tmp_path: Path) -> None:
        recording = _sinusoidal_recording()
        output = tmp_path / "signal.png"

        result = plot_vibration_signal(recording, output)

        assert result == output
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        recording = _sinusoidal_recording()
        output = tmp_path / "nested" / "dir" / "signal.png"

        plot_vibration_signal(recording, output)
        assert output.is_file()


class TestAnalyzeVibration:
    def test_computes_metrics_end_to_end(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        frequency_hz = 50.0
        amplitude_m = 1e-4
        omega = 2 * math.pi * frequency_hz
        sample_rate_hz = 5000.0
        times_s = np.arange(0, 0.5, 1 / sample_rate_hz)
        acceleration_g = (amplitude_m * omega**2 * np.sin(omega * times_s)) / GRAVITY_M_S2
        _write_csv(
            path,
            list(zip(times_s.tolist(), acceleration_g.tolist(), strict=True)),
            value_column="acceleration_g",
        )

        metrics = analyze_vibration(path, value_column="acceleration_g", value_units="g")

        assert isinstance(metrics, VibrationMetrics)
        assert metrics.frequency_hz == pytest.approx(frequency_hz, abs=2.0)
        assert metrics.amplitude_m == pytest.approx(amplitude_m, rel=0.05)
        assert metrics.peak_acceleration_g == metrics.gamma

    def test_optionally_writes_a_plot(self, tmp_path: Path) -> None:
        path = tmp_path / "accel.csv"
        recording = _sinusoidal_recording()
        _write_csv(
            path,
            list(zip(recording.times_s.tolist(), recording.acceleration_g.tolist(), strict=True)),
            value_column="acceleration_g",
        )
        plot_path = tmp_path / "plot.png"

        analyze_vibration(path, value_column="acceleration_g", value_units="g", plot_path=plot_path)
        assert plot_path.is_file()


class TestSynchronizeWithFrames:
    def _make_frames(self, host_timestamps_ns: list[int]) -> list[Frame]:
        return [
            Frame(
                frame_id=i,
                image=np.zeros((2, 2), dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=ts,
                device_timestamp_ticks=i,
            )
            for i, ts in enumerate(host_timestamps_ns)
        ]

    def test_matches_nearest_sample(self) -> None:
        recording = AccelerometerRecording(
            times_s=np.array([0.0, 1.0, 2.0, 3.0]),
            acceleration_g=np.array([10.0, 20.0, 30.0, 40.0]),
        )
        # Frames at 0.0s, 0.1s, 1.4s, 1.6s, 3.0s relative time.
        frames = self._make_frames([0, 100_000_000, 1_400_000_000, 1_600_000_000, 3_000_000_000])

        result = synchronize_with_frames(recording, frames)

        assert result == pytest.approx([10.0, 10.0, 20.0, 30.0, 40.0])

    def test_offset_shifts_alignment(self) -> None:
        recording = AccelerometerRecording(
            times_s=np.array([0.0, 1.0, 2.0]),
            acceleration_g=np.array([10.0, 20.0, 30.0]),
        )
        frames = self._make_frames([0])

        # offset_s=1.0 shifts the accelerometer timeline forward by 1s,
        # so frame 0 (relative time 0) now lines up with the sample that
        # was originally at relative time -1.0 -> clamped to the first
        # sample once shifted, i.e. accel_relative = [1.0, 2.0, 3.0],
        # nearest to frame time 0.0 is 1.0 -> value 10.0.
        result = synchronize_with_frames(recording, frames, offset_s=1.0)
        assert result == pytest.approx([10.0])

    def test_empty_frames_raises(self) -> None:
        recording = _sinusoidal_recording()
        with pytest.raises(ValueError):
            synchronize_with_frames(recording, [])

    def test_output_length_matches_frame_count(self) -> None:
        recording = _sinusoidal_recording()
        frames = self._make_frames([0, 50_000_000, 100_000_000])
        result = synchronize_with_frames(recording, frames)
        assert result.shape == (3,)

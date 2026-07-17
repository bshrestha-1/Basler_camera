"""Tests for glas.analysis.comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from glas.analysis.comparison import (
    ParameterSweepResult,
    compare_runs,
    export_sweep_csv,
    plot_parameter_sweep,
)
from glas.exceptions import BrazilNutError
from glas.experiment import ExperimentSummary, get_physical_parameters
from glas.metadata import DatasetMetadata


def _make_summary(run_id: str, gamma: float | None, folder: Path) -> ExperimentSummary:
    extra = {}
    if gamma is not None:
        extra = {"physical_parameters": {"target_acceleration_g": gamma}}
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=10,
        height=10,
        created_at_utc="2026-01-01T00:00:00+00:00",
        extra=extra,
    )
    return ExperimentSummary(
        folder=folder,
        run_id=run_id,
        name="",
        tags=[],
        notes="",
        created_at_utc=metadata.created_at_utc,
        frame_count=10,
        camera_model=metadata.camera_model,
        metadata=metadata,
    )


def _parameter_fn(md: DatasetMetadata) -> float | None:
    return get_physical_parameters(md).target_acceleration_g


class TestCompareRuns:
    def test_groups_runs_by_parameter_value(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
            _make_summary("Run0003", 2.0, tmp_path / "c"),
        ]
        metrics = {"a": 5.0, "b": 5.5, "c": 8.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        assert len(result.points) == 2
        assert result.points[0].parameter_value == pytest.approx(1.0)
        assert result.points[0].run_ids == ["Run0001", "Run0002"]
        assert result.points[1].parameter_value == pytest.approx(2.0)

    def test_points_are_sorted_by_parameter_value(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 3.0, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
            _make_summary("Run0003", 2.0, tmp_path / "c"),
        ]
        metrics = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        values = [point.parameter_value for point in result.points]
        assert values == sorted(values)

    def test_skips_runs_with_no_parameter_value(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", None, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
        ]
        metrics = {"a": 1.0, "b": 2.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        all_run_ids = [run_id for point in result.points for run_id in point.run_ids]
        assert all_run_ids == ["Run0002"]

    def test_skips_runs_whose_metric_extraction_raises_glas_error(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
        ]

        def metric_fn(folder: Path) -> float:
            if folder.name == "a":
                raise BrazilNutError("no intruder found")
            return 5.0

        result = compare_runs(summaries, _parameter_fn, metric_fn, compute_fit=False)
        all_run_ids = [run_id for point in result.points for run_id in point.run_ids]
        assert all_run_ids == ["Run0002"]

    def test_non_glas_error_propagates(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", 1.0, tmp_path / "a")]

        def metric_fn(folder: Path) -> float:
            raise KeyError("bug")

        with pytest.raises(KeyError):
            compare_runs(summaries, _parameter_fn, metric_fn)

    def test_raises_value_error_when_nothing_matches(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", None, tmp_path / "a")]
        with pytest.raises(ValueError, match="nothing to compare"):
            compare_runs(summaries, _parameter_fn, lambda folder: 1.0)

    def test_computes_correct_descriptive_stats_per_group(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
            _make_summary("Run0003", 1.0, tmp_path / "c"),
        ]
        metrics = {"a": 2.0, "b": 4.0, "c": 6.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        assert result.points[0].stats.n == 3
        assert result.points[0].stats.mean == pytest.approx(4.0)

    def test_fit_computed_with_enough_points(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary(f"Run{i:04d}", float(i), tmp_path / f"r{i}") for i in range(1, 5)
        ]
        metrics = {f"r{i}": 2.0 * i for i in range(1, 5)}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=True
        )
        assert result.fit is not None
        assert result.fit.slope == pytest.approx(2.0, abs=0.01)

    def test_fit_none_with_too_few_points(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 2.0, tmp_path / "b"),
        ]
        metrics = {"a": 1.0, "b": 2.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=True
        )
        assert result.fit is None

    def test_fit_none_when_compute_fit_false(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary(f"Run{i:04d}", float(i), tmp_path / f"r{i}") for i in range(1, 5)
        ]
        metrics = {f"r{i}": float(i) for i in range(1, 5)}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        assert result.fit is None

    def test_custom_axis_labels_are_recorded(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", 1.0, tmp_path / "a")]
        result = compare_runs(
            summaries,
            _parameter_fn,
            lambda folder: 1.0,
            parameter_name="Gamma",
            metric_name="Rise time (s)",
        )
        assert result.parameter_name == "Gamma"
        assert result.metric_name == "Rise time (s)"

    def test_returns_parameter_sweep_result(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", 1.0, tmp_path / "a")]
        result = compare_runs(summaries, _parameter_fn, lambda folder: 1.0)
        assert isinstance(result, ParameterSweepResult)


class TestPlotParameterSweep:
    def _make_result(self, tmp_path: Path) -> ParameterSweepResult:
        summaries = [
            _make_summary(f"Run{i:04d}", float(i), tmp_path / f"r{i}") for i in range(1, 5)
        ]
        metrics = {f"r{i}": 2.0 * i + 1.0 for i in range(1, 5)}
        return compare_runs(
            summaries,
            _parameter_fn,
            lambda folder: metrics[folder.name],
            parameter_name="Gamma",
            metric_name="Rise time (s)",
        )

    def test_writes_a_file(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path)
        output_path = tmp_path / "sweep.png"
        plot_parameter_sweep(result, output_path)
        assert output_path.exists()

    def test_returns_output_path(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path)
        output_path = tmp_path / "sweep.png"
        assert plot_parameter_sweep(result, output_path) == output_path

    def test_works_without_a_fit(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 2.0, tmp_path / "b"),
        ]
        metrics = {"a": 1.0, "b": 2.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        output_path = tmp_path / "sweep.png"
        plot_parameter_sweep(result, output_path, show_fit=True)
        assert output_path.exists()

    def test_show_fit_false_still_writes_a_file(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path)
        output_path = tmp_path / "sweep.png"
        plot_parameter_sweep(result, output_path, show_fit=False)
        assert output_path.exists()


class TestExportSweepCsv:
    def test_returns_row_count(self, tmp_path: Path) -> None:
        result_tmp_path = tmp_path
        summaries = [
            _make_summary(f"Run{i:04d}", float(i), result_tmp_path / f"r{i}") for i in range(1, 4)
        ]
        metrics = {f"r{i}": float(i) for i in range(1, 4)}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        output_path = tmp_path / "sweep.csv"
        row_count = export_sweep_csv(result, output_path)
        assert row_count == 3

    def test_writes_expected_header(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", 1.0, tmp_path / "a")]
        result = compare_runs(summaries, _parameter_fn, lambda folder: 5.0)
        output_path = tmp_path / "sweep.csv"
        export_sweep_csv(result, output_path)

        with output_path.open() as handle:
            header = next(csv.reader(handle))
        assert header == [
            "parameter_value",
            "n",
            "mean",
            "std",
            "sem",
            "ci_low",
            "ci_high",
            "run_ids",
        ]

    def test_run_ids_are_semicolon_joined(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary("Run0001", 1.0, tmp_path / "a"),
            _make_summary("Run0002", 1.0, tmp_path / "b"),
        ]
        metrics = {"a": 1.0, "b": 2.0}
        result = compare_runs(
            summaries, _parameter_fn, lambda folder: metrics[folder.name], compute_fit=False
        )
        output_path = tmp_path / "sweep.csv"
        export_sweep_csv(result, output_path)

        with output_path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["run_ids"] == "Run0001;Run0002"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        summaries = [_make_summary("Run0001", 1.0, tmp_path / "a")]
        result = compare_runs(summaries, _parameter_fn, lambda folder: 5.0)
        output_path = tmp_path / "nested" / "dir" / "sweep.csv"
        export_sweep_csv(result, output_path)
        assert output_path.exists()

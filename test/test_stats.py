"""Tests for glas.stats."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glas.stats import DescriptiveStats, LinearFitResult, describe, linear_fit


class TestDescribe:
    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            describe([])

    def test_rejects_confidence_level_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            describe([1.0, 2.0], confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            describe([1.0, 2.0], confidence_level=1.0)

    def test_mean_matches_numpy(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = describe(values)
        assert result.mean == pytest.approx(np.mean(values))

    def test_std_matches_sample_std(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = describe(values)
        assert result.std == pytest.approx(np.std(values, ddof=1))

    def test_sem_is_std_over_sqrt_n(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = describe(values)
        assert result.sem == pytest.approx(result.std / np.sqrt(len(values)))

    def test_n_matches_sample_size(self) -> None:
        result = describe([1.0, 2.0, 3.0])
        assert result.n == 3

    def test_confidence_interval_contains_mean(self) -> None:
        result = describe([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.ci_low < result.mean < result.ci_high

    def test_single_observation_has_zero_std_and_point_interval(self) -> None:
        result = describe([42.0])
        assert result.n == 1
        assert result.mean == pytest.approx(42.0)
        assert result.std == 0.0
        assert result.sem == 0.0
        assert result.ci_low == pytest.approx(42.0)
        assert result.ci_high == pytest.approx(42.0)

    def test_identical_values_have_zero_std(self) -> None:
        result = describe([5.0, 5.0, 5.0])
        assert result.std == pytest.approx(0.0)
        assert result.sem == pytest.approx(0.0)

    def test_identical_values_collapse_interval_to_the_mean_not_nan(self) -> None:
        result = describe([5.0, 5.0, 5.0])
        assert result.ci_low == pytest.approx(5.0)
        assert result.ci_high == pytest.approx(5.0)
        assert not math.isnan(result.ci_low)
        assert not math.isnan(result.ci_high)

    def test_wider_interval_for_higher_confidence_level(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        narrow = describe(values, confidence_level=0.80)
        wide = describe(values, confidence_level=0.99)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_confidence_level_is_recorded(self) -> None:
        result = describe([1.0, 2.0, 3.0], confidence_level=0.90)
        assert result.confidence_level == pytest.approx(0.90)

    def test_returns_descriptive_stats(self) -> None:
        assert isinstance(describe([1.0, 2.0]), DescriptiveStats)

    def test_accepts_a_numpy_array(self) -> None:
        result = describe(np.array([1.0, 2.0, 3.0]))
        assert result.n == 3


class TestLinearFit:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            linear_fit([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            linear_fit([1.0, 2.0], [1.0, 2.0])

    def test_recovers_exact_line(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [2.0 * xi + 1.0 for xi in x]
        result = linear_fit(x, y)
        assert result.slope == pytest.approx(2.0)
        assert result.intercept == pytest.approx(1.0)
        assert result.r_squared == pytest.approx(1.0)

    def test_zero_stderr_for_exact_line(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y = [2.0 * xi + 1.0 for xi in x]
        result = linear_fit(x, y)
        assert result.slope_stderr == pytest.approx(0.0, abs=1e-9)

    def test_flat_line_has_zero_slope(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0, 5.0]
        result = linear_fit(x, y)
        assert result.slope == pytest.approx(0.0, abs=1e-9)
        assert result.intercept == pytest.approx(5.0)

    def test_noisy_data_yields_r_squared_below_one(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 30)
        y = 3.0 * x + 2.0 + rng.normal(scale=5.0, size=x.size)
        result = linear_fit(x, y)
        assert 0.0 <= result.r_squared < 1.0

    def test_predict_evaluates_the_fitted_line(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0, 3.0, 5.0, 7.0]
        result = linear_fit(x, y)
        assert result.predict(10.0) == pytest.approx(21.0)

    def test_flat_line_yields_finite_values_not_nan(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0, 5.0]
        result = linear_fit(x, y)
        assert result.r_squared == pytest.approx(1.0)
        assert result.slope_stderr == pytest.approx(0.0)
        assert not math.isnan(result.p_value)

    def test_p_value_small_for_strong_linear_relationship(self) -> None:
        x = list(range(20))
        y = [2.0 * xi + 1.0 for xi in x]
        result = linear_fit(x, y)
        assert result.p_value < 0.001

    def test_returns_linear_fit_result(self) -> None:
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        assert isinstance(linear_fit(x, y), LinearFitResult)

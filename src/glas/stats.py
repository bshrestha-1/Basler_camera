"""Descriptive statistics and linear fits for turning repeated-trial data into publishable numbers.

A single recording's analysis (:mod:`glas.analysis`) gives point
estimates -- one rise time, one packing fraction. Publishable results
need uncertainty on those estimates and, for a parameter sweep
(:mod:`glas.analysis.comparison`), a fit through them. This module is
the thin, correctly-implemented statistics layer under both: sample
mean/std/standard error/confidence interval (:func:`describe`, using
Student's t distribution rather than a fixed z-score, since granular
experiments typically have few repeated trials per condition) and
ordinary least-squares linear regression (:func:`linear_fit`).

Built on ``scipy.stats`` rather than hand-rolled formulas -- unlike
particle linking (see :mod:`glas.analysis.tracking_utils`), there is no
simpler correct alternative to the t-distribution and least-squares
regression that established statistics libraries already implement
carefully.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as _scipy_stats


class DescriptiveStats(BaseModel):
    """Sample mean, standard deviation, standard error, and confidence interval.

    Attributes
    ----------
    n : int
        Number of observations. At least 1.
    mean : float
        Sample mean.
    std : float
        Sample standard deviation (``ddof=1``). ``0.0`` if ``n == 1``.
    sem : float
        Standard error of the mean (``std / sqrt(n)``). ``0.0`` if
        ``n == 1``.
    ci_low, ci_high : float
        Confidence interval bounds for the mean, via Student's t
        distribution with ``n - 1`` degrees of freedom. Equal to
        ``mean`` (a zero-width interval) if ``n == 1``, since no
        variability can be estimated from a single observation.
    confidence_level : float
        Confidence level used for ``ci_low``/``ci_high``, e.g. ``0.95``
        for a 95% confidence interval.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=1)
    mean: float
    std: float = Field(ge=0)
    sem: float = Field(ge=0)
    ci_low: float
    ci_high: float
    confidence_level: float = Field(gt=0, lt=1)


def describe(values: Sequence[float], *, confidence_level: float = 0.95) -> DescriptiveStats:
    """Compute descriptive statistics for a sample of repeated-trial measurements.

    Parameters
    ----------
    values : sequence of float
        Repeated measurements of the same quantity (e.g. rise time
        across several trials at the same target Gamma). Must be
        non-empty.
    confidence_level : float, default 0.95
        Confidence level for the mean's confidence interval, in
        ``(0, 1)``.

    Returns
    -------
    DescriptiveStats

    Raises
    ------
    ValueError
        If ``values`` is empty, or ``confidence_level`` is not in
        ``(0, 1)``.
    """
    if len(values) == 0:
        raise ValueError("Cannot compute statistics for an empty sample.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    array = list(values)
    n = len(array)
    mean = float(_scipy_stats.tmean(array))

    if n == 1:
        return DescriptiveStats(
            n=n,
            mean=mean,
            std=0.0,
            sem=0.0,
            ci_low=mean,
            ci_high=mean,
            confidence_level=confidence_level,
        )

    std = float(_scipy_stats.tstd(array))
    sem = float(_scipy_stats.sem(array))
    if sem == 0.0:
        # Every observation is identical: the interval collapses to a
        # single point rather than the 0/0 NaN scipy's t-distribution
        # would otherwise produce for a zero-scale distribution.
        ci_low, ci_high = mean, mean
    else:
        ci_low, ci_high = _scipy_stats.t.interval(confidence_level, df=n - 1, loc=mean, scale=sem)
    return DescriptiveStats(
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
    )


class LinearFitResult(BaseModel):
    """An ordinary least-squares linear fit ``y = slope * x + intercept``.

    Attributes
    ----------
    slope, intercept : float
        Fitted line parameters.
    slope_stderr, intercept_stderr : float
        Standard errors of ``slope``/``intercept``.
    r_squared : float
        Coefficient of determination, in ``[0, 1]``.
    p_value : float
        Two-sided p-value for the null hypothesis that ``slope == 0``.
    """

    model_config = ConfigDict(frozen=True)

    slope: float
    intercept: float
    slope_stderr: float = Field(ge=0)
    intercept_stderr: float = Field(ge=0)
    r_squared: float = Field(ge=0, le=1)
    p_value: float = Field(ge=0, le=1)

    def predict(self, x: float) -> float:
        """Evaluate the fitted line at ``x``."""
        return self.slope * x + self.intercept


def linear_fit(x: Sequence[float], y: Sequence[float]) -> LinearFitResult:
    """Fit a line to ``(x, y)`` pairs via ordinary least-squares regression.

    Parameters
    ----------
    x, y : sequence of float
        Same length, at least 2 points (a line needs two points to be
        well-defined, and standard errors need at least one degree of
        freedom beyond that).

    Returns
    -------
    LinearFitResult

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths, or fewer than 3
        points are given.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}.")
    if len(x) < 3:
        raise ValueError(f"linear_fit needs at least 3 points, got {len(x)}.")

    result = _scipy_stats.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    slope_stderr = float(result.stderr)
    intercept_stderr = float(result.intercept_stderr)
    r_squared = float(result.rvalue) ** 2
    p_value = float(result.pvalue)

    if math.isnan(r_squared):
        # y has zero variance (every point lies on the same horizontal
        # line): the correlation coefficient is formally 0/0. The fitted
        # line (slope 0, intercept = that constant value) still fits
        # every point exactly, so the degenerate-but-correct values are
        # a perfect fit with no uncertainty, not "undefined".
        slope_stderr = 0.0
        intercept_stderr = 0.0
        r_squared = 1.0
        p_value = 0.0

    return LinearFitResult(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_stderr,
        intercept_stderr=intercept_stderr,
        r_squared=r_squared,
        p_value=p_value,
    )

"""Multi-run comparison and parameter sweeps: the figures papers actually use.

Every earlier analysis module answers "what happened in this one
recording." A publishable result almost always needs the next step:
"how did this measurement change across many recordings at different
Gamma / fill depth / grain size" -- a parameter sweep, with repeated
trials at each condition averaged and given real uncertainty.

    ExperimentManager.search_experiments() -> compare_runs() -> ParameterSweepResult
                                                                        |
                                                            plot_parameter_sweep()

:func:`compare_runs` is deliberately generic rather than hardcoded to one
analysis: it takes a *parameter extractor* (how to read the independent
variable, typically a :class:`~glas.experiment.PhysicalParameters` field,
off each recording's metadata) and a *metric extractor* (how to compute
the dependent variable -- any existing ``analyze_*`` function's output,
e.g. ``lambda folder: analyze_brazil_nut(folder).rise_time_s``). GLAS
already has every individual analysis; this module's only job is
grouping, averaging, and plotting across many of them.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- see glas.analysis.brazil_nut
# for why this is always headless-safe, unconditionally, with no pre-flight check needed.
import matplotlib.pyplot as plt  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from glas.exceptions import GLASError  # noqa: E402
from glas.experiment import ExperimentSummary  # noqa: E402
from glas.logger import get_logger  # noqa: E402
from glas.metadata import DatasetMetadata  # noqa: E402
from glas.plotting import apply_publication_style, savefig_publication, style_axes  # noqa: E402
from glas.stats import DescriptiveStats, LinearFitResult, describe, linear_fit  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MIN_POINTS_FOR_FIT = 3

ParameterExtractor = Callable[[DatasetMetadata], "float | None"]
MetricExtractor = Callable[[Path], float]


class SweepPoint(BaseModel):
    """Every measurement at one value of the swept parameter.

    Attributes
    ----------
    parameter_value : float
        The independent variable's value shared by every run in this
        point (e.g. one target Gamma).
    run_ids : list of str
        Which runs (:attr:`~glas.experiment.ExperimentSummary.run_id`)
        contributed to this point, in the order their metrics were
        computed.
    metric_values : list of float
        The extracted metric from each contributing run, same order as
        ``run_ids``.
    stats : DescriptiveStats
        Summary statistics over ``metric_values``.
    """

    model_config = ConfigDict(frozen=True)

    parameter_value: float
    run_ids: list[str]
    metric_values: list[float]
    stats: DescriptiveStats


class ParameterSweepResult(BaseModel):
    """The result of comparing a metric across many recordings, grouped by a parameter.

    Attributes
    ----------
    parameter_name, metric_name : str
        Human-readable axis labels for :func:`plot_parameter_sweep`.
    points : list of SweepPoint
        One entry per distinct parameter value found, sorted by
        ``parameter_value`` ascending.
    fit : LinearFitResult, optional
        A linear fit through each point's mean metric value vs. its
        parameter value, if requested and at least
        :data:`DEFAULT_MIN_POINTS_FOR_FIT` points were found. ``None``
        otherwise.
    """

    model_config = ConfigDict(frozen=True)

    parameter_name: str
    metric_name: str
    points: list[SweepPoint] = Field(min_length=1)
    fit: LinearFitResult | None = None


def compare_runs(
    summaries: Sequence[ExperimentSummary],
    parameter_fn: ParameterExtractor,
    metric_fn: MetricExtractor,
    *,
    parameter_name: str = "parameter",
    metric_name: str = "metric",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    compute_fit: bool = True,
) -> ParameterSweepResult:
    """Group many recordings by a parameter value and summarize a metric within each group.

    Parameters
    ----------
    summaries : sequence of ExperimentSummary
        Typically the result of
        :meth:`~glas.experiment.ExperimentManager.search_experiments`.
    parameter_fn : callable
        ``(DatasetMetadata) -> float | None``, extracting the
        independent variable's value for one recording -- e.g.
        ``lambda md: get_physical_parameters(md).target_acceleration_g``.
        A recording ``parameter_fn`` returns ``None`` for is skipped
        (e.g. that field was never filled in for that recording).
    metric_fn : callable
        ``(pathlib.Path) -> float``, computing the dependent variable
        from a recording's dataset folder -- e.g. ``lambda folder:
        analyze_brazil_nut(folder).rise_time_s``. If this raises a
        :class:`~glas.exceptions.GLASError` (the recording has too few
        particles, too few frames, etc.), that recording is skipped with
        a logged warning rather than aborting the whole sweep -- a
        parameter sweep across dozens of recordings shouldn't fail
        outright because one of them is unusable.
    parameter_name, metric_name : str
        Human-readable labels, carried through to
        :attr:`ParameterSweepResult.parameter_name`/``metric_name`` for
        :func:`plot_parameter_sweep`.
    confidence_level : float, default 0.95
        See :func:`glas.stats.describe`.
    compute_fit : bool, default True
        Whether to compute :attr:`ParameterSweepResult.fit`.

    Returns
    -------
    ParameterSweepResult

    Raises
    ------
    ValueError
        If no recording yields both a parameter value and a metric value
        (nothing to compare).
    """
    grouped: dict[float, list[tuple[str, float]]] = {}
    for summary in summaries:
        parameter_value = parameter_fn(summary.metadata)
        if parameter_value is None:
            continue
        try:
            metric_value = metric_fn(summary.folder)
        except GLASError as exc:
            logger.warning("Skipping %s in comparison: %s", summary.run_id, exc)
            continue
        grouped.setdefault(parameter_value, []).append((summary.run_id, metric_value))

    if not grouped:
        raise ValueError(
            "No recording yielded both a parameter value and a metric value -- nothing to compare."
        )

    points = [
        SweepPoint(
            parameter_value=parameter_value,
            run_ids=[run_id for run_id, _ in entries],
            metric_values=[value for _, value in entries],
            stats=describe([value for _, value in entries], confidence_level=confidence_level),
        )
        for parameter_value, entries in sorted(grouped.items())
    ]

    fit: LinearFitResult | None = None
    if compute_fit and len(points) >= DEFAULT_MIN_POINTS_FOR_FIT:
        fit = linear_fit(
            [point.parameter_value for point in points], [point.stats.mean for point in points]
        )

    return ParameterSweepResult(
        parameter_name=parameter_name, metric_name=metric_name, points=points, fit=fit
    )


def plot_parameter_sweep(
    result: ParameterSweepResult, output_path: Path, *, show_fit: bool = True
) -> Path:
    """Plot a parameter sweep's mean metric vs. parameter value, with confidence intervals.

    Each point is the group mean with error bars spanning its confidence
    interval (:attr:`~glas.stats.DescriptiveStats.ci_low`/``ci_high``);
    the optional fit line is annotated with its R^2.

    Parameters
    ----------
    result : ParameterSweepResult
    output_path : pathlib.Path
        Destination file. Parent directories are created if missing.
    show_fit : bool, default True
        Draw :attr:`ParameterSweepResult.fit`, if present.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    apply_publication_style()

    x = [point.parameter_value for point in result.points]
    y = [point.stats.mean for point in result.points]
    y_low = [point.stats.mean - point.stats.ci_low for point in result.points]
    y_high = [point.stats.ci_high - point.stats.mean for point in result.points]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        x, y, yerr=[y_low, y_high], marker="o", markersize=5, capsize=3, linestyle="-", linewidth=1
    )

    if show_fit and result.fit is not None:
        x_line = [min(x), max(x)]
        y_line = [result.fit.predict(value) for value in x_line]
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            color="gray",
            label=f"fit: R²={result.fit.r_squared:.3f}",
        )
        ax.legend()

    ax.set_xlabel(result.parameter_name)
    ax.set_ylabel(result.metric_name)
    ax.set_title(f"{result.metric_name} vs. {result.parameter_name}")
    style_axes(ax)

    fig.tight_layout()
    return savefig_publication(fig, output_path)


def export_sweep_csv(result: ParameterSweepResult, output_path: Path) -> int:
    """Write one summary row per parameter value to a CSV file.

    Parameters
    ----------
    result : ParameterSweepResult
    output_path : pathlib.Path
        Destination CSV file. Parent directories are created if missing;
        an existing file is overwritten.

    Returns
    -------
    int
        Number of rows written (one per :attr:`ParameterSweepResult.points` entry).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["parameter_value", "n", "mean", "std", "sem", "ci_low", "ci_high", "run_ids"]
    count = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in result.points:
            writer.writerow(
                {
                    "parameter_value": point.parameter_value,
                    "n": point.stats.n,
                    "mean": point.stats.mean,
                    "std": point.stats.std,
                    "sem": point.stats.sem,
                    "ci_low": point.stats.ci_low,
                    "ci_high": point.stats.ci_high,
                    "run_ids": ";".join(point.run_ids),
                }
            )
            count += 1
    return count

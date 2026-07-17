"""Segregation analysis: segregation index, mixing index, and mixing entropy.

Measures how separated two particle populations are -- the standard
diagnostic for size segregation in a vibrated bidisperse granular bed
(e.g. large particles rising to the surface, small particles sifting to
the bottom). Particles are split into "large"/"small" populations by an
equivalent-radius threshold, then binned onto a coarse spatial grid to
measure how each cell's local composition compares to the bed's overall
composition::

    glas.dataset.iter_frames() -> detect_particles() -> split by radius
                                                                |
                                             compute_segregation_metrics() -> SegregationMetrics
                                                                |
                                                  plot_segregation_summary()

:func:`analyze_segregation` runs the whole pipeline over a finalized
dataset folder in one call, the same role :func:`glas.analysis.track_dataset`
and :func:`glas.analysis.analyze_packing` play for their own phases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- see glas.analysis.brazil_nut
# for why this is always headless-safe, unconditionally, with no pre-flight check needed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from glas.analysis.tracking_utils import (  # noqa: E402
    DEFAULT_MIN_AREA,
    Detection,
    detect_particles,
)
from glas.dataset import iter_frames  # noqa: E402
from glas.exceptions import SegregationError  # noqa: E402
from glas.metadata import load_metadata_json  # noqa: E402
from glas.plotting import apply_publication_style, savefig_publication, style_axes  # noqa: E402

DEFAULT_GRID_SPACING = 32
_METADATA_FILENAME = "metadata.json"


class SegregationMetrics(BaseModel):
    """Segregation statistics for a single frame.

    Attributes
    ----------
    frame_id : int
        The frame these statistics were computed from.
    large_count, small_count : int
        Number of particles classified as "large" and "small" (see
        :func:`analyze_segregation`'s ``size_threshold``).
    segregation_index : float
        How separated the two populations are spatially, in ``[0, 1]``:
        0 means the local composition of every sampled region matches
        the bed's overall composition (no segregation), 1 means every
        sampled region is spatially pure (fully segregated). The
        complement of :attr:`mixing_index`.
    mixing_index : float
        Lacey's mixing index, in ``[0, 1]``: 0 for a fully segregated bed,
        1 for a bed mixed as well as random chance allows. ``1 -
        segregation_index``.
    mixing_entropy : float
        Mean local Shannon entropy of composition, normalized by the
        entropy of the bed's overall composition, in ``[0, 1]``: 0 when
        every sampled region is spatially pure, 1 when every region's
        local composition matches the bed's overall composition. Unlike
        :attr:`mixing_index`, this is not a linear function of local
        variance, so it can disagree with it in its details even though
        both move in the same direction.
    """

    model_config = ConfigDict(frozen=True)

    frame_id: int
    large_count: int
    small_count: int
    segregation_index: float
    mixing_index: float
    mixing_entropy: float


class SegregationSummary(BaseModel):
    """Segregation statistics over time across a whole recording.

    Attributes
    ----------
    frame_ids : list of int
        Frame each entry in :attr:`metrics` came from, in order.
    times_s : list of float
        Elapsed time since the first frame, in seconds.
    metrics : list of SegregationMetrics
        Per-frame segregation statistics, same order as :attr:`frame_ids`.
    """

    model_config = ConfigDict(frozen=True)

    frame_ids: list[int]
    times_s: list[float]
    metrics: list[SegregationMetrics]


def _bin_counts(
    detections: Sequence[Detection], *, n_rows: int, n_cols: int, grid_spacing: int
) -> np.ndarray:
    """Count detections per grid cell, by centroid (see compute_packing_field)."""
    counts = np.zeros((n_rows, n_cols), dtype=np.float64)
    for detection in detections:
        col = min(max(int(detection.x // grid_spacing), 0), n_cols - 1)
        row = min(max(int(detection.y // grid_spacing), 0), n_rows - 1)
        counts[row, col] += 1
    return counts


def compute_segregation_metrics(
    large: Sequence[Detection],
    small: Sequence[Detection],
    *,
    image_width: int,
    image_height: int,
    grid_spacing: int = DEFAULT_GRID_SPACING,
    frame_id: int = 0,
) -> SegregationMetrics:
    """Compute segregation index, mixing index, and mixing entropy for one frame.

    Bins both populations onto the same coarse grid (``grid_spacing``
    pixels per cell, by detection centroid) and compares each occupied
    cell's local fraction of "large" particles against the bed's overall
    fraction, via Lacey's mixing index and a normalized Shannon entropy.

    Parameters
    ----------
    large, small : sequence of Detection
        The two particle populations for one frame (see
        :func:`analyze_segregation`'s ``size_threshold`` for how to split
        a frame's detections into these).
    image_width, image_height : int
        Frame dimensions, in pixels.
    grid_spacing : int, default 32
        Pixel size of each sampling cell. Needs to be large enough that
        most occupied cells contain more than one particle -- see the
        ``SegregationError`` documented below.
    frame_id : int, default 0
        Recorded onto the returned metrics as
        :attr:`SegregationMetrics.frame_id`.

    Returns
    -------
    SegregationMetrics

    Raises
    ------
    ValueError
        If ``grid_spacing``, ``image_width``, or ``image_height`` is not
        at least 1.
    SegregationError
        If ``large`` and ``small`` are both empty (nothing to analyze),
        or if every occupied cell contains at most one particle on
        average -- at that resolution, a fully segregated bed and a
        randomly mixed one look statistically identical (every occupied
        cell is trivially "pure" when it holds only one particle), so
        Lacey's index cannot distinguish them. Use a larger
        ``grid_spacing``.
    """
    if grid_spacing < 1:
        raise ValueError(f"grid_spacing must be at least 1, got {grid_spacing}.")
    if image_width < 1:
        raise ValueError(f"image_width must be at least 1, got {image_width}.")
    if image_height < 1:
        raise ValueError(f"image_height must be at least 1, got {image_height}.")

    total = len(large) + len(small)
    if total == 0:
        raise SegregationError("Cannot compute segregation metrics: no particles in this frame.")

    p = len(large) / total

    if p == 0.0 or p == 1.0:
        # Only one population is present: there is nothing to segregate
        # from, so by convention this counts as fully mixed (no
        # separation possible) with zero entropy (a single outcome).
        return SegregationMetrics(
            frame_id=frame_id,
            large_count=len(large),
            small_count=len(small),
            segregation_index=0.0,
            mixing_index=1.0,
            mixing_entropy=0.0,
        )

    n_cols = math.ceil(image_width / grid_spacing)
    n_rows = math.ceil(image_height / grid_spacing)
    large_counts = _bin_counts(large, n_rows=n_rows, n_cols=n_cols, grid_spacing=grid_spacing)
    small_counts = _bin_counts(small, n_rows=n_rows, n_cols=n_cols, grid_spacing=grid_spacing)
    cell_totals = large_counts + small_counts

    occupied = cell_totals > 0
    concentrations = large_counts[occupied] / cell_totals[occupied]
    mean_n = float(cell_totals[occupied].mean())

    variance = float(np.mean((concentrations - p) ** 2))
    fully_segregated_variance = p * (1 - p)
    random_mix_variance = fully_segregated_variance / mean_n

    denominator = fully_segregated_variance - random_mix_variance
    if denominator <= 0:
        raise SegregationError(
            f"Cannot distinguish segregation from randomness: occupied cells contain "
            f"{mean_n:.2f} particle(s) on average with grid_spacing={grid_spacing}. "
            "Use a larger grid_spacing so each cell samples multiple particles."
        )

    mixing_index = min(1.0, max(0.0, (fully_segregated_variance - variance) / denominator))
    segregation_index = 1.0 - mixing_index

    overall_entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    pure = (concentrations <= 0) | (concentrations >= 1)
    safe_concentrations = np.where(pure, 0.5, concentrations)  # avoid log(0); result discarded
    local_entropies = np.where(
        pure,
        0.0,
        -(
            safe_concentrations * np.log(safe_concentrations)
            + (1 - safe_concentrations) * np.log(1 - safe_concentrations)
        ),
    )
    mixing_entropy = min(1.0, max(0.0, float(local_entropies.mean()) / overall_entropy))

    return SegregationMetrics(
        frame_id=frame_id,
        large_count=len(large),
        small_count=len(small),
        segregation_index=segregation_index,
        mixing_index=mixing_index,
        mixing_entropy=mixing_entropy,
    )


def plot_segregation_summary(summary: SegregationSummary, output_path: Path) -> Path:
    """Plot segregation/mixing index and mixing entropy over time, saved as a PNG.

    Two stacked panels: segregation index and mixing index together
    (they're complements of each other, so plotting them on the same
    axes makes the relationship visible), and mixing entropy. Uses
    matplotlib's non-interactive ``"Agg"`` backend, so this never
    requires or attempts to open a display.

    Parameters
    ----------
    summary : SegregationSummary
        Summary to plot, from :func:`analyze_segregation`.
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    apply_publication_style()

    segregation = [m.segregation_index for m in summary.metrics]
    mixing = [m.mixing_index for m in summary.metrics]
    entropy = [m.mixing_entropy for m in summary.metrics]

    fig, (index_ax, entropy_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    index_ax.plot(summary.times_s, segregation, marker="o", markersize=3, label="segregation index")
    index_ax.plot(summary.times_s, mixing, marker="o", markersize=3, label="mixing index")
    index_ax.set_ylabel("Index")
    index_ax.set_ylim(-0.05, 1.05)
    index_ax.set_title("Segregation analysis")
    index_ax.legend()
    style_axes(index_ax)

    entropy_ax.plot(summary.times_s, entropy, marker="o", markersize=3, color="black")
    entropy_ax.set_ylabel("Mixing entropy")
    entropy_ax.set_xlabel("Time (s)")
    entropy_ax.set_ylim(-0.05, 1.05)
    style_axes(entropy_ax)

    fig.tight_layout()
    return savefig_publication(fig, output_path)


def analyze_segregation(
    folder: Path,
    *,
    size_threshold: float | None = None,
    grid_spacing: int = DEFAULT_GRID_SPACING,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
    plot_path: Path | None = None,
) -> SegregationSummary:
    """Detect particles and compute segregation statistics for every frame in a recorded dataset.

    Convenience wrapper combining :func:`glas.dataset.iter_frames` and
    :func:`glas.analysis.detect_particles` with :func:`compute_segregation_metrics`
    (and, optionally, :func:`plot_segregation_summary`) into one call over
    a dataset folder.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    size_threshold : float, optional
        Equivalent-radius threshold, in pixels, splitting each frame's
        detections into "large" (``radius >= size_threshold``) and
        "small" (``radius < size_threshold``) populations. ``None`` (the
        default) uses the median radius across every detection in the
        whole dataset -- computed once and applied consistently to every
        frame, so a particle's classification doesn't drift from frame to
        frame due to detection noise.
    grid_spacing, min_area, max_area, threshold, invert : see
        :func:`compute_segregation_metrics` and
        :func:`glas.analysis.detect_particles`.
    plot_path : pathlib.Path, optional
        If given, also save a segregation/mixing/entropy plot here (see
        :func:`plot_segregation_summary`).

    Returns
    -------
    SegregationSummary

    Raises
    ------
    SegregationError
        If the dataset has no frames, if ``size_threshold`` is ``None``
        and no particles were detected anywhere in the dataset (so a
        threshold cannot be determined), or if any single frame fails
        :func:`compute_segregation_metrics` (e.g. only one population
        present in every frame is fine, but a genuinely empty frame, or a
        ``grid_spacing`` too fine for the particle density, is not).
    """
    metadata = load_metadata_json(folder / _METADATA_FILENAME)

    frames_data = [
        (
            frame,
            detect_particles(
                frame.image,
                min_area=min_area,
                max_area=max_area,
                threshold=threshold,
                invert=invert,
            ),
        )
        for frame in iter_frames(folder)
    ]
    if not frames_data:
        raise SegregationError("Cannot analyze segregation: dataset has no frames.")

    if size_threshold is None:
        all_radii = [d.radius for _, dets in frames_data for d in dets]
        if not all_radii:
            raise SegregationError(
                "Cannot analyze segregation: no particles were detected in the dataset, so "
                "an automatic size threshold could not be determined. Pass size_threshold "
                "explicitly."
            )
        size_threshold = float(np.median(all_radii))

    frame_ids: list[int] = []
    times_s: list[float] = []
    metrics: list[SegregationMetrics] = []
    start_ns = frames_data[0][0].host_timestamp_ns

    for frame, detections in frames_data:
        large = [d for d in detections if d.radius >= size_threshold]
        small = [d for d in detections if d.radius < size_threshold]
        frame_metrics = compute_segregation_metrics(
            large,
            small,
            image_width=metadata.width,
            image_height=metadata.height,
            grid_spacing=grid_spacing,
            frame_id=frame.frame_id,
        )

        frame_ids.append(frame.frame_id)
        times_s.append((frame.host_timestamp_ns - start_ns) / 1e9)
        metrics.append(frame_metrics)

    summary = SegregationSummary(frame_ids=frame_ids, times_s=times_s, metrics=metrics)

    if plot_path is not None:
        plot_segregation_summary(summary, plot_path)

    return summary

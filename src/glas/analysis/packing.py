"""Packing analysis: packing fraction, void fraction, number density, and spatial fields.

Builds directly on :mod:`glas.analysis.tracking_utils`' particle detection
(no tracking needed -- packing statistics are per-frame, not per-trajectory)::

    glas.dataset.iter_frames() -> detect_particles() -> compute_packing_metrics() -> PackingMetrics
                                                              |
                                                   compute_packing_field() -> plot_packing_heatmap()

:func:`analyze_packing` runs the whole pipeline over a finalized dataset
folder in one call, the same role :func:`glas.analysis.track_dataset` and
:func:`glas.analysis.analyze_convection` play for their own phases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- see glas.analysis.brazil_nut
# for why this is always headless-safe, unconditionally, with no pre-flight check needed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from glas.analysis.tracking_utils import (  # noqa: E402
    DEFAULT_MIN_AREA,
    Detection,
    detect_particles,
)
from glas.dataset import iter_frames  # noqa: E402
from glas.exceptions import PackingError  # noqa: E402
from glas.metadata import load_metadata_json  # noqa: E402

DEFAULT_GRID_SPACING = 32
_METADATA_FILENAME = "metadata.json"


class PackingMetrics(BaseModel):
    """Packing statistics for a single frame.

    Attributes
    ----------
    frame_id : int
        The frame these statistics were computed from.
    particle_count : int
        Number of particles detected in the frame.
    packing_fraction : float
        Fraction of the region of interest covered by particles
        (``sum(particle areas) / roi_area``). Not clamped to ``[0, 1]``:
        overlapping or merged blob detections can legitimately push this
        above 1.0, which is a meaningful (if surprising) result rather
        than an error to guard against.
    void_fraction : float
        ``1 - packing_fraction`` -- the fraction of the region of
        interest *not* covered by particles.
    number_density : float
        Particles per unit area (``particle_count / roi_area``).
    """

    model_config = ConfigDict(frozen=True)

    frame_id: int
    particle_count: int
    packing_fraction: float
    void_fraction: float
    number_density: float


class PackingSummary(BaseModel):
    """Packing statistics over time across a whole recording.

    Attributes
    ----------
    frame_ids : list of int
        Frame each entry in :attr:`metrics` came from, in order.
    times_s : list of float
        Elapsed time since the first frame, in seconds.
    metrics : list of PackingMetrics
        Per-frame packing statistics, same order as :attr:`frame_ids`.
    """

    model_config = ConfigDict(frozen=True)

    frame_ids: list[int]
    times_s: list[float]
    metrics: list[PackingMetrics]


@dataclass(frozen=True)
class PackingField:
    """A spatial map of packing fraction across a single frame, binned onto a coarse grid.

    Like :class:`glas.analysis.convection.VelocityField`, this is a plain
    dataclass rather than a Pydantic model, since most fields are numpy
    arrays (see ``Frame``'s docstring for the identical reasoning).

    Attributes
    ----------
    frame_id : int
        The frame this field was computed from.
    grid_spacing : int
        Nominal pixel size of each grid cell (edge cells may be smaller,
        clipped to the actual image bounds).
    x, y : numpy.ndarray
        Cell center coordinates, in source-image pixels, shape
        ``(rows, cols)``.
    packing_fraction : numpy.ndarray
        Packing fraction within each cell, shape ``(rows, cols)``. Each
        detection's full area is assigned to the single cell containing
        its centroid (not split across cells it may overlap), so this is
        an approximation that gets coarser as ``grid_spacing`` approaches
        typical particle size.
    """

    frame_id: int
    grid_spacing: int
    x: NDArray[np.integer]
    y: NDArray[np.integer]
    packing_fraction: NDArray[np.float64]


def compute_packing_metrics(
    detections: Sequence[Detection], *, roi_area: float, frame_id: int = 0
) -> PackingMetrics:
    """Compute packing fraction, void fraction, and number density from a frame's detections.

    Parameters
    ----------
    detections : sequence of Detection
        Particles detected in one frame (see
        :func:`glas.analysis.detect_particles`).
    roi_area : float
        Area of the region of interest the detections were found in, in
        pixels² (typically the whole frame: ``width * height``).
    frame_id : int, default 0
        Recorded onto the returned metrics as
        :attr:`PackingMetrics.frame_id`.

    Returns
    -------
    PackingMetrics

    Raises
    ------
    ValueError
        If ``roi_area`` is not positive.
    """
    if roi_area <= 0:
        raise ValueError(f"roi_area must be positive, got {roi_area}.")

    total_area = sum(detection.area for detection in detections)
    packing_fraction = total_area / roi_area

    return PackingMetrics(
        frame_id=frame_id,
        particle_count=len(detections),
        packing_fraction=packing_fraction,
        void_fraction=1.0 - packing_fraction,
        number_density=len(detections) / roi_area,
    )


def compute_packing_field(
    detections: Sequence[Detection],
    *,
    image_width: int,
    image_height: int,
    grid_spacing: int = DEFAULT_GRID_SPACING,
    frame_id: int = 0,
) -> PackingField:
    """Compute a spatial packing-fraction field, binned onto a coarse grid.

    Each detection's full area is assigned to the single grid cell
    containing its centroid (centroid-based binning, not exact
    area-overlap splitting across cells it may straddle) -- simple,
    fast, and accurate as long as particles are small relative to
    ``grid_spacing``.

    Parameters
    ----------
    detections : sequence of Detection
        Particles detected in one frame.
    image_width, image_height : int
        Frame dimensions, in pixels.
    grid_spacing : int, default 32
        Nominal pixel size of each grid cell. Edge cells are clipped to
        the actual image bounds when the dimensions aren't an exact
        multiple of ``grid_spacing``.
    frame_id : int, default 0
        Recorded onto the returned field as :attr:`PackingField.frame_id`.

    Returns
    -------
    PackingField

    Raises
    ------
    ValueError
        If ``grid_spacing``, ``image_width``, or ``image_height`` is not
        at least 1.
    """
    if grid_spacing < 1:
        raise ValueError(f"grid_spacing must be at least 1, got {grid_spacing}.")
    if image_width < 1:
        raise ValueError(f"image_width must be at least 1, got {image_width}.")
    if image_height < 1:
        raise ValueError(f"image_height must be at least 1, got {image_height}.")

    n_cols = math.ceil(image_width / grid_spacing)
    n_rows = math.ceil(image_height / grid_spacing)

    area_sums = np.zeros((n_rows, n_cols), dtype=np.float64)
    for detection in detections:
        col = min(max(int(detection.x // grid_spacing), 0), n_cols - 1)
        row = min(max(int(detection.y // grid_spacing), 0), n_rows - 1)
        area_sums[row, col] += detection.area

    cell_areas = np.empty((n_rows, n_cols), dtype=np.float64)
    for row in range(n_rows):
        cell_height = min(grid_spacing, image_height - row * grid_spacing)
        for col in range(n_cols):
            cell_width = min(grid_spacing, image_width - col * grid_spacing)
            cell_areas[row, col] = cell_width * cell_height

    packing_fraction = area_sums / cell_areas

    xs = np.arange(n_cols) * grid_spacing + grid_spacing / 2
    ys = np.arange(n_rows) * grid_spacing + grid_spacing / 2
    grid_x, grid_y = np.meshgrid(xs, ys)

    return PackingField(
        frame_id=frame_id,
        grid_spacing=grid_spacing,
        x=grid_x,
        y=grid_y,
        packing_fraction=packing_fraction,
    )


def plot_packing_heatmap(field: PackingField, output_path: Path) -> Path:
    """Render a packing-fraction field as a color-coded heat map, saved as a PNG.

    Uses matplotlib's non-interactive ``"Agg"`` backend (see
    :mod:`glas.analysis.brazil_nut` for why this is always headless-safe),
    so this never requires or attempts to open a display.

    Parameters
    ----------
    field : PackingField
        Field to plot, from :func:`compute_packing_field`.
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        field.x, field.y, field.packing_fraction, shading="auto", cmap="magma", vmin=0, vmax=1
    )
    fig.colorbar(mesh, ax=ax, label="Packing fraction")

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # image coordinates: y increases downward
    ax.set_aspect("equal")
    ax.set_title(f"Packing fraction -- frame {field.frame_id}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_packing_summary(summary: PackingSummary, output_path: Path) -> Path:
    """Plot packing fraction and particle count over time, saved as a PNG.

    Two stacked panels: packing fraction vs. time, and particle count vs.
    time. Uses matplotlib's non-interactive ``"Agg"`` backend, so this
    never requires or attempts to open a display.

    Parameters
    ----------
    summary : PackingSummary
        Summary to plot, from :func:`analyze_packing`.
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    packing_fractions = [m.packing_fraction for m in summary.metrics]
    particle_counts = [m.particle_count for m in summary.metrics]

    fig, (fraction_ax, count_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    fraction_ax.plot(summary.times_s, packing_fractions, marker="o", markersize=3)
    fraction_ax.set_ylabel("Packing fraction")
    fraction_ax.set_title("Packing analysis")

    count_ax.plot(summary.times_s, particle_counts, marker="o", markersize=3)
    count_ax.set_ylabel("Particle count")
    count_ax.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def analyze_packing(
    folder: Path,
    *,
    roi_area: float | None = None,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
    field_grid_spacing: int | None = None,
    field_dir: Path | None = None,
) -> PackingSummary:
    """Detect particles and compute packing statistics for every frame in a recorded dataset.

    Convenience wrapper combining :func:`glas.dataset.iter_frames` and
    :func:`glas.analysis.detect_particles` with :func:`compute_packing_metrics`
    (and, optionally, :func:`compute_packing_field` and
    :func:`plot_packing_heatmap`) into one call over a dataset folder.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    roi_area : float, optional
        Area of the region of interest, in pixels². ``None`` (the
        default) uses the whole frame (``metadata.width * metadata.height``).
    min_area, max_area, threshold, invert : see
        :func:`glas.analysis.detect_particles`.
    field_grid_spacing : int, optional
        If given (together with ``field_dir``), also compute a spatial
        packing-fraction field for each frame (see
        :func:`compute_packing_field`).
    field_dir : pathlib.Path, optional
        If given (together with ``field_grid_spacing``), save one
        packing-field heat map PNG per frame here, named
        ``packing_{frame_id:06d}.png``.

    Returns
    -------
    PackingSummary

    Raises
    ------
    PackingError
        If the dataset has no frames.
    """
    metadata = load_metadata_json(folder / _METADATA_FILENAME)
    if roi_area is None:
        roi_area = float(metadata.width * metadata.height)

    frame_ids: list[int] = []
    times_s: list[float] = []
    metrics: list[PackingMetrics] = []
    start_ns: int | None = None

    for frame in iter_frames(folder):
        if start_ns is None:
            start_ns = frame.host_timestamp_ns

        detections = detect_particles(
            frame.image,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
        )
        frame_metrics = compute_packing_metrics(
            detections, roi_area=roi_area, frame_id=frame.frame_id
        )

        frame_ids.append(frame.frame_id)
        times_s.append((frame.host_timestamp_ns - start_ns) / 1e9)
        metrics.append(frame_metrics)

        if field_grid_spacing is not None and field_dir is not None:
            field = compute_packing_field(
                detections,
                image_width=metadata.width,
                image_height=metadata.height,
                grid_spacing=field_grid_spacing,
                frame_id=frame.frame_id,
            )
            plot_packing_heatmap(field, field_dir / f"packing_{frame.frame_id:06d}.png")

    if not metrics:
        raise PackingError("Cannot analyze packing: dataset has no frames.")

    return PackingSummary(frame_ids=frame_ids, times_s=times_s, metrics=metrics)

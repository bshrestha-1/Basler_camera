"""Convection analysis: dense optical flow, velocity fields, circulation, and heat maps.

Unlike :mod:`glas.analysis.particle_tracking` (which follows individual
particles), this module measures bulk flow directly from pixel motion
between consecutive frames via dense optical flow (Farneback's method) --
well suited to granular convection, where textured material moves in
bulk patterns (convection rolls) that don't require identifying
individual particles::

    glas.dataset.iter_frames() -> compute_optical_flow() -> VelocityField
                                                                  |
                                              compute_vorticity() / total_circulation()
                                                                  |
                                                      plot_velocity_heatmap()

:func:`analyze_convection` runs the whole pipeline over a finalized
dataset folder in one call, the same role :func:`glas.analysis.track_dataset`
and :func:`glas.analysis.analyze_brazil_nut` play for their own phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- see glas.analysis.brazil_nut
# for why this is always headless-safe, unconditionally, with no pre-flight check needed.
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from glas.analysis.tracking_utils import to_uint8_mono  # noqa: E402
from glas.dataset import iter_frames  # noqa: E402
from glas.exceptions import ConvectionError  # noqa: E402
from glas.plotting import apply_publication_style, savefig_publication  # noqa: E402

DEFAULT_GRID_SPACING = 16
HeatmapBackground = Literal["speed", "vorticity"]


@dataclass(frozen=True)
class VelocityField:
    """A dense velocity field computed between two consecutive frames.

    Like :class:`glas.frame.Frame`, this is a plain dataclass rather than
    a Pydantic model: every field but ``frame_id``/``elapsed_s``/
    ``grid_spacing`` is a numpy array, and Pydantic would need
    ``arbitrary_types_allowed`` without buying any real shape/dtype
    validation for them (see ``Frame``'s own docstring for the identical
    reasoning).

    Attributes
    ----------
    frame_id : int
        The later of the two frames this field was computed between (the
        "to" frame).
    elapsed_s : float
        Real elapsed time between the two frames, in seconds -- used to
        convert pixel displacement into a pixels/second velocity.
    grid_spacing : int
        Pixel spacing between adjacent velocity samples -- the dense
        per-pixel optical flow is downsampled to a grid this coarse for
        readability and size.
    x, y : numpy.ndarray
        Sample point coordinates, in source-image pixels, shape
        ``(rows, cols)``.
    vx, vy : numpy.ndarray
        Velocity components at each sample point, in pixels/second,
        shape ``(rows, cols)``.
    """

    frame_id: int
    elapsed_s: float
    grid_spacing: int
    x: NDArray[np.integer]
    y: NDArray[np.integer]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]

    @property
    def speed(self) -> NDArray[np.float64]:
        """Velocity magnitude at each sample point, in pixels/second."""
        return np.hypot(self.vx, self.vy)


class ConvectionSummary(BaseModel):
    """Circulation over time across a whole recording.

    Attributes
    ----------
    frame_ids : list of int
        The later frame of each consecutive pair a velocity field was
        computed for.
    times_s : list of float
        Elapsed time since the first frame pair, in seconds.
    circulations : list of float
        Net circulation at each frame pair (see :func:`total_circulation`),
        in pixels²/second.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame_ids: list[int]
    times_s: list[float]
    circulations: list[float]
    fields: list[VelocityField]


def compute_optical_flow(
    previous_image: NDArray[np.integer],
    current_image: NDArray[np.integer],
    *,
    grid_spacing: int = DEFAULT_GRID_SPACING,
    elapsed_s: float = 1.0,
    frame_id: int = 0,
) -> VelocityField:
    """Compute a dense velocity field between two consecutive mono frames.

    Uses Farneback dense optical flow (``cv2.calcOpticalFlowFarneback``),
    downsampled to a regular grid for a readable, size-bounded result --
    the raw per-pixel flow field is one 2-vector per pixel, too large and
    too noisy to be a useful "field" on its own.

    Parameters
    ----------
    previous_image, current_image : numpy.ndarray
        Consecutive mono frames, same shape. Non-``uint8`` dtypes are
        scaled to 8-bit first.
    grid_spacing : int, default 16
        Pixel spacing between adjacent velocity samples.
    elapsed_s : float, default 1.0
        Real elapsed time between the two frames, in seconds -- used to
        convert pixel displacement into a pixels/second velocity. Pass
        the real value (e.g. from ``Frame.host_timestamp_ns``) for a
        physically meaningful velocity; the default of 1.0 leaves the
        result in pixels/frame, useful for quick, timing-agnostic checks.
    frame_id : int, default 0
        Recorded onto the returned field as :attr:`VelocityField.frame_id`.

    Returns
    -------
    VelocityField

    Raises
    ------
    ValueError
        If ``grid_spacing`` is not positive, or ``elapsed_s`` is not
        positive.
    """
    if grid_spacing < 1:
        raise ValueError(f"grid_spacing must be at least 1, got {grid_spacing}.")
    if elapsed_s <= 0:
        raise ValueError(f"elapsed_s must be positive, got {elapsed_s}.")

    previous_8bit = to_uint8_mono(previous_image)
    current_8bit = to_uint8_mono(current_image)

    # cv2's stubs omit the documented `flow=None` ("let OpenCV allocate the
    # output") convention its overloads accept at runtime for every other
    # optional-output parameter.
    flow = cv2.calcOpticalFlowFarneback(
        previous_8bit,
        current_8bit,
        None,  # type: ignore[call-overload]
        0.5,  # pyr_scale
        3,  # levels
        15,  # winsize
        3,  # iterations
        5,  # poly_n
        1.2,  # poly_sigma
        0,  # flags
    )

    height, width = previous_8bit.shape[:2]
    xs = np.arange(0, width, grid_spacing)
    ys = np.arange(0, height, grid_spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)

    vx = flow[grid_y, grid_x, 0].astype(np.float64) / elapsed_s
    vy = flow[grid_y, grid_x, 1].astype(np.float64) / elapsed_s

    return VelocityField(
        frame_id=frame_id,
        elapsed_s=elapsed_s,
        grid_spacing=grid_spacing,
        x=grid_x,
        y=grid_y,
        vx=vx,
        vy=vy,
    )


def compute_vorticity(field: VelocityField) -> NDArray[np.float64]:
    """Compute the vorticity (curl) of a velocity field at each grid sample.

    Vorticity ``omega = d(vy)/dx - d(vx)/dy``, estimated via finite
    differences across the grid (central differences away from the
    edges, one-sided at the edges -- ``numpy.gradient`` handles this
    automatically). Positive values indicate counter-clockwise rotation
    in image coordinates (where ``y`` increases downward), negative
    clockwise -- the classic diagnostic for convection rolls in vibrated
    granular media.

    Parameters
    ----------
    field : VelocityField

    Returns
    -------
    numpy.ndarray
        Vorticity at each grid sample, shape matching :attr:`VelocityField.vx`,
        in units of 1/second.
    """
    spacing = field.grid_spacing
    dvy_dx = np.gradient(field.vy, spacing, axis=1)
    dvx_dy = np.gradient(field.vx, spacing, axis=0)
    result: NDArray[np.float64] = dvy_dx - dvx_dy
    return result


def total_circulation(field: VelocityField) -> float:
    """Net circulation across a whole velocity field.

    Approximates the area integral of vorticity (Stokes' theorem:
    circulation around a closed loop equals the enclosed vorticity's
    integral) as a sum of each grid cell's vorticity times its area.
    Positive means net counter-clockwise circulation dominates the frame,
    negative net clockwise, near zero means rotational motion is balanced
    or absent (no strong convection roll, or two roughly equal and
    opposite rolls).

    Parameters
    ----------
    field : VelocityField

    Returns
    -------
    float
        Net circulation, in pixels²/second.
    """
    vorticity = compute_vorticity(field)
    return float(np.sum(vorticity) * field.grid_spacing**2)


def plot_velocity_heatmap(
    field: VelocityField,
    output_path: Path,
    *,
    background: HeatmapBackground = "speed",
    show_quiver: bool = True,
) -> Path:
    """Render a velocity field as a color-coded heat map, with an optional quiver overlay.

    Uses matplotlib's non-interactive ``"Agg"`` backend (see
    :mod:`glas.analysis.brazil_nut` for why this is always headless-safe),
    so this never requires or attempts to open a display.

    Parameters
    ----------
    field : VelocityField
        Field to plot, from :func:`compute_optical_flow`.
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.
    background : {"speed", "vorticity"}, default "speed"
        Which scalar field to render as the color-coded background:
        velocity magnitude, or vorticity (see :func:`compute_vorticity`).
    show_quiver : bool, default True
        Whether to overlay velocity vectors as arrows.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.

    Raises
    ------
    ValueError
        If ``background`` is not ``"speed"`` or ``"vorticity"``.
    """
    if background == "speed":
        values = field.speed
        label = "Speed (px/s)"
        cmap = "viridis"
    elif background == "vorticity":
        values = compute_vorticity(field)
        label = "Vorticity (1/s)"
        cmap = "coolwarm"
    else:
        raise ValueError(f"Unknown background {background!r}; expected 'speed' or 'vorticity'.")

    apply_publication_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(field.x, field.y, values, shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label=label)

    if show_quiver:
        ax.quiver(field.x, field.y, field.vx, field.vy, color="black", alpha=0.6)

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # image coordinates: y increases downward
    ax.set_aspect("equal")
    ax.set_title(f"Velocity field -- frame {field.frame_id}")

    fig.tight_layout()
    return savefig_publication(fig, output_path)


def analyze_convection(
    folder: Path,
    *,
    grid_spacing: int = DEFAULT_GRID_SPACING,
    heatmap_dir: Path | None = None,
    heatmap_background: HeatmapBackground = "speed",
) -> ConvectionSummary:
    """Compute the velocity field and circulation for every consecutive frame pair.

    Convenience wrapper combining :func:`glas.dataset.iter_frames` and
    :func:`compute_optical_flow` (and, optionally,
    :func:`plot_velocity_heatmap`) into one call over a dataset folder.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    grid_spacing : int, default 16
        See :func:`compute_optical_flow`.
    heatmap_dir : pathlib.Path, optional
        If given, save one heat map PNG per frame pair here, named
        ``flow_{frame_id:06d}.png``.
    heatmap_background : {"speed", "vorticity"}, default "speed"
        See :func:`plot_velocity_heatmap`.

    Returns
    -------
    ConvectionSummary

    Raises
    ------
    ConvectionError
        If the dataset has fewer than 2 frames, or two consecutive
        frames have a non-increasing timestamp (zero or negative elapsed
        time -- cannot compute a velocity from it).
    """
    frames = iter_frames(folder)
    try:
        previous = next(frames)
    except StopIteration:
        raise ConvectionError("Cannot analyze convection: dataset has no frames.") from None

    frame_ids: list[int] = []
    times_s: list[float] = []
    circulations: list[float] = []
    fields: list[VelocityField] = []
    start_ns = previous.host_timestamp_ns

    for frame in frames:
        elapsed_ns = frame.host_timestamp_ns - previous.host_timestamp_ns
        if elapsed_ns <= 0:
            raise ConvectionError(
                f"Non-increasing timestamps between frame {previous.frame_id} and "
                f"{frame.frame_id}; cannot compute a velocity from zero or negative "
                "elapsed time."
            )

        field = compute_optical_flow(
            previous.image,
            frame.image,
            grid_spacing=grid_spacing,
            elapsed_s=elapsed_ns / 1e9,
            frame_id=frame.frame_id,
        )

        frame_ids.append(frame.frame_id)
        times_s.append((frame.host_timestamp_ns - start_ns) / 1e9)
        circulations.append(total_circulation(field))
        fields.append(field)

        if heatmap_dir is not None:
            plot_velocity_heatmap(
                field,
                heatmap_dir / f"flow_{frame.frame_id:06d}.png",
                background=heatmap_background,
            )

        previous = frame

    if not fields:
        raise ConvectionError("Cannot analyze convection: dataset has fewer than 2 frames.")

    return ConvectionSummary(
        frame_ids=frame_ids,
        times_s=times_s,
        circulations=circulations,
        fields=fields,
    )

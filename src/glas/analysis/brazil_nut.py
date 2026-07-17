"""Brazil nut (intruder) effect analysis: height, rise time, velocity, and plots.

Automatically identifies the intruder particle among a recording's tracked
trajectories (:mod:`glas.analysis.particle_tracking` output) as the one
with the largest average radius -- the standard setup for a Brazil nut
effect experiment, where a single particle larger than the surrounding
matrix is seeded into a granular bed and rises to the surface under
vibration -- then measures its height and rise velocity over real elapsed
time (from each observation's :attr:`~glas.analysis.TrackedParticle.host_timestamp_ns`,
not an assumed frame rate)::

    glas.analysis.track_dataset() -> identify_brazil_nut() -> compute_brazil_nut_trajectory()
                                                                        |
                                                          plot_brazil_nut_trajectory()

:func:`analyze_brazil_nut` runs the whole pipeline over a finalized
dataset folder in one call, the same role :func:`glas.export.export_dataset`
and :func:`glas.analysis.track_dataset` play for their own phases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive, raster-only backend -- never touches a display,
# unlike a GUI backend, so this is safe unconditionally (no headless-hang
# risk of the kind glas.display._display_available() guards against).
import matplotlib.pyplot as plt  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from glas.analysis.particle_tracking import (  # noqa: E402
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_GAP,
    TrackedParticle,
    track_dataset,
)
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA  # noqa: E402
from glas.exceptions import BrazilNutError  # noqa: E402
from glas.metadata import load_metadata_json  # noqa: E402
from glas.plotting import apply_publication_style, savefig_publication, style_axes  # noqa: E402

DEFAULT_SETTLE_FRACTION = 0.9
_METADATA_FILENAME = "metadata.json"


class BrazilNutTrajectory(BaseModel):
    """Height and velocity of the identified Brazil nut (intruder) over time.

    Attributes
    ----------
    track_id : int
        Which tracked particle this is (see
        :class:`~glas.analysis.ParticleTracker`).
    frame_ids : list of int
        Frame each observation came from, in order.
    times_s : list of float
        Elapsed time since the first observation, in seconds -- derived
        from real per-frame timestamps, not an assumed frame rate.
    heights_px : list of float
        Height above the bottom of the frame, in pixels
        (``frame_height - y``, so this increases as the particle rises --
        the physically intuitive direction, unlike raw image-coordinate
        ``y``, which increases downward).
    velocities_px_s : list of float
        Instantaneous rise velocity between consecutive observations, in
        pixels per second (finite differences of ``heights_px`` over
        ``times_s``). One shorter than ``heights_px``/``times_s``.
    rise_time_s : float or None
        Elapsed time at which the particle first reached
        ``settle_fraction`` of the frame height. ``None`` if it never
        did within this recording.
    mean_velocity_px_s : float
        Overall average rise velocity across the whole trajectory
        (total height change over total elapsed time).
    """

    model_config = ConfigDict(frozen=True)

    track_id: int
    frame_ids: list[int]
    times_s: list[float]
    heights_px: list[float]
    velocities_px_s: list[float]
    rise_time_s: float | None
    mean_velocity_px_s: float


def identify_brazil_nut(history: dict[int, list[TrackedParticle]]) -> int:
    """Identify the Brazil nut (intruder) track: the one with the largest mean radius.

    Parameters
    ----------
    history : dict of int to list of TrackedParticle
        Trajectories, as returned by
        :meth:`glas.analysis.ParticleTracker.history` or
        :func:`glas.analysis.track_dataset`.

    Returns
    -------
    int
        The ``track_id`` of the largest particle, by mean equivalent
        radius across its whole trajectory.

    Raises
    ------
    BrazilNutError
        If ``history`` is empty (no particles were tracked at all).
    """
    if not history:
        raise BrazilNutError("Cannot identify a Brazil nut: no tracked particles.")

    def _mean_radius(observations: list[TrackedParticle]) -> float:
        return sum(obs.radius for obs in observations) / len(observations)

    return max(history, key=lambda track_id: _mean_radius(history[track_id]))


def compute_brazil_nut_trajectory(
    history: dict[int, list[TrackedParticle]],
    *,
    frame_height: int,
    track_id: int | None = None,
    settle_fraction: float = DEFAULT_SETTLE_FRACTION,
) -> BrazilNutTrajectory:
    """Compute height, rise time, and velocity for the Brazil nut track.

    Parameters
    ----------
    history : dict of int to list of TrackedParticle
        Trajectories from :func:`glas.analysis.track_dataset` or
        :meth:`glas.analysis.ParticleTracker.history`.
    frame_height : int
        Frame height, in pixels -- used to convert each observation's
        image-coordinate ``y`` (0 at the top) into height above the
        bottom of the frame (0 at the bottom, increasing upward).
    track_id : int, optional
        Which track to analyze. ``None`` (the default) auto-identifies
        it via :func:`identify_brazil_nut`.
    settle_fraction : float, default 0.9
        Fraction of ``frame_height`` the particle must reach to count as
        "risen" for :attr:`BrazilNutTrajectory.rise_time_s`.

    Returns
    -------
    BrazilNutTrajectory

    Raises
    ------
    BrazilNutError
        If ``history`` is empty, ``track_id`` doesn't exist in it, the
        selected track has fewer than 2 observations (no velocity can be
        computed from a single point), or two consecutive observations
        have the same or a decreasing ``host_timestamp_ns`` (zero or
        negative elapsed time -- typically means ``host_timestamp_ns``
        was never set, since it defaults to ``0``).
    ValueError
        If ``settle_fraction`` is not in ``(0, 1]``.
    """
    if not 0 < settle_fraction <= 1:
        raise ValueError(f"settle_fraction must be in (0, 1], got {settle_fraction}.")
    if not history:
        raise BrazilNutError("Cannot compute a Brazil nut trajectory: no tracked particles.")

    if track_id is None:
        track_id = identify_brazil_nut(history)
    elif track_id not in history:
        raise BrazilNutError(f"track_id {track_id} not found in history.")

    observations = history[track_id]
    if len(observations) < 2:
        raise BrazilNutError(
            f"Track {track_id} has only {len(observations)} observation(s); "
            "at least 2 are needed to compute a trajectory."
        )

    frame_ids = [obs.frame_id for obs in observations]
    start_ns = observations[0].host_timestamp_ns
    times_s = [(obs.host_timestamp_ns - start_ns) / 1e9 for obs in observations]

    for i in range(len(times_s) - 1):
        if times_s[i + 1] <= times_s[i]:
            raise BrazilNutError(
                f"Track {track_id} has non-increasing timestamps between frame "
                f"{frame_ids[i]} and {frame_ids[i + 1]}; cannot compute a velocity from zero "
                "or negative elapsed time. Make sure host_timestamp_ns is set on every "
                "TrackedParticle (glas.analysis.track_dataset() does this automatically)."
            )

    heights_px = [frame_height - obs.y for obs in observations]
    velocities_px_s = [
        (heights_px[i + 1] - heights_px[i]) / (times_s[i + 1] - times_s[i])
        for i in range(len(heights_px) - 1)
    ]

    threshold = settle_fraction * frame_height
    rise_time_s = next(
        (t for t, h in zip(times_s, heights_px, strict=True) if h >= threshold), None
    )

    mean_velocity_px_s = (heights_px[-1] - heights_px[0]) / (times_s[-1] - times_s[0])

    return BrazilNutTrajectory(
        track_id=track_id,
        frame_ids=frame_ids,
        times_s=times_s,
        heights_px=heights_px,
        velocities_px_s=velocities_px_s,
        rise_time_s=rise_time_s,
        mean_velocity_px_s=mean_velocity_px_s,
    )


def plot_brazil_nut_trajectory(trajectory: BrazilNutTrajectory, output_path: Path) -> Path:
    """Plot height and velocity over time for a Brazil nut trajectory, saved as a PNG.

    Two stacked panels: height vs. time (with the rise time marked, if
    found), and velocity vs. time. Uses matplotlib's non-interactive
    ``"Agg"`` backend, so this never requires or attempts to open a
    display -- safe in CI, over SSH, or any other headless environment.

    Parameters
    ----------
    trajectory : BrazilNutTrajectory
        Trajectory to plot, from :func:`compute_brazil_nut_trajectory`.
    output_path : pathlib.Path
        Destination PNG file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    apply_publication_style()

    fig, (height_ax, velocity_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    height_ax.plot(trajectory.times_s, trajectory.heights_px, marker="o", markersize=3)
    height_ax.set_ylabel("Height above bottom (px)")
    height_ax.set_title(f"Brazil nut trajectory -- track {trajectory.track_id}")
    if trajectory.rise_time_s is not None:
        height_ax.axvline(trajectory.rise_time_s, color="red", linestyle="--", label="rise time")
        height_ax.legend()
    style_axes(height_ax)

    velocity_ax.plot(trajectory.times_s[1:], trajectory.velocities_px_s, marker="o", markersize=3)
    velocity_ax.set_ylabel("Rise velocity (px/s)")
    velocity_ax.set_xlabel("Time (s)")
    velocity_ax.axhline(0, color="gray", linewidth=0.8)
    style_axes(velocity_ax)

    fig.tight_layout()
    return savefig_publication(fig, output_path)


def analyze_brazil_nut(
    folder: Path,
    *,
    track_id: int | None = None,
    settle_fraction: float = DEFAULT_SETTLE_FRACTION,
    plot_path: Path | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_gap: int = DEFAULT_MAX_GAP,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
) -> BrazilNutTrajectory:
    """Detect, track, and analyze the Brazil nut (intruder) in a recorded dataset.

    Convenience wrapper combining :func:`glas.analysis.track_dataset` and
    :func:`compute_brazil_nut_trajectory` (and, optionally,
    :func:`plot_brazil_nut_trajectory`) into one call over a dataset
    folder.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    track_id, settle_fraction : see :func:`compute_brazil_nut_trajectory`.
    plot_path : pathlib.Path, optional
        If given, also save a height/velocity plot here (see
        :func:`plot_brazil_nut_trajectory`).
    max_distance, max_gap, min_area, max_area, threshold, invert : see
        :func:`glas.analysis.track_dataset`.

    Returns
    -------
    BrazilNutTrajectory
    """
    metadata = load_metadata_json(folder / _METADATA_FILENAME)
    history = track_dataset(
        folder,
        max_distance=max_distance,
        max_gap=max_gap,
        min_area=min_area,
        max_area=max_area,
        threshold=threshold,
        invert=invert,
    )
    trajectory = compute_brazil_nut_trajectory(
        history,
        frame_height=metadata.height,
        track_id=track_id,
        settle_fraction=settle_fraction,
    )

    if plot_path is not None:
        plot_brazil_nut_trajectory(trajectory, plot_path)

    return trajectory

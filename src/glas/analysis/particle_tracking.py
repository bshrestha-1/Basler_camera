"""Links per-frame particle detections into trajectories across a recording.

    glas.dataset.iter_frames() -> detect_particles() -> ParticleTracker -> trajectories

:class:`ParticleTracker` is the incremental, call-once-per-frame API;
:func:`track_dataset` is the convenience wrapper that runs the whole
pipeline over a finalized dataset folder in one call, the same role
:func:`glas.export.export_dataset` plays for exporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from glas.analysis.tracking_utils import DEFAULT_MIN_AREA, Detection, detect_particles, link_nearest
from glas.dataset import iter_frames

DEFAULT_MAX_DISTANCE = 20.0
DEFAULT_MAX_GAP = 0


class TrackedParticle(BaseModel):
    """One particle observation, linked to a trajectory.

    Attributes
    ----------
    track_id : int
        Identifies the trajectory this observation belongs to. Stable
        across frames for the same physical particle; never reused after
        a track is retired.
    frame_id : int
        Frame this observation was made in.
    x, y : float
        Centroid position, in pixels.
    radius : float
        Equivalent radius, in pixels.
    area : float
        Blob area, in pixels.
    host_timestamp_ns : int
        The source frame's :attr:`glas.frame.Frame.host_timestamp_ns`, if
        known (``0`` otherwise -- e.g. when a track is built from
        synthetic detections with no real frame behind them). Suitable
        for computing real elapsed time between observations (see
        :mod:`glas.analysis.brazil_nut`), the same way it is on `Frame`
        itself -- not wall-clock time.
    """

    model_config = ConfigDict(frozen=True)

    track_id: int
    frame_id: int
    x: float
    y: float
    radius: float
    area: float
    host_timestamp_ns: int = 0


@dataclass
class _ActiveTrack:
    detection: Detection
    last_seen_frame_id: int


class ParticleTracker:
    """Links successive frames' particle detections into trajectories.

    Call :meth:`update` once per frame, in strictly increasing
    ``frame_id`` order.

    Parameters
    ----------
    max_distance : float, default 20.0
        Maximum pixel distance between a track's last known position and
        a new detection for them to be linked as the same particle. See
        :func:`glas.analysis.tracking_utils.link_nearest`.
    max_gap : int, default 0
        Number of consecutive frames a track may go unmatched before
        it's retired. ``0`` means a track is retired the very next frame
        it isn't matched (no occlusion tolerance); a positive value lets
        a track survive a brief occlusion and resume if a nearby
        detection reappears within ``max_gap`` frames.

    Raises
    ------
    ValueError
        If ``max_distance`` is not positive, or ``max_gap`` is negative.
    """

    def __init__(
        self, max_distance: float = DEFAULT_MAX_DISTANCE, max_gap: int = DEFAULT_MAX_GAP
    ) -> None:
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}.")
        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}.")
        self._max_distance = max_distance
        self._max_gap = max_gap
        self._next_track_id = 0
        self._active: dict[int, _ActiveTrack] = {}
        self._history: dict[int, list[TrackedParticle]] = {}

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently active (matched within the last ``max_gap`` frames)."""
        return len(self._active)

    @property
    def history(self) -> dict[int, list[TrackedParticle]]:
        """Every observation recorded so far, keyed by ``track_id``.

        Includes retired tracks -- this is the full trajectory record,
        not just currently active ones. Each call returns a fresh copy;
        mutating the result does not affect the tracker.
        """
        return {track_id: list(observations) for track_id, observations in self._history.items()}

    def update(
        self, frame_id: int, detections: Sequence[Detection], host_timestamp_ns: int = 0
    ) -> list[TrackedParticle]:
        """Link one frame's detections onto existing tracks.

        Matches ``detections`` against currently active tracks (see
        :func:`~glas.analysis.tracking_utils.link_nearest`), spawns a new
        track for every unmatched detection, and retires any track that
        has gone unmatched for more than ``max_gap`` frames.

        Parameters
        ----------
        frame_id : int
            The frame these detections came from. Must be strictly
            greater than the ``frame_id`` of the previous call.
        detections : sequence of Detection
            This frame's detected particles, e.g. from
            :func:`~glas.analysis.tracking_utils.detect_particles`.
        host_timestamp_ns : int, default 0
            The source frame's :attr:`glas.frame.Frame.host_timestamp_ns`,
            recorded onto every :class:`TrackedParticle` produced by this
            call. Optional -- omit it if you don't need real elapsed-time
            calculations downstream (see :mod:`glas.analysis.brazil_nut`).

        Returns
        -------
        list of TrackedParticle
            One entry per detection processed this call, covering both
            continued and newly spawned tracks.
        """
        track_ids = list(self._active.keys())
        previous_detections = [self._active[track_id].detection for track_id in track_ids]
        pairs = link_nearest(previous_detections, list(detections), self._max_distance)

        observed: list[TrackedParticle] = []
        matched_current: set[int] = set()

        for i, j in pairs:
            track_id = track_ids[i]
            detection = detections[j]
            self._active[track_id] = _ActiveTrack(detection=detection, last_seen_frame_id=frame_id)
            observed.append(self._record(track_id, frame_id, detection, host_timestamp_ns))
            matched_current.add(j)

        for j, detection in enumerate(detections):
            if j in matched_current:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._active[track_id] = _ActiveTrack(detection=detection, last_seen_frame_id=frame_id)
            observed.append(self._record(track_id, frame_id, detection, host_timestamp_ns))

        for track_id in list(self._active):
            if frame_id - self._active[track_id].last_seen_frame_id > self._max_gap:
                del self._active[track_id]

        return observed

    def _record(
        self, track_id: int, frame_id: int, detection: Detection, host_timestamp_ns: int
    ) -> TrackedParticle:
        particle = TrackedParticle(
            track_id=track_id,
            frame_id=frame_id,
            x=detection.x,
            y=detection.y,
            radius=detection.radius,
            area=detection.area,
            host_timestamp_ns=host_timestamp_ns,
        )
        self._history.setdefault(track_id, []).append(particle)
        return particle


def track_dataset(
    folder: Path,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_gap: int = DEFAULT_MAX_GAP,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float | None = None,
    threshold: int | None = None,
    invert: bool = False,
) -> dict[int, list[TrackedParticle]]:
    """Detect and track particles across every frame of a recorded dataset.

    Convenience wrapper combining :func:`glas.dataset.iter_frames`,
    :func:`~glas.analysis.tracking_utils.detect_particles`, and
    :class:`ParticleTracker` into a single call over a whole dataset
    folder.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder (see :func:`glas.dataset.iter_frames`).
    max_distance, max_gap : see :class:`ParticleTracker`.
    min_area, max_area, threshold, invert : see
        :func:`~glas.analysis.tracking_utils.detect_particles`.

    Returns
    -------
    dict of int to list of TrackedParticle
        Every trajectory found, keyed by ``track_id`` -- equivalent to
        :attr:`ParticleTracker.history` after processing the whole
        dataset.
    """
    tracker = ParticleTracker(max_distance=max_distance, max_gap=max_gap)
    for frame in iter_frames(folder):
        detections = detect_particles(
            frame.image,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
            invert=invert,
        )
        tracker.update(frame.frame_id, detections, frame.host_timestamp_ns)
    return tracker.history

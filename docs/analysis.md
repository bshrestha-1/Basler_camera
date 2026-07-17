# Analysis Engine

Phase 11 adds the foundation every later analysis phase (Brazil nut
effect, convection, packing, segregation) builds its own measurements on
top of: detecting particles in each frame and linking those detections
into trajectories across a recording.

```
glas.dataset.iter_frames() -> detect_particles() -> ParticleTracker -> trajectories
```

## Module layout

- `glas.analysis.tracking_utils` -- pure image-processing functions, no
  state, no I/O: `detect_particles()` (per-frame blob detection) and
  `link_nearest()` (frame-to-frame matching). Fully unit-testable without
  a camera or a dataset.
- `glas.analysis.particle_tracking` -- `ParticleTracker` (the incremental,
  call-once-per-frame API) and `track_dataset()` (runs the whole pipeline
  over a finalized dataset folder in one call, the same role
  `glas.export.export_dataset()` plays for exporting).
- `glas.analysis.brazil_nut` -- Brazil nut (intruder) effect measurements
  built on top of tracked trajectories: automatic intruder identification,
  height/rise-time/velocity, and plots. See [`brazil-nut.md`](brazil-nut.md).
- `glas.analysis.convection` -- bulk flow measurements via dense optical
  flow (not particle tracking): velocity fields, vorticity, circulation,
  and heat maps. See [`convection.md`](convection.md).
- `glas.analysis.packing` -- packing fraction, void fraction, number
  density, and spatial fields, built on top of per-frame detections (not
  tracking). See [`packing.md`](packing.md).
- `glas.analysis.segregation` -- segregation index, mixing index, and
  mixing entropy, comparing two size-classified populations' spatial
  distribution, also built on per-frame detections (not tracking). See
  [`segregation.md`](segregation.md).

For AI-based detection (YOLO) and pixel-exact segmentation (SAM2) as an
alternative/complement to the classical `detect_particles()` pipeline
above, see [`ai.md`](ai.md) -- `glas.ai.yolo_detector.YoloDetection` is a
`Detection` subclass, so YOLO output plugs directly into
`ParticleTracker` and every pipeline in this file with no changes.

## Quickstart

```python
from glas.analysis import track_dataset

history = track_dataset(dataset.folder, max_distance=20.0)
for track_id, observations in history.items():
    print(f"track {track_id}: {len(observations)} frames")
```

Or from the command line:

```bash
glas analyze ~/glas_data/Run0001
```

For frame-by-frame control (e.g. to inspect or visualize each step):

```python
from glas.analysis.tracking_utils import detect_particles
from glas.analysis.particle_tracking import ParticleTracker
from glas.dataset import iter_frames

tracker = ParticleTracker(max_distance=20.0, max_gap=1)
for frame in iter_frames(dataset.folder):
    detections = detect_particles(frame.image, min_area=4.0)
    observed = tracker.update(frame.frame_id, detections)
```

## Detection

`detect_particles()` thresholds a mono frame to a binary mask (Otsu's
method by default, or an explicit 0-255 `threshold`), finds each
connected blob's contour, and reports its centroid and *equivalent
radius* (`sqrt(area / pi)`) -- a standard particle-sizing convention that
stays meaningful even for imperfectly circular blobs.

`invert` controls which side of the threshold counts as a particle:
`False` (the default) treats bright pixels as particles (bright particles
on a dark background); `True` treats dark pixels as particles (dark
particles on a bright background). Getting this backwards doesn't error
-- it silently detects the *background* as one big blob instead of the
real particles, so pick the direction that matches your lighting setup
and sanity-check the first frame's detection count and position before
processing a whole recording.

`min_area`/`max_area` filter by blob size, in pixels -- `min_area`
filters out single-pixel noise; `max_area` filters out large lit regions
that aren't particles (reflections, or several overlapping particles
merged into one blob by thresholding).

## Tracking

`ParticleTracker.update(frame_id, detections)` must be called once per
frame, in strictly increasing `frame_id` order. Each call:

1. Matches `detections` against currently active tracks via
   `link_nearest()`, within `max_distance` pixels.
2. Spawns a new track for every unmatched detection.
3. Retires any track that's gone unmatched for more than `max_gap`
   consecutive frames -- `max_gap=0` (the default) means no occlusion
   tolerance; a positive value lets a track survive a brief occlusion and
   resume if a nearby detection reappears within the window.

Track IDs are never reused. `ParticleTracker.history` returns every
observation recorded so far, including retired tracks -- the full
trajectory record, not just currently active ones.

### Why greedy nearest-neighbor, not the Hungarian algorithm

`link_nearest()` is a greedy nearest-neighbor match: every candidate pair
within `max_distance` is considered in increasing distance order, and the
first available match commits both indices. This is not a globally
optimal assignment (unlike the Hungarian algorithm,
`scipy.optimize.linear_sum_assignment`), and can occasionally produce a
suboptimal linking when two particles' plausible matches cross paths
between frames. The tradeoff buys no new dependency (`scipy` isn't
required) and is accurate as long as particles move less than roughly
half their typical inter-particle spacing between frames -- true for most
granular-material video at typical frame rates. Revisit with a Hungarian
assignment if dense, fast-crossing trajectories turn out to matter in
practice.

## Testing

`detect_particles()` and `link_nearest()` are tested against synthetic
images built with `cv2.circle()` at known positions (both bright-on-dark
and dark-on-bright, including the "wrong `invert`" failure mode described
above), not real recordings -- fully deterministic, no camera needed.
`ParticleTracker` is tested for track continuity, new-track spawning,
retirement at exactly `max_gap`, and resuming within the gap window.
`track_dataset()` and `glas analyze` are tested end to end against a real
on-disk dataset with a synthetically moving blob, and appear together
with every other phase in `tests/test_integration.py`'s full-pipeline
test.

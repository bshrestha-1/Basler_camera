# Brazil Nut (Intruder) Effect Analysis

Phase 12 measures the classic Brazil nut effect: a single particle larger
than the surrounding matrix, seeded into a granular bed, rises to the
surface under vibration. `glas.analysis.brazil_nut` identifies that
particle automatically, then measures its height, rise time, and velocity
over real elapsed time, and plots the result.

```
glas.analysis.track_dataset() -> identify_brazil_nut() -> compute_brazil_nut_trajectory()
                                                                    |
                                                      plot_brazil_nut_trajectory()
```

## Quickstart

```python
from glas.analysis import analyze_brazil_nut

trajectory = analyze_brazil_nut(dataset.folder, plot_path=Path("brazil_nut.png"))
print(f"rise time: {trajectory.rise_time_s} s")
print(f"mean velocity: {trajectory.mean_velocity_px_s:.1f} px/s")
```

or from the command line:

```bash
glas brazil-nut ~/glas_data/Run0001 --plot brazil_nut.png
```

`analyze_brazil_nut()` runs the whole pipeline over a dataset folder in
one call -- detection, tracking, intruder identification, and the height/
velocity/rise-time measurements -- the same role `track_dataset()` and
`export_dataset()` play for their own phases. For finer control (e.g.
reusing an already-computed `track_dataset()` result, or picking a
specific track), call `identify_brazil_nut()` and
`compute_brazil_nut_trajectory()` directly.

## Automatic intruder identification

`identify_brazil_nut()` picks the track with the largest mean equivalent
radius across its whole trajectory -- the standard Brazil nut experiment
setup seeds exactly one particle larger than the rest, so this needs no
manual selection. Pass `track_id` explicitly to `compute_brazil_nut_trajectory()`/
`analyze_brazil_nut()` to override it (e.g. to analyze a specific known
particle instead).

## Height, rise time, and velocity

- **Height** (`heights_px`) is `frame_height - y`: image-coordinate `y`
  increases downward, so this inversion makes "height" increase as the
  particle actually rises -- the physically intuitive direction.
- **Time** (`times_s`) comes from each observation's real
  `host_timestamp_ns` (propagated from `Frame.host_timestamp_ns` through
  `ParticleTracker`/`TrackedParticle`), not an assumed constant frame
  rate -- GLAS has no frame-rate concept anywhere else, and real
  per-frame timestamps were already available in the pipeline, so using
  them is both more accurate and requires no new parameter.
- **Rise time** (`rise_time_s`) is the elapsed time at which the particle
  first reaches `settle_fraction` (default `0.9`) of the frame height.
  `None` if it never does within the recording.
- **Velocity** (`velocities_px_s`) is the finite difference of height
  over time between consecutive observations; `mean_velocity_px_s` is the
  overall average (total height change over total elapsed time).

Two consecutive observations sharing the same (or a decreasing)
timestamp raise `BrazilNutError` rather than silently producing an
infinite or nonsensical velocity -- this typically means
`host_timestamp_ns` was never set (it defaults to `0` on a manually
constructed `TrackedParticle`); `track_dataset()` always sets it
correctly from the source `Frame`.

## Plots

`plot_brazil_nut_trajectory()` saves a two-panel PNG: height vs. time
(with the rise time marked, if found) and velocity vs. time. It uses
matplotlib's non-interactive `"Agg"` backend, set at module import time,
so it never requires or attempts to open a display -- safe in CI, over
SSH, or any other headless environment, the same headless-safety
guarantee `glas.display.PreviewWindow` has for live preview (though by a
simpler mechanism here: `Agg` is a raster-only backend that never touches
a display at all, unlike `PreviewWindow`'s GUI backend, which needs an
explicit pre-flight check).

## Testing

`identify_brazil_nut()`, `compute_brazil_nut_trajectory()`, and
`plot_brazil_nut_trajectory()` are tested against hand-computed synthetic
trajectories (not a real camera or dataset) -- height/velocity/rise-time
values are checked against values computed by hand, not just "did it
run." `plot_brazil_nut_trajectory()`'s output is verified as a real,
readable PNG via Pillow. `analyze_brazil_nut()` is tested end to end
against a real on-disk dataset with a synthetically rising blob (drawn
with `cv2.circle()`) alongside a small static one, confirming the larger
one is identified automatically. Every error path (empty history, unknown
track, single-observation track, non-increasing timestamps, invalid
`settle_fraction`) is exercised directly.

# Convection Analysis

Phase 13 measures bulk flow in the granular bed -- convection rolls,
circulation patterns -- directly from pixel motion between consecutive
frames, via dense optical flow. Unlike `glas.analysis.particle_tracking`
(which follows individual particles), this doesn't need to identify or
detect anything first: it works on the raw, textured motion of the
material itself, which is exactly what convection is.

```
glas.dataset.iter_frames() -> compute_optical_flow() -> VelocityField
                                                              |
                                          compute_vorticity() / total_circulation()
                                                              |
                                                  plot_velocity_heatmap()
```

## Quickstart

```python
from glas.analysis import analyze_convection

summary = analyze_convection(dataset.folder, heatmap_dir=Path("flow_maps"))
for frame_id, circulation in zip(summary.frame_ids, summary.circulations):
    print(frame_id, circulation)
```

or from the command line:

```bash
glas convection ~/glas_data/Run0001 --heatmap-dir flow_maps
```

`analyze_convection()` runs the whole pipeline over a dataset folder in
one call -- optical flow between every consecutive frame pair, plus
circulation and (optionally) one heat map per pair -- the same role
`track_dataset()` and `analyze_brazil_nut()` play for their own phases.

## Optical flow and the velocity field

`compute_optical_flow()` computes dense motion between two frames via
Farneback's method (`cv2.calcOpticalFlowFarneback`), then downsamples the
result to a regular grid (`grid_spacing` pixels apart) -- the raw
per-pixel flow is one 2-vector per pixel, too large and too noisy to be a
useful "field" on its own. Pass the real elapsed time between the two
frames (`elapsed_s`) to get a physically meaningful pixels/second
velocity; `analyze_convection()` does this automatically from each
frame's `host_timestamp_ns`, the same real-time-not-assumed-frame-rate
approach `glas.analysis.brazil_nut` uses.

## Vorticity and circulation

`compute_vorticity()` computes the curl of the velocity field
(`d(vy)/dx - d(vx)/dy`) at each grid point via finite differences --
positive values are counter-clockwise rotation (in image coordinates,
where `y` increases downward), negative clockwise. This is the classic
diagnostic for convection rolls: a roll shows up as a coherent region of
one sign.

`total_circulation()` approximates the area integral of vorticity over
the whole field (Stokes' theorem: circulation around a closed loop equals
the enclosed vorticity's integral) as a single scalar per frame pair, in
pixels²/second. It's consistent across different `grid_spacing` choices
for the same physical region -- sampling more coarsely doesn't change the
answer, only the resolution.

## Heat maps

`plot_velocity_heatmap()` renders a velocity field as a color-coded
background (`background="speed"` or `"vorticity"`) with an optional
quiver (arrow) overlay showing direction, saved as a PNG. Like
`glas.analysis.brazil_nut`'s plots, this uses matplotlib's non-interactive
`"Agg"` backend, set at module import time, so it's always headless-safe.

## Testing

`compute_optical_flow()` is tested against synthetic textured images
(random noise, so Farneback has something to track -- a blank image has
no texture to lock onto) with a known uniform shift, checking both
direction and approximate magnitude. `compute_vorticity()`/
`total_circulation()` are tested against a hand-constructed, exact
solid-body-rotation field (`vx = -y, vy = x`), which has an exact,
constant vorticity of `2` everywhere -- since both components are linear
functions of the coordinates, finite differencing recovers the exact
analytic derivative with zero truncation error, giving an exact (not
approximate) expected value to assert against.
`plot_velocity_heatmap()`'s output is verified as a real, readable PNG.
`analyze_convection()` is tested end to end against a real on-disk
dataset with a synthetically shifting texture pattern; every error path
(empty dataset, single-frame dataset, non-increasing timestamps) is
exercised directly.

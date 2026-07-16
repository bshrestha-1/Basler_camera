# Packing Analysis

Phase 14 measures how densely particles fill the frame: packing fraction,
void fraction, and number density, per frame and (optionally) as a
spatial field across the image. It builds directly on
`glas.analysis.tracking_utils.detect_particles()` -- packing statistics
are per-frame, not per-trajectory, so no tracking step is needed.

```
glas.dataset.iter_frames() -> detect_particles() -> compute_packing_metrics() -> PackingMetrics
                                                          |
                                               compute_packing_field() -> plot_packing_heatmap()
```

## Quickstart

```python
from glas.analysis import analyze_packing

summary = analyze_packing(dataset.folder)
for frame_id, metrics in zip(summary.frame_ids, summary.metrics):
    print(frame_id, metrics.packing_fraction, metrics.void_fraction)
```

or from the command line:

```bash
glas packing ~/glas_data/Run0001 --field-dir packing_maps --plot packing.png
```

`analyze_packing()` runs the whole pipeline over a dataset folder in one
call -- particle detection and packing statistics for every frame, plus
(optionally) a spatial field and heat map per frame -- the same role
`track_dataset()` and `analyze_convection()` play for their own phases.

## Packing fraction, void fraction, and number density

`compute_packing_metrics()` takes a frame's detections (from
`detect_particles()`) and a region-of-interest area, and computes:

- **Packing fraction**: `sum(particle areas) / roi_area` -- the fraction
  of the region covered by particles.
- **Void fraction**: `1 - packing_fraction` -- the fraction that isn't.
- **Number density**: `particle_count / roi_area` -- particles per unit
  area.

Packing fraction is *not* clamped to `[0, 1]`. Overlapping or merged blob
detections (two particles touching, detected as one larger contour, or a
segmentation artifact) can legitimately push the sum of detected areas
above the ROI area. That's a valid, if surprising, computed result --
worth noticing in the output, not an error condition to crash on.

By default, `analyze_packing()` uses the whole frame as the region of
interest (`metadata.width * metadata.height`); pass `roi_area` to measure
packing within a smaller region instead (e.g. excluding a border strip).

## Spatial packing fields

`compute_packing_field()` bins detections onto a coarse grid
(`grid_spacing` pixels per cell) and computes packing fraction within
each cell independently -- useful for seeing *where* a bed is dense or
loose, not just the frame-wide average. Each detection's full area is
assigned to the single cell containing its centroid (not split across
cells it may straddle), which is simple, fast, and accurate as long as
particles are small relative to `grid_spacing`. Cells at the image
boundary are clipped to their actual pixel coverage rather than assumed
to be a full `grid_spacing x grid_spacing` square, since image dimensions
aren't generally exact multiples of the grid spacing.

Pass both `field_grid_spacing` and `field_dir` to `analyze_packing()` to
compute and save one field (as a heat map PNG) per frame.

## Heat maps and summary plots

`plot_packing_heatmap()` renders a `PackingField` as a color-coded heat
map (`cmap="magma"`, fixed `0..1` color scale so frames are directly
comparable), saved as a PNG. `plot_packing_summary()` plots packing
fraction and particle count over time in two stacked panels. Like
`glas.analysis.brazil_nut` and `glas.analysis.convection`'s plots, both
use matplotlib's non-interactive `"Agg"` backend, set at module import
time, so they're always headless-safe.

## Testing

`compute_packing_metrics()` is tested against hand-constructed detection
lists with known areas, asserting exact packing/void fraction and number
density math, including the deliberate no-clamping behavior above 1.0.
`compute_packing_field()` is tested against detections placed at known
grid cells, including boundary cells smaller than a full `grid_spacing`,
and against out-of-bounds centroids to confirm they clamp into the grid
rather than silently wrapping around via numpy's negative indexing.
`plot_packing_heatmap()`/`plot_packing_summary()`'s output is verified as
a real, readable PNG. `analyze_packing()` is tested end to end against a
real on-disk dataset with particles of known total area, verifying the
computed packing fraction matches expectation, plus the `roi_area`
override, `field_dir` output, and the empty-dataset error path.

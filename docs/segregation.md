# Segregation Analysis

Phase 15 measures how separated two particle populations are -- the
standard diagnostic for size segregation in a vibrated bidisperse
granular bed (e.g. large particles rising to the surface, small
particles sifting to the bottom). Particles are split into "large"/
"small" populations by an equivalent-radius threshold, then binned onto
a coarse spatial grid to compare each cell's local composition against
the bed's overall composition.

```
glas.dataset.iter_frames() -> detect_particles() -> split by radius
                                                            |
                                         compute_segregation_metrics() -> SegregationMetrics
                                                            |
                                              plot_segregation_summary()
```

## Quickstart

```python
from glas.analysis import analyze_segregation

summary = analyze_segregation(dataset.folder, plot_path=Path("segregation.png"))
for frame_id, metrics in zip(summary.frame_ids, summary.metrics):
    print(frame_id, metrics.segregation_index, metrics.mixing_entropy)
```

or from the command line:

```bash
glas segregation ~/glas_data/Run0001 --plot segregation.png
```

`analyze_segregation()` runs the whole pipeline over a dataset folder in
one call -- particle detection, size classification, and segregation
statistics for every frame, plus (optionally) a summary plot -- the same
role `track_dataset()` and `analyze_packing()` play for their own phases.
Like `glas.analysis.packing` (and unlike `glas.analysis.brazil_nut`), no
tracking step is needed: segregation is a per-frame spatial statistic.

## Classifying particles by size

Each frame's detections are split into "large" (`radius >= size_threshold`)
and "small" (`radius < size_threshold`) populations. By default,
`analyze_segregation()` computes `size_threshold` automatically as the
median equivalent radius across *every* detection in the whole dataset --
computed once and applied consistently to every frame, so a particle
near the boundary doesn't flip populations from frame to frame due to
detection noise. Pass `size_threshold` explicitly when you know the two
species' sizes in advance.

## Segregation index, mixing index, and mixing entropy

`compute_segregation_metrics()` bins both populations onto the same
coarse grid (`grid_spacing` pixels per cell, by detection centroid -- the
same binning `glas.analysis.packing` uses) and compares each occupied
cell's local fraction of "large" particles against the bed's overall
fraction:

- **Mixing index**: Lacey's mixing index, the standard measure from
  powder/granular mixing literature. It compares the actual variance of
  local composition across sampled cells against two reference points:
  the variance of a fully segregated bed (`p * (1 - p)`, where `p` is the
  overall large-particle fraction) and the variance expected from purely
  random mixing (`p * (1 - p) / n`, where `n` is the mean particle count
  per cell). `0` means fully segregated, `1` means mixed as well as
  random chance allows.
- **Segregation index**: `1 - mixing_index`, in `[0, 1]`. The complement,
  provided directly since "how segregated is it" and "how mixed is it"
  are both natural questions to ask.
- **Mixing entropy**: the mean local Shannon entropy of composition,
  normalized by the entropy of the bed's overall composition. Unlike the
  two indices above (which are both linear functions of variance), this
  is a genuinely different, nonlinear measure -- it can disagree with
  the indices in its details even though both move in the same
  direction, which is useful as a cross-check.

If only one population is present in a frame (e.g. `size_threshold` was
set too aggressively), there's nothing to segregate from: by convention,
`segregation_index=0.0`, `mixing_index=1.0`, and `mixing_entropy=0.0`
(zero entropy, a single outcome).

### Choosing `grid_spacing`

`grid_spacing` needs to be coarse enough that most occupied cells sample
more than one particle. At the limit of exactly one particle per
occupied cell, a fully segregated bed and a randomly mixed one look
statistically identical -- every occupied cell is trivially "pure" by
definition when it holds only one particle, so Lacey's index cannot
distinguish the two states. `compute_segregation_metrics()` detects this
degenerate case and raises `SegregationError` with a clear message
rather than returning a meaningless number.

## Plots

`plot_segregation_summary()` plots segregation index and mixing index
together (they're complements of each other, so overlaying them on one
axis makes the relationship visible) in the top panel, and mixing
entropy in the bottom panel, saved as a PNG. Like
`glas.analysis.packing`'s plots, this uses matplotlib's non-interactive
`"Agg"` backend, set at module import time, so it's always headless-safe.

## Testing

`compute_segregation_metrics()` is tested against hand-constructed grids
with an exact, analytically known composition -- a fully segregated
layout (every occupied cell purely one population) gives an exact
`mixing_index=0.0`, and a fully mixed layout (every cell's local
composition exactly matches the overall composition) gives an exact
`mixing_index=1.0` and `mixing_entropy=1.0` -- plus the single-population
convention, the degenerate one-particle-per-cell error path, and a
regression test guarding against spurious `numpy` warnings from
`log(0)` in cells that are spatially pure. `plot_segregation_summary()`'s
output is verified as a real, readable PNG. `analyze_segregation()` is
tested end to end against real on-disk datasets built with two
distinctly sized particle populations, both clearly segregated and
clearly mixed, confirming the segregation index moves in the expected
direction for each; `size_threshold` override, real-elapsed-time
`times_s`, and the empty-dataset/no-particles-detected error paths are
also exercised directly.

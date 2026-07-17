# Data-Taking Quality Assurance

Phase 20 adds two independent quality checks, covering the two moments a
bad recording is cheapest to catch:

```
glas.camera.Camera (connected) -> run_preflight_checks() -> HealthCheckResult
glas.dataset.iter_frames()     -> assess_recording_quality() -> RecordingQualityReport
```

## Preflight checks (`glas doctor`)

Runs *before* recording starts: disk space, camera connectivity,
exposure/gain not pinned at their device limits, a grabbed frame's focus
(variance of the Laplacian, a standard sharpness proxy) and exposure
level, and whether a spatial calibration file exists. None of these
guarantee a good recording, but each catches a specific, common way a lab
session gets wasted (out of disk space, camera asleep, lens cap on,
badly out of focus).

```bash
glas doctor ~/glas_data --calibration calibration.json
```

```python
from pathlib import Path
from glas.camera import Camera
from glas.qa import run_preflight_checks

camera = Camera()
camera.connect()
result = run_preflight_checks(camera, Path("~/glas_data").expanduser())
for item in result.items:
    print(item.name, item.passed, item.message)
print(result.all_passed)
```

Only checks that are actually meaningful given the camera's current
state run: exposure/gain sanity and the frame-based checks (focus,
exposure level) are skipped entirely -- not reported as failures -- if
the camera isn't connected yet.

## Post-recording quality (`glas qa`)

Runs *after* recording finishes, on top of
`glas.dataset.validate_dataset()`'s structural/checksum validation:
dropped frames and frame-rate jitter (replayed through
`glas.timestamps.TimestampLog`), and per-frame particle-count sanity (via
classical blob detection on a subsample, for large recordings) --
catching a recording that's structurally fine but scientifically
useless (e.g. the camera was pointed at a wall, or lost focus partway
through).

```bash
glas qa ~/glas_data/Run0001 --expected-fps 30
```

```python
from glas.qa import assess_recording_quality

report = assess_recording_quality(Path("~/glas_data/Run0001").expanduser(), expected_fps=30.0)
print(report.dropped_frame_count, report.fps_jitter_percent, report.mean_particle_count)
print(report.is_clean, report.warnings)
```

`--strict` makes `glas qa` exit with a nonzero status if any warning was
found -- useful for gating a processing pipeline on clean data without
breaking the default, informational usage.

## Design note

Both functions build on existing infrastructure rather than duplicating
it: `assess_recording_quality()` reuses `validate_dataset()` for
structural checks and `TimestampLog` for gap detection (the same class
`glas.acquisition.Acquisition` uses live during recording, just replayed
from the finalized frames instead), and `run_preflight_checks()` reuses
the same disk-space pattern `glas.monitor.PerformanceMonitor` already
uses.

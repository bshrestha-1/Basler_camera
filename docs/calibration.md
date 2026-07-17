# Spatial Calibration

Phase 20 adds pixel-to-physical-unit conversion: every earlier analysis
phase (tracking, Brazil nut, packing, segregation, AI segmentation)
reports sizes, positions, and velocities in pixels -- correct for
comparing frames within one recording, but not directly publishable,
since a paper needs millimeters, not "42.3 px". A
`glas.calibration.SpatialCalibration` bridges the gap: measure a known
real-world distance once per camera/lens/working-distance setup, and
every later pixel measurement converts to millimeters with one
multiplication.

```
known real-world distance -> calibrate_from_known_distance()/calibrate_from_checkerboard()
    -> SpatialCalibration -> save_calibration()/load_calibration()
```

Nothing elsewhere in GLAS requires a calibration to exist -- every
analysis function continues to work in pixels with no changes. A
calibration is an optional, explicit conversion step applied by the
caller wherever physical units are wanted.

## Two calibration methods

- **Two-point** (`calibrate_from_known_distance`): the simplest possible
  method. Lay a ruler (or any object of known length) in the field of
  view, identify two points on it in a captured frame, and give their
  pixel coordinates and the real distance between them. No special
  equipment.
- **Checkerboard** (`calibrate_from_checkerboard`): the standard
  machine-vision method. A checkerboard pattern of known square size
  gives many independent spacing measurements (every pair of adjacent
  internal corners), averaged into one more precise result, at the cost
  of needing a printed checkerboard.

```python
from glas.calibration import calibrate_from_known_distance, calibrate_from_checkerboard

# Two-point: a 50mm ruler spans from (100, 200) to (100, 340) in pixels.
calibration = calibrate_from_known_distance((100, 200), (100, 340), distance_mm=50.0)
print(calibration.mm_per_pixel)

# Checkerboard: 7x7 internal corners (an 8x8-square board), 25mm squares.
import cv2
image = cv2.imread("checkerboard.png")
calibration = calibrate_from_checkerboard(image, pattern_size=(7, 7), square_size_mm=25.0)
```

## Using a calibration

```python
mm = calibration.px_to_mm(42.3)          # length/distance
mm2 = calibration.px_to_mm_area(1800.0)  # area
px = calibration.mm_to_px(5.0)           # inverse
```

## Saving and loading

```python
from pathlib import Path
from glas.calibration import save_calibration, load_calibration

save_calibration(calibration, Path("calibration.json"))
calibration = load_calibration(Path("calibration.json"))
```

## CLI

```bash
glas calibrate two-point 100 200 100 340 50.0 --output calibration.json
glas calibrate checkerboard checkerboard.png 7 7 25.0 --output calibration.json
```

`glas doctor` (see [`qa.md`](qa.md)) can check that a calibration file
exists before a recording starts, via its `--calibration` option.

## Raises

Both calibration functions raise `glas.exceptions.CalibrationError` for
invalid input (non-positive distance/square size, coincident points, or
a checkerboard pattern that can't be found in the given image).
`save_calibration()`/`load_calibration()` raise the same for I/O or
parsing failures.

# Getting Started

This is a lab-operator's walkthrough of GLAS end to end: install it,
record an experiment, watch it live, keep an eye on performance, export
the footage, track particles, and find it again later. For per-module API
reference, see the other files in `docs/`; this page is about the
workflow that ties them together.

## 1. Install

```bash
git clone https://github.com/bshrestha-1/basler_camera.git
cd basler_camera
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Plug in the Basler ace acA640-750um camera. GLAS talks to it through
[pypylon](https://github.com/basler/pypylon), which is installed
automatically. Check GLAS can see it:

```bash
glas --version
```

If you don't have a physical camera handy (e.g. developing on a laptop),
pypylon ships a built-in camera emulator. Set `PYLON_CAMEMU=1` in your
shell before running any `glas` command and everything below works
identically against a simulated device.

## 2. Record an experiment

The fastest path is the CLI:

```bash
glas record ~/glas_data \
  --name "shaker sweep" \
  --tag brazil-nut --tag 60hz \
  --notes "shaker at 60 Hz, 4g" \
  --exposure-us 5000 --gain-db 6
```

This connects to the first camera it finds, creates the next `RunNNNN`
folder under `~/glas_data`, and records until you press Ctrl+C -- the
dataset is finalized safely no matter how the process ends (Ctrl+C, an
unrelated crash, or a `--duration` timer running out). Pass
`--duration 30` to record for a fixed 30 seconds instead of waiting for
Ctrl+C -- useful for scripted, unattended sessions.

Run `glas record --help` for the full flag list (`--serial` to pick a
specific camera when more than one is connected, `--format` to force
`hdf5` or `raw_binary` storage).

The same thing from Python, with more control (pause/resume, live
progress, exposure/gain tweaks mid-session):

```python
import time
from pathlib import Path
from glas.controller import RecorderController

controller = RecorderController(Path("~/glas_data").expanduser())
controller.connect()
controller.camera.exposure_time_us = 5000.0
controller.camera.gain_db = 6.0

with controller.graceful_shutdown() as shutdown:
    controller.start_recording(name="shaker sweep", tags=["brazil-nut", "60hz"])
    while not shutdown.is_set():
        time.sleep(0.1)
        progress = controller.progress()
        print(f"\r{progress.frame_count} frames, {progress.elapsed_seconds:.1f}s", end="")

controller.disconnect()
```

See [`recorder.md`](recorder.md) for pause/resume and the full state
machine.

## 3. Watch it live (optional)

Running in a script with a real display attached, attach a preview
window to the *same* recording -- it reads frames non-destructively, so
it can never slow the recording down or steal a frame the dataset writer
needs:

```python
from glas.preview import Preview
from glas.display import PreviewWindow

preview = Preview(recorder.buffer)
preview.crosshair = True
window = PreviewWindow(preview)
window.run()  # press "q" to close; the recording keeps going
```

See [`preview.md`](preview.md) for zoom, ROI overlay, and histograms.

## 4. Keep an eye on performance (optional)

For long recordings, watch whether the pipeline is keeping up and
whether the disk is filling up:

```python
from glas.monitor import PerformanceMonitor

monitor = PerformanceMonitor(recorder.buffer, data_dir=recorder.dataset.folder)
snapshot = monitor.sample()
print(snapshot.fps, snapshot.buffer_occupancy_percent, snapshot.disk_free_gb)
```

See [`monitor.md`](monitor.md) for what each field means.

## 5. Export the footage

```bash
glas export ~/glas_data/Run0001 ~/clips/run1.mp4 --format mp4 --fps 30
glas export ~/glas_data/Run0001 ~/clips/run1_frames --format tiff
```

TIFF/PNG preserve each frame's native pixel data exactly, for downstream
analysis; MP4/AVI/GIF are for sharing or quick review. `--start-frame`/
`--end-frame` export just a clip. See [`export.md`](export.md) for format
tradeoffs.

## 6. Detect and track particles

```bash
glas analyze ~/glas_data/Run0001
```

or from Python, for the full trajectory data (not just the summary the
CLI prints):

```python
from glas.analysis import track_dataset

history = track_dataset(Path("~/glas_data/Run0001").expanduser())
for track_id, observations in history.items():
    print(track_id, len(observations), "frames")
```

This is the foundation the Brazil nut analysis builds its own
measurements on top of (convection, packing, and segregation are the
exceptions -- convection works directly on bulk pixel motion, packing
and segregation work on per-frame detections, none of them need
tracking; see steps 8-10). See [`analysis.md`](analysis.md) for detection
parameters (thresholding, `invert`, area filters) and tracking parameters
(`max_distance`, `max_gap`).

## 7. Measure the Brazil nut effect

```bash
glas brazil-nut ~/glas_data/Run0001 --plot brazil_nut.png
```

or from Python, for the full trajectory data:

```python
from glas.analysis import analyze_brazil_nut

trajectory = analyze_brazil_nut(Path("~/glas_data/Run0001").expanduser())
print(f"rise time: {trajectory.rise_time_s} s")
print(f"mean velocity: {trajectory.mean_velocity_px_s:.1f} px/s")
```

The intruder (the larger particle) is identified automatically; height,
rise time, and velocity are computed from each frame's real timestamp,
not an assumed frame rate. See [`brazil-nut.md`](brazil-nut.md) for the
full design.

## 8. Measure convection

```bash
glas convection ~/glas_data/Run0001 --heatmap-dir flow_maps
```

or from Python:

```python
from glas.analysis import analyze_convection

summary = analyze_convection(Path("~/glas_data/Run0001").expanduser())
for frame_id, circulation in zip(summary.frame_ids, summary.circulations):
    print(frame_id, circulation)
```

Unlike Brazil nut analysis, this doesn't need particle tracking at all --
it measures dense optical flow (bulk pixel motion) directly, which is
exactly what convection rolls in a vibrated granular bed look like. See
[`convection.md`](convection.md) for velocity fields, vorticity, and
circulation.

## 9. Measure packing fraction

```bash
glas packing ~/glas_data/Run0001 --field-dir packing_maps --plot packing.png
```

or from Python:

```python
from glas.analysis import analyze_packing

summary = analyze_packing(Path("~/glas_data/Run0001").expanduser())
for frame_id, metrics in zip(summary.frame_ids, summary.metrics):
    print(frame_id, metrics.packing_fraction, metrics.void_fraction)
```

Like convection, this doesn't need particle tracking -- packing fraction,
void fraction, and number density are computed per frame directly from
that frame's detections. See [`packing.md`](packing.md) for spatial
fields and heat maps.

## 10. Measure segregation

```bash
glas segregation ~/glas_data/Run0001 --plot segregation.png
```

or from Python:

```python
from glas.analysis import analyze_segregation

summary = analyze_segregation(Path("~/glas_data/Run0001").expanduser())
for frame_id, metrics in zip(summary.frame_ids, summary.metrics):
    print(frame_id, metrics.segregation_index, metrics.mixing_entropy)
```

Also per-frame, no tracking needed: particles are split into "large"/
"small" populations by size (automatically, via the dataset-wide median
radius, or an explicit threshold), then compared spatially via Lacey's
mixing index and a normalized mixing entropy. See
[`segregation.md`](segregation.md) for the full design, including why
`grid_spacing` needs to be chosen carefully.

## 11. Find it again later

```bash
glas experiment list ~/glas_data --tag brazil-nut
glas experiment show ~/glas_data Run0001
```

or from Python:

```python
from glas.experiment import ExperimentManager

manager = ExperimentManager(Path("~/glas_data").expanduser())
for summary in manager.search_experiments(tag="brazil-nut"):
    print(summary.run_id, summary.name, summary.frame_count)
```

See [`experiment.md`](experiment.md) for the full search API.

## Putting it together

`tests/test_integration.py` runs exactly this workflow end to end against
a real (emulated) camera as part of the test suite -- record with a
preview and a performance monitor both attached concurrently, stop,
validate the dataset on disk, read every frame back, export to two
formats, track particles, and find the recording again by tag. If you
want to see every piece work together in one place, that test is the
reference.

## Where to go from here

- [`docs/index.md`](index.md) -- the full module map and roadmap.
- [`docs/camera.md`](camera.md) -- camera control (exposure, gain, ROI,
  pixel format) and hardware timestamps.
- [`docs/dataset.md`](dataset.md) -- on-disk layout, checksums, and
  `validate_dataset()`.
- [`docs/development.md`](development.md) -- contributing to GLAS itself.

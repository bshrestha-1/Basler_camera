# Getting Started

This is a lab-operator's walkthrough of GLAS end to end: install it,
record an experiment, watch it live, keep an eye on performance, export
the footage, track particles (classically or with AI), and find it again
later. For per-module API reference, see the other files in `docs/`; this
page is about the workflow that ties them together.

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

Everything in this walkthrough is also available from the desktop GUI
(`pip install -e ".[gui]"`, then `glas gui ~/glas_data`) -- a live
preview, camera and recording controls, an experiment metadata form,
hardware status, an analysis panel, a dataset browser, and a log
console, all built on the exact same backend these CLI commands use.
See [`gui.md`](gui.md) for the full panel-by-panel tour; the rest of
this page sticks to the CLI.

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
tracking; see steps 9-11). See [`analysis.md`](analysis.md) for detection
parameters (thresholding, `invert`, area filters) and tracking parameters
(`max_distance`, `max_gap`).

## 7. Detect, classify, and segment particles with AI

Classical blob detection above works well for well-lit, non-overlapping
particles of a single material. For poor lighting, heavy overlap, mixed
particle types, or automatic intruder identification, train (or load a
pretrained) YOLO model instead -- and refine any detection into an exact
pixel mask with SAM2:

```bash
glas ai detect ~/glas_data/Run0001 glass_beads.pt --csv tracks.csv
glas ai segment ~/glas_data/Run0001 glass_beads.pt --model-id facebook/sam2.1-hiera-large
```

or from Python:

```python
from glas.ai.yolo_detector import track_dataset_yolo

history = track_dataset_yolo(Path("~/glas_data/Run0001").expanduser(), "glass_beads.pt")
for track_id, observations in history.items():
    last = observations[-1]
    print(track_id, last.label, last.confidence, last.is_intruder)
```

`torch`/`ultralytics`/`sam2` are an optional dependency
(`pip install "glas[ai]"`) -- everything else in GLAS, including this
guide's earlier steps, works without them installed. `YoloDetection` is a
`Detection` subclass, so YOLO output plugs directly into the same
`ParticleTracker` from step 6 with no changes -- Brazil nut, packing, and
segregation all work unchanged with YOLO-sourced tracks. See
[`ai.md`](ai.md) for training a custom YOLO detector or fine-tuning SAM2
on your own material.

## 8. Measure the Brazil nut effect

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

## 9. Measure convection

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

## 10. Measure packing fraction

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

## 11. Measure segregation

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

## 12. Import and synchronize an accelerometer recording (optional)

If you recorded a PCB 352C22 accelerometer alongside the camera (e.g.
exported from a DAQ as a CSV), measure the vibration and align it with
the frames:

```bash
glas accelerometer analyze shaker_run.csv --plot signal.png
glas accelerometer sync shaker_run.csv ~/glas_data/Run0001 --output synced.csv
```

or from Python:

```python
from glas.accelerometer import analyze_vibration, import_accelerometer_csv, synchronize_with_frames
from glas.dataset import iter_frames

metrics = analyze_vibration(Path("shaker_run.csv").expanduser())
print(f"{metrics.frequency_hz:.1f} Hz, Gamma={metrics.gamma:.2f}")

recording = import_accelerometer_csv(Path("shaker_run.csv").expanduser())
frames = list(iter_frames(Path("~/glas_data/Run0001").expanduser()))
per_frame_g = synchronize_with_frames(recording, frames)
```

Synchronization here is a best-effort software alignment (assuming both
recordings started at roughly the same moment, or a known `offset_s`
apart). Wiring the camera to a hardware trigger line gives an exact
common zero point instead -- see step 13. See
[`accelerometer.md`](accelerometer.md) for the full design.

## 13. Drive and monitor lab hardware (optional)

If the experiment uses a Siglent SDG1032X function generator (directly,
or driving a Modal Shop 2025E shaker), a LabJack or National Instruments
DAQ, or a SCPI oscilloscope, GLAS can control them directly:

```bash
# Wire the camera to the generator's sync output for exact frame sync
glas trigger enable --source Line1 --activation RisingEdge

# Drive a shaker to a target Gamma (needs a calibration measured once
# for the specific shaker/amplifier/fixture combination)
glas shaker set-gamma 192.168.1.50 2.0 --volts-per-g 0.5 --calibration-frequency-hz 60 --start

# Read a DAQ channel, or query an oscilloscope
glas daq read labjack --channel 0
glas oscilloscope query 192.168.1.60 "*IDN?"
```

or from Python, for the same operations with more control:

```python
from glas.camera import Camera
from glas.hardware.scpi import SocketSCPITransport
from glas.hardware.shaker import ShakerCalibration, ShakerController
from glas.hardware.waveform_generator import SiglentSDG1032X

camera = Camera()
camera.connect()
camera.enable_hardware_trigger(source="Line1", activation="RisingEdge")

generator = SiglentSDG1032X(SocketSCPITransport("192.168.1.50"))
shaker = ShakerController(generator, ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0))
shaker.set_target_gamma(2.0)
shaker.start()
```

Every hardware class is built so its own command-building and
error-handling logic is unit-testable without physical hardware -- see
[`hardware.md`](hardware.md) for the full design, including why the
Modal Shop 2025E itself has no digital protocol to talk to.

## 14. Perfect your data: calibration, quality checks, comparison, and reports

Before recording, check the setup and (optionally) compute a
pixel-to-millimeter calibration:

```bash
glas doctor ~/glas_data --calibration calibration.json
glas calibrate two-point 100 200 100 340 50.0 --output calibration.json
```

After recording, check the data is actually usable before trusting it:

```bash
glas qa ~/glas_data/Run0001 --expected-fps 30
```

Once you have several recordings at different conditions (e.g. several
Gammas), compare a metric across all of them with real statistical
uncertainty:

```bash
glas compare ~/glas_data --parameter target-acceleration-g \
    --metric brazil-nut-rise-time --tag brazil-nut --plot sweep.png --csv sweep.csv
```

And generate a publication-ready summary of everything for one
recording:

```bash
glas report ~/glas_data/Run0001 report.html
```

Every `plot_*` function used throughout this walkthrough (Brazil nut,
convection, packing, segregation, vibration, and the comparison plot
above) already draws through a shared, colorblind-safe, 300-DPI
publication style -- pass a `.pdf`/`.svg` output path anywhere a plot
path is accepted for a vector figure instead of a raster PNG. See
[`calibration.md`](calibration.md), [`qa.md`](qa.md), and
[`publishing.md`](publishing.md) for the full design of all four pieces.

## 15. Find it again later

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
- [`docs/gui.md`](gui.md) -- the desktop GUI, panel by panel.
- [`docs/ai.md`](ai.md) -- training and using YOLO/SAM2 models.
- [`docs/calibration.md`](calibration.md) -- pixel-to-millimeter spatial
  calibration.
- [`docs/qa.md`](qa.md) -- preflight checks and post-recording quality
  assessment.
- [`docs/publishing.md`](publishing.md) -- publication-quality plots,
  statistics, comparison, and reports.
- [`docs/development.md`](development.md) -- contributing to GLAS itself.

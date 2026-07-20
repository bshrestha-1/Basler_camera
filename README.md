# GLAS — Granular Lab Acquisition System

GLAS is a production-quality acquisition and analysis platform for a
**Basler ace acA640-750um** camera, built for granular-material physics
research (Brazil nut effect, convection, packing fraction, segregation,
and related studies). It is developed in phases, each shipped as a fully
tested, documented, production-ready release.

**Current release: Phase 20 — Full Research Platform (v3.2.0).**
GLAS closes out its roadmap with the tools that turn a recording into a
publishable result: spatial calibration (px -> mm, two-point or
checkerboard), preflight and post-recording data-quality checks
(`glas doctor`/`glas qa`), a shared colorblind-safe 300-DPI publication
plot style applied across every existing analysis plot, proper
statistics (confidence intervals, linear regression) for repeated-trial
data, a generic multi-run parameter-sweep comparison engine
(`glas compare`), and self-contained HTML experiment reports
(`glas report`) with a matching GUI tab. See
[`docs/calibration.md`](docs/calibration.md),
[`docs/qa.md`](docs/qa.md), [`docs/publishing.md`](docs/publishing.md),
or [`CHANGELOG.md`](CHANGELOG.md) for details.

## Installation

```bash
git clone https://github.com/bshrestha-1/basler_camera.git
cd basler_camera
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the desktop GUI:

```bash
pip install -e ".[gui]"
```

For AI-based detection (YOLO) and segmentation (SAM2):

```bash
pip install -e ".[ai]"
```

For development (tests, linting, type checking):

```bash
pip install -e ".[dev]"
```

```bash
conda activate glas311
```

## Quickstart

```bash
glas --version

# Write a default configuration file
glas config init --path ~/.config/glas/config.yaml

# Validate it
glas config validate ~/.config/glas/config.yaml

# See the resolved settings (defaults merged with your file)
glas config show

# Record an experiment (records until Ctrl+C, or use --duration)
glas record ~/glas_data --name "shaker sweep" --tag brazil-nut

# Find it again, and export it
glas experiment list ~/glas_data --tag brazil-nut
glas export ~/glas_data/Run0001 ~/glas_data/Run0001.mp4 --format mp4

# Detect and track particles across the recording
glas analyze ~/glas_data/Run0001

# Detect, classify, and track particles with a trained YOLO model
glas ai detect ~/glas_data/Run0001 glass_beads.pt --csv tracks.csv

# Segment every particle's exact outline with YOLO + SAM2
glas ai segment ~/glas_data/Run0001 glass_beads.pt --model-id facebook/sam2.1-hiera-large

# Measure the Brazil nut effect: height, rise time, velocity, and a plot
glas brazil-nut ~/glas_data/Run0001 --plot brazil_nut.png

# Measure bulk convection: velocity field, circulation, and heat maps
glas convection ~/glas_data/Run0001 --heatmap-dir flow_maps

# Measure packing fraction, void fraction, and number density
glas packing ~/glas_data/Run0001 --field-dir packing_maps --plot packing.png

# Measure segregation index, mixing index, and mixing entropy
glas segregation ~/glas_data/Run0001 --plot segregation.png

# Analyze an accelerometer recording, and synchronize it with the frames
glas accelerometer analyze shaker_run.csv --plot signal.png
glas accelerometer sync shaker_run.csv ~/glas_data/Run0001 --output synced.csv

# Drive lab hardware: camera trigger, shaker, DAQ, oscilloscope
glas trigger enable --source Line1 --activation RisingEdge
glas shaker set-gamma 192.168.1.50 2.0 --volts-per-g 0.5 --calibration-frequency-hz 60
glas daq read labjack --channel 0
glas oscilloscope query 192.168.1.60 "*IDN?"

# Check the setup before recording, and compute a spatial calibration
glas doctor ~/glas_data --calibration calibration.json
glas calibrate two-point 100 200 100 340 50.0 --output calibration.json

# Check a recording's structural integrity and scientific data quality
glas qa ~/glas_data/Run0001 --expected-fps 30

# Compare a metric across many recordings, with real statistical uncertainty
glas compare ~/glas_data --parameter target-acceleration-g \
    --metric brazil-nut-rise-time --tag brazil-nut --plot sweep.png

# Generate a self-contained, publication-ready HTML report for one recording
glas report ~/glas_data/Run0001 report.html

# Or launch the desktop GUI (pip install -e ".[gui]" first)
glas gui ~/glas_data
```

See [`docs/getting-started.md`](docs/getting-started.md) for the full
lab-operator workflow these commands fit into.

```python
from glas import Settings, configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

settings = Settings.load()
settings.ensure_directories()
logger.info("GLAS ready, data_dir=%s", settings.data_dir)
```

See [`docs/configuration.md`](docs/configuration.md) for the full
configuration schema and file-resolution order.

### Camera

```python
from glas.camera import Camera
from glas.camera_validator import ROI

with Camera() as camera:
    info = camera.get_info()
    print(f"Connected to {info.model_name} (serial {info.serial_number})")

    camera.exposure_time_us = 5000.0
    camera.gain_db = 6.0
    camera.roi = ROI(width=640, height=480, offset_x=0, offset_y=0)
    camera.pixel_format = "Mono8"
```

`Camera` requires [pypylon](https://github.com/basler/pypylon) (installed
automatically as a dependency) and a connected Basler USB3 Vision camera.
See [`docs/camera.md`](docs/camera.md) for the full API, error handling,
and how the test suite exercises camera code without physical hardware
using pypylon's built-in camera emulation.

### Acquisition

```python
import time
from glas.camera import Camera
from glas.acquisition import Acquisition

with Camera() as camera:
    acquisition = Acquisition(camera, buffer_capacity=256)
    acquisition.start()
    time.sleep(2.0)
    acquisition.stop()

    print(acquisition.stats())
    frame = acquisition.buffer.pop(timeout=0)
    print(frame.frame_id, frame.image.shape)
```

See [`docs/acquisition.md`](docs/acquisition.md) for the ring buffer's
design, how frame loss is tracked, and memory-management guidance for
choosing `buffer_capacity`.

### Dataset writer

```python
import time
from pathlib import Path
from glas.camera import Camera
from glas.acquisition import Acquisition
from glas.dataset import Dataset, create_experiment_folder, resolve_dataset_format
from glas.metadata import DatasetMetadata
from glas.writer import DatasetWriter

with Camera() as camera:
    info = camera.get_info()
    folder = create_experiment_folder(Path("~/glas_data").expanduser())
    # DatasetMetadata.dataset_format only accepts a concrete "hdf5" or
    # "raw_binary" (it validates on construction); resolve "auto" first.
    dataset_format = resolve_dataset_format("auto")
    metadata = DatasetMetadata(
        dataset_format=dataset_format,
        camera_model=info.model_name,
        camera_serial=info.serial_number,
        pixel_format=camera.pixel_format,
        width=camera.roi.width,
        height=camera.roi.height,
        created_at_utc="2026-07-13T12:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format=dataset_format)

    acquisition = Acquisition(camera, buffer_capacity=256)
    writer = DatasetWriter(acquisition.buffer, dataset)

    writer.start()
    acquisition.start()
    time.sleep(10.0)
    acquisition.stop()
    writer.stop()  # drains the buffer, then finalizes the dataset
```

`Dataset` requires [h5py](https://www.h5py.org/) (installed automatically
as a dependency) for its default HDF5 storage format, with a raw binary
fallback when it isn't available. See [`docs/dataset.md`](docs/dataset.md)
for the on-disk layout, metadata schema, checksums, and
`validate_dataset()`.

### Recorder — the easiest way to actually record something

`RecorderController` wraps everything above into one call per recording:

```python
import time
from pathlib import Path
from glas.controller import RecorderController

controller = RecorderController(Path("~/glas_data").expanduser())
controller.connect()

with controller.graceful_shutdown() as shutdown:
    controller.start_recording(
        notes="shaker at 60 Hz, 4g", name="shaker sweep", tags=["brazil-nut", "60hz"]
    )
    while not shutdown.is_set():
        time.sleep(0.1)
        progress = controller.progress()
        print(f"\r{progress.frame_count} frames, {progress.elapsed_seconds:.1f}s", end="")

controller.disconnect()
```

Pausing (`controller.pause_recording()`) and resuming
(`controller.resume_recording()`) continue writing into the same dataset
with no gap or duplication in frame numbering, and `graceful_shutdown()`
means Ctrl+C finalizes the dataset safely instead of losing it. See
[`docs/recorder.md`](docs/recorder.md) for the full state machine,
progress reporting, and graceful shutdown.

### Live preview

Watch a camera, or a recording already in progress, without ever
affecting it:

```python
from glas.preview import Preview
from glas.display import PreviewWindow

preview = Preview(recorder.buffer)  # or acquisition.buffer
preview.crosshair = True
preview.crosshair_position = (320, 240)

window = PreviewWindow(preview)
window.run()  # blocking display loop; press "q" to quit
```

`Preview` reads with `RingBuffer.peek()`, never `pop()`, so it can never
steal a frame a `DatasetWriter` still needs. See
[`docs/preview.md`](docs/preview.md) for zoom/histogram/FPS, the
`glas.preview`/`glas.display` module split, and why `PreviewWindow` checks
for a display before ever calling into OpenCV.

### Performance monitor

Watch whether the pipeline is keeping up and whether the host machine is
about to become the bottleneck, live, during a recording:

```python
import time
from glas.monitor import PerformanceMonitor

monitor = PerformanceMonitor(recorder.buffer, data_dir=recorder.dataset.folder)

while recording:
    snapshot = monitor.sample()
    print(
        f"\r{snapshot.fps:.1f} fps | buffer {snapshot.buffer_occupancy_percent:.0f}% | "
        f"dropped {snapshot.dropped_frame_count} | cpu {snapshot.cpu_percent:.0f}% | "
        f"disk free {snapshot.disk_free_gb:.1f} GB",
        end="",
    )
    time.sleep(1.0)
```

Like `Preview`, `PerformanceMonitor` only ever reads a `RingBuffer`'s
lock-free `stats()` snapshot, so it can never affect a recording or a
preview attached to the same buffer. See
[`docs/monitor.md`](docs/monitor.md) for what each field means and why
CPU/RAM are reported per-process rather than system-wide.

### Export

Turn a recorded dataset into a TIFF/PNG image sequence, an MP4/AVI video,
or an animated GIF:

```python
from pathlib import Path
from glas.export import export_dataset

export_dataset(dataset.folder, Path("recording.mp4"), format="mp4", fps=30.0)
export_dataset(dataset.folder, Path("frames_tiff"), format="tiff")

# Just a clip, frames 100-199, as a GIF:
export_dataset(
    dataset.folder, Path("clip.gif"), format="gif", fps=15.0,
    start_frame=100, end_frame=200,
)
```

TIFF/PNG preserve each frame's native pixel data exactly (no color
conversion); MP4/AVI/GIF convert to 8-bit BGR the way `glas.display`
already does for live preview. See [`docs/export.md`](docs/export.md) for
format details and `glas.dataset.iter_frames()`, the frame-reading
function export builds on.

### Experiment manager

Find recordings later by name, tag, or camera model:

```python
from pathlib import Path
from glas.experiment import ExperimentManager

manager = ExperimentManager(Path("~/glas_data").expanduser())
for summary in manager.search_experiments(tag="brazil-nut"):
    print(summary.run_id, summary.name, summary.frame_count)

one = manager.get_experiment("Run0001")
```

`controller.start_recording(name=..., tags=[...])` is all it takes to make
a recording discoverable this way -- no `DatasetMetadata` schema change,
just two reserved keys inside its existing, forward-compatible `extra`
field. See [`docs/experiment.md`](docs/experiment.md) for the full API.

### Analysis: particle detection and tracking

Detect and track particles across a recording -- the foundation the
Brazil nut analysis (Phase 12) builds its own measurements on top of
(convection, packing, and segregation are the exceptions: they work
directly from bulk pixel motion / per-frame detections, without
tracking):

```python
from glas.analysis import track_dataset

history = track_dataset(dataset.folder, max_distance=20.0)
for track_id, observations in history.items():
    print(f"track {track_id}: {len(observations)} frames")
```

or from the command line:

```bash
glas analyze ~/glas_data/Run0001
```

`detect_particles()` thresholds each frame (Otsu by default) and reports
each blob's centroid and equivalent radius; `ParticleTracker` links
detections across frames into trajectories, retiring a track after
`max_gap` consecutive frames without a match. See
[`docs/analysis.md`](docs/analysis.md) for the detection parameters and
why particle linking uses a greedy nearest-neighbor match rather than
pulling in `scipy` for the Hungarian algorithm.

### Brazil nut (intruder) effect analysis

Automatically identify the intruder and measure its height, rise time,
and velocity over real elapsed time:

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

The intruder is identified as the track with the largest mean radius --
the standard experiment setup seeds exactly one particle larger than the
rest. Height and velocity come from each frame's real timestamp, not an
assumed frame rate. See [`docs/brazil-nut.md`](docs/brazil-nut.md) for
the full design.

### Convection analysis

Measure bulk flow directly from pixel motion -- no particle tracking
needed:

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

`compute_optical_flow()` computes a velocity field between consecutive
frames via dense optical flow (Farneback's method); `compute_vorticity()`/
`total_circulation()` measure rotational flow, the classic diagnostic for
convection rolls. See [`docs/convection.md`](docs/convection.md) for the
full design.

### Packing analysis

Measure how densely particles fill the frame -- packing fraction, void
fraction, and number density, per frame and (optionally) as a spatial
field:

```python
from glas.analysis import analyze_packing

summary = analyze_packing(dataset.folder, field_grid_spacing=32, field_dir=Path("packing_maps"))
for frame_id, metrics in zip(summary.frame_ids, summary.metrics):
    print(frame_id, metrics.packing_fraction, metrics.void_fraction)
```

or from the command line:

```bash
glas packing ~/glas_data/Run0001 --field-dir packing_maps --plot packing.png
```

`compute_packing_metrics()` computes packing/void fraction and number
density from a frame's detections (no tracking needed);
`compute_packing_field()` bins them onto a coarse grid for a spatial map
of where the bed is dense or loose. See [`docs/packing.md`](docs/packing.md)
for the full design.

### Segregation analysis

Measure how separated two particle populations are -- no tracking
needed, splitting each frame's detections by size and comparing local
composition against the bed's overall composition:

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

`compute_segregation_metrics()` splits detections into "large"/"small"
by an equivalent-radius threshold (automatic, via the dataset-wide
median, or explicit) and computes Lacey's mixing index, its complement
(segregation index), and a normalized Shannon mixing entropy. See
[`docs/segregation.md`](docs/segregation.md) for the full design,
including why `grid_spacing` needs to be chosen carefully.

### Accelerometer import and synchronization

Import a PCB 352C22 accelerometer recording, compute its vibration
frequency, displacement amplitude, and Gamma, and align it with a
recorded camera dataset's frames:

```python
from glas.accelerometer import analyze_vibration, import_accelerometer_csv, synchronize_with_frames
from glas.dataset import iter_frames

metrics = analyze_vibration(Path("shaker_run.csv"), plot_path=Path("signal.png"))
print(f"{metrics.frequency_hz:.1f} Hz, Gamma={metrics.gamma:.2f}")

recording = import_accelerometer_csv(Path("shaker_run.csv"))
per_frame_g = synchronize_with_frames(recording, list(iter_frames(dataset.folder)))
```

or from the command line:

```bash
glas accelerometer analyze shaker_run.csv --plot signal.png
glas accelerometer sync shaker_run.csv ~/glas_data/Run0001 --output synced.csv
```

`compute_gamma()` returns the dimensionless vibration intensity
`Gamma = peak acceleration / g`, the standard control parameter for a
vibrated granular bed; `synchronize_with_frames()` finds the nearest
accelerometer sample in time for each frame, assuming the two recordings
started at the same moment unless an explicit `offset_s` is given (wiring
the camera to a hardware trigger, below, gives an exact common zero point
instead). See [`docs/accelerometer.md`](docs/accelerometer.md) for the
full design.

### Hardware integration

Camera hardware triggering, a Siglent SDG1032X function generator, a
Modal Shop 2025E shaker (driven via the generator), LabJack/NI DAQ
devices, and a generic SCPI oscilloscope:

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
```

or from the command line:

```bash
glas trigger enable --source Line1 --activation RisingEdge
glas shaker set-gamma 192.168.1.50 2.0 --volts-per-g 0.5 --calibration-frequency-hz 60
glas daq read labjack --channel 0
```

Every class is built so its own logic is unit-testable without physical
hardware: the SCPI-based classes take an injectable transport, the DAQ
classes defer importing their vendor SDK until connection, and camera
triggering extends the existing `Camera` and is tested against
pypylon's built-in emulator. See [`docs/hardware.md`](docs/hardware.md)
for the full design, including why the Modal Shop 2025E itself has no
digital protocol to talk to.

### AI: YOLO detection and SAM2 segmentation

Detect, classify, and automatically identify intruders with a trained
YOLO model, then refine any detection into an exact pixel mask with SAM2:

```python
from glas.ai.yolo_detector import track_dataset_yolo
from glas.ai.sam2_segmenter import Sam2Segmenter, compute_segmentation_summary
from glas.analysis.tracking_utils import detect_particles
from glas.dataset import iter_frames

history = track_dataset_yolo(dataset.folder, "glass_beads.pt")
for track_id, observations in history.items():
    last = observations[-1]
    print(track_id, last.label, last.confidence, last.is_intruder)

last_frame = list(iter_frames(dataset.folder))[-1]
segmenter = Sam2Segmenter(model_id="facebook/sam2.1-hiera-large")
segments = segmenter.segment_frame(last_frame.image, detect_particles(last_frame.image))
summary = compute_segmentation_summary(segments, last_frame.image.shape[:2])
print(summary.packing_fraction, summary.void_fraction, len(summary.contacts))
```

or from the command line:

```bash
glas ai detect ~/glas_data/Run0001 glass_beads.pt --csv tracks.csv
glas ai segment ~/glas_data/Run0001 glass_beads.pt --model-id facebook/sam2.1-hiera-large
```

`YoloDetection` is a `Detection` subclass, so YOLO output plugs directly
into the same `ParticleTracker` used above with no changes -- Brazil
nut, packing, and segregation all work unchanged with YOLO-sourced
tracks. Both models support full training pipelines too (`glas ai
prepare-yolo-dataset`/`train-yolo`, `glas ai prepare-sam2-dataset`/
`train-sam2`) -- see [`docs/ai.md`](docs/ai.md) for the full design,
including how to fine-tune SAM2 on your own material.

### Calibration, data quality, comparison, and reports

Convert pixel measurements to millimeters, check a recording is
trustworthy, compare a metric across many recordings with real
statistical uncertainty, and generate a publication-ready summary:

```python
from glas.calibration import calibrate_from_known_distance
from glas.qa import assess_recording_quality
from glas.experiment import ExperimentManager, get_physical_parameters
from glas.analysis import analyze_brazil_nut
from glas.analysis.comparison import compare_runs, plot_parameter_sweep
from glas.report import generate_report

calibration = calibrate_from_known_distance((100, 200), (100, 340), distance_mm=50.0)
print(calibration.mm_per_pixel)

report = assess_recording_quality(dataset.folder, expected_fps=30.0)
print(report.is_clean, report.warnings)

manager = ExperimentManager(dataset.folder.parent)
result = compare_runs(
    manager.search_experiments(tag="brazil-nut"),
    parameter_fn=lambda md: get_physical_parameters(md).target_acceleration_g,
    metric_fn=lambda folder: analyze_brazil_nut(folder).rise_time_s,
    parameter_name="Gamma", metric_name="Rise time (s)",
)
plot_parameter_sweep(result, Path("sweep.pdf"))

generate_report(dataset.folder, Path("report.html"))
```

or from the command line:

```bash
glas doctor ~/glas_data --calibration calibration.json
glas calibrate two-point 100 200 100 340 50.0 --output calibration.json
glas qa ~/glas_data/Run0001 --expected-fps 30
glas compare ~/glas_data --parameter target-acceleration-g --metric brazil-nut-rise-time --plot sweep.png
glas report ~/glas_data/Run0001 report.html
```

Every `plot_*` function across the whole project (including
`plot_parameter_sweep` above) draws through a shared, colorblind-safe,
300-DPI publication style (`glas.plotting`) -- pass a `.pdf`/`.svg`
output path anywhere a plot path is accepted for a vector figure. See
[`docs/calibration.md`](docs/calibration.md), [`docs/qa.md`](docs/qa.md),
and [`docs/publishing.md`](docs/publishing.md) for the full design.

## Project layout

```
src/glas/          Package source (import glas)
  ai/                Optional: YOLO detection, SAM2 segmentation (pip install glas[ai])
  gui/               Optional: PySide6/Qt6 desktop GUI (pip install glas[gui])
  calibration.py     Spatial (px -> mm) calibration
  qa.py              Preflight checks, post-recording quality assessment
  plotting.py        Shared publication-quality plot styling
  stats.py           Descriptive statistics, linear regression
  report.py          Self-contained HTML experiment reports
tests/              pytest unit tests (one file per module)
docs/               Documentation
pyproject.toml      Packaging, dependencies, tool configuration
requirements.txt    Runtime dependencies
CHANGELOG.md         Release history (Keep a Changelog format)
```

## Development

```bash
pytest                        # run tests
pytest --cov=glas             # with coverage
ruff check src tests          # lint
ruff format src tests         # format
mypy                          # type check
```

See [`docs/development.md`](docs/development.md) for the full developer
workflow and coding standards, and [`docs/index.md`](docs/index.md) for
the complete 20-phase roadmap (camera layer, acquisition, dataset storage,
recording, live preview, analysis, hardware synchronization, GUI, and
beyond).

## License

MIT — see [`LICENSE`](LICENSE).

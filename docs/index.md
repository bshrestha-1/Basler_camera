# GLAS Documentation

GLAS (Granular Lab Acquisition System) is a production-quality acquisition
and analysis platform built around a Basler ace acA640-750um camera, for
granular-material physics experiments (Brazil nut effect, convection,
packing fraction, segregation, and related studies).

The project is developed in phases; each phase ships as a fully tested,
documented, production-ready release before the next one starts.

## Current status: Phase 15 — Segregation Analysis (v1.5.0)

Phase 15 measures how separated two particle populations are -- the
standard diagnostic for size segregation in a vibrated bidisperse
granular bed. Particles are split into "large"/"small" populations by an
equivalent-radius threshold (automatic, via the dataset-wide median, or
explicit), then binned onto a coarse spatial grid to compare each cell's
local composition against the bed's overall composition. No tracking
step is needed -- like packing, segregation is a per-frame spatial
statistic. `glas.analysis.compute_segregation_metrics()` computes Lacey's
mixing index, its complement (segregation index), and a normalized
Shannon mixing entropy from a frame's classified detections;
`glas.analysis.plot_segregation_summary()` renders a time-series plot.
`glas.analysis.analyze_segregation()` runs the whole pipeline over a
dataset folder in one call, and `glas segregation` exposes it from the
command line.

- **Segregation index, mixing index, mixing entropy** (`glas.analysis.segregation`)
- **`glas segregation`** (`glas.cli`)

This release ships as v1.5.0, matching the roadmap table's own numbering
below exactly. See the `[1.5.0]` entry in `CHANGELOG.md` and
[`segregation.md`](segregation.md) for the full design.

Every phase before this one remains exactly what it was:

- **Packing fraction, void fraction, number density, spatial fields, heat maps** (`glas.analysis.packing`)
- **`glas packing`** (`glas.cli`)
- **Optical flow, velocity fields, vorticity, circulation, heat maps** (`glas.analysis.convection`)
- **`glas convection`** (`glas.cli`)
- **Brazil nut effect: automatic intruder identification, height, rise time, velocity, plots** (`glas.analysis.brazil_nut`)
- **`glas brazil-nut`** (`glas.cli`)
- **Particle detection and frame-to-frame tracking** (`glas.analysis`)
- **`glas analyze`** (`glas.cli`)
- **Production hardening: `glas record`/`experiment`/`export`, full-pipeline integration test** (`glas.cli`, `tests/test_integration.py`)
- **Experiment manager: search/list recordings by name, tag, camera model** (`glas.experiment`)
- **Export: TIFF/PNG sequences, MP4/AVI, GIF, frame-range selection** (`glas.export`)
- **Reading a finalized dataset's frames back** (`glas.dataset.iter_frames`)
- **Performance monitoring: FPS, queue usage, dropped frames, CPU/RAM, disk space** (`glas.monitor`)
- **Live preview: zoom, crosshair, ROI, FPS, histogram** (`glas.preview`, `glas.display`)
- **Recording orchestration: start/stop/pause/resume, progress, graceful shutdown** (`glas.recorder`, `glas.controller`)
- **Dataset storage: HDF5/raw binary, metadata, checksums, background writer** (`glas.dataset`, `glas.metadata`, `glas.timestamps`, `glas.writer`)
- **Frame acquisition: producer thread, ring buffer** (`glas.frame`, `glas.ringbuffer`, `glas.acquisition`)
- **Camera discovery, control, and validation** (`glas.camera`, `glas.camera_info`, `glas.camera_validator`)
- **Configuration management** (`glas.config`, `glas.settings`)
- **Logging** (`glas.logger`)
- **Error handling** (`glas.exceptions`)
- **Command-line interface** (`glas.cli`)
- **Unit tests** (`tests/`)

See [`getting-started.md`](getting-started.md) for a lab-operator's
workflow tying every phase together,
[`configuration.md`](configuration.md) for how configuration works,
[`camera.md`](camera.md) for the camera API,
[`acquisition.md`](acquisition.md) for the acquisition pipeline,
[`dataset.md`](dataset.md) for the dataset writer and reading frames back,
[`recorder.md`](recorder.md) for the recorder and graceful shutdown,
[`preview.md`](preview.md) for the live preview,
[`monitor.md`](monitor.md) for the performance monitor,
[`export.md`](export.md) for the export engine,
[`experiment.md`](experiment.md) for the experiment manager,
[`analysis.md`](analysis.md) for the analysis engine,
[`brazil-nut.md`](brazil-nut.md) for the Brazil nut effect analysis,
[`convection.md`](convection.md) for the convection analysis,
[`packing.md`](packing.md) for the packing analysis,
[`segregation.md`](segregation.md) for the segregation analysis, and
[`development.md`](development.md) for the developer workflow.

## Roadmap

| Phase | Version | Goal |
|-------|---------|------|
| 1 | v0.1 | Core infrastructure |
| 2 | v0.2 | Camera layer — detect, connect, configure the Basler camera |
| 3 | v0.3 | Image acquisition — producer thread, ring buffer, no disk writes |
| 4 | v0.4 | Dataset writer — HDF5 storage, metadata, checksums |
| 5 | v0.5 | Recorder — start/stop/pause/resume (first usable release) |
| 6 | v0.6 | Live preview — histogram, FPS, zoom, crosshair (shipped as v0.7.0) |
| 7 | v0.7 | Performance monitor — FPS, queue usage, CPU/RAM/disk (shipped as v0.8.0) |
| 8 | v0.8 | Export engine — TIFF, PNG, MP4, AVI, GIF (shipped as v0.9.0) |
| 9 | v0.9 | Experiment manager — automatic run folders, metadata (shipped as v0.10.0) |
| 10 | v1.0 | Production release — the version used to collect scientific data (shipped as v1.0.0) |
| 11 | v1.1 | Analysis engine — particle tracking (shipped as v1.1.0) |
| 12 | v1.2 | Brazil nut analysis (shipped as v1.2.0) |
| 13 | v1.3 | Convection analysis — optical flow, velocity fields (shipped as v1.3.0) |
| 14 | v1.4 | Packing analysis — packing fraction, void fraction, density (shipped as v1.4.0) |
| 15 | v1.5 | Segregation analysis — segregation index, entropy, mixing (this release, shipped as v1.5.0) |
| 16 | v1.6 | Accelerometer synchronization (PCB 352C22, Γ, frequency/amplitude) |
| 17 | v1.7 | Hardware integration — function generators, shakers, DAQ |
| 18 | v2.0 | GUI |
| 19 | v2.5 | AI analysis — YOLO / SAM2 |
| 20 | v3.0 | Full research platform |

Each phase's files, features, and acceptance criteria are tracked in the
project's development roadmap.

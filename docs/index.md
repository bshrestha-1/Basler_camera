# GLAS Documentation

GLAS (Granular Lab Acquisition System) is a production-quality acquisition
and analysis platform built around a Basler ace acA640-750um camera, for
granular-material physics experiments (Brazil nut effect, convection,
packing fraction, segregation, and related studies).

The project is developed in phases; each phase ships as a fully tested,
documented, production-ready release before the next one starts.

## Current status: Phase 20 — Full Research Platform (v3.0.1)

Phase 20 closes out the roadmap: perfecting data taking, analysis, and
publishable results. Spatial calibration (`glas.calibration`) converts
every earlier phase's pixel measurements into physical units (mm) from a
two-point or checkerboard measurement. Data-taking quality tooling
(`glas.qa`) adds preflight checks before recording (`glas doctor`: disk
space, camera connectivity, exposure/gain sanity, focus) and structural
plus scientific quality assessment after recording (`glas qa`: dropped
frames, frame-rate jitter, particle-count sanity), building on the
existing dataset validator and timestamp gap detector rather than
duplicating them. Every existing `plot_*` function across
`glas.analysis`/`glas.accelerometer` now draws through a shared,
colorblind-safe, 300-DPI publication style (`glas.plotting`). Proper
statistics (`glas.stats`: confidence intervals via Student's t, linear
regression) turn repeated-trial data into real uncertainty, and a
generic multi-run comparison engine (`glas.analysis.comparison`) sweeps
any metric against any `PhysicalParameters` field across many
recordings, built entirely on the existing experiment manager and
analysis functions rather than reimplementing them. `glas.report`
generates a self-contained HTML report -- every analysis, its summary
statistics, and a publication-styled plot, for one recording in one
file -- with the GUI's analysis panel gaining a matching **Report** tab.

- **Spatial calibration: two-point and checkerboard methods, save/load** (`glas.calibration`)
- **Preflight checks and post-recording quality assessment** (`glas.qa`)
- **Shared publication-quality plot styling (colorblind-safe, 300 DPI, vector export)** (`glas.plotting`)
- **Descriptive statistics with confidence intervals, linear regression** (`glas.stats`)
- **Generic multi-run parameter-sweep comparison** (`glas.analysis.comparison`)
- **Self-contained HTML experiment reports** (`glas.report`)
- **`glas doctor`/`qa`/`report`/`compare`/`calibrate`** (`glas.cli`)
- **Report analysis-panel tab** (`glas.gui`)

This release shipped as v3.0.0, matching the roadmap table's own numbering
below exactly; v3.0.1 is a follow-up patch closing a reproducibility gap
found in a full Phase 1-20 audit -- every recording's metadata now also
captures frame rate, ROI offset, and the full Phase 17 camera settings
(gamma, binning, flip, auto-exposure/auto-gain, trigger state), not just
exposure/gain/ROI size, so a recording is always fully reproducible from
its own `metadata.json` (see [`dataset.md`](dataset.md#reproducibility)).
See the `[3.0.0]`/`[3.0.1]` entries in `CHANGELOG.md` and
[`calibration.md`](calibration.md), [`qa.md`](qa.md), and
[`publishing.md`](publishing.md) for the full design.

Every phase before this one remains exactly what it was:

- **AI-based YOLO detection, SAM2 segmentation, training pipelines** (`glas.ai`)
- **`glas ai detect`/`prepare-yolo-dataset`/`train-yolo`/`segment`/`prepare-sam2-dataset`/`train-sam2`** (`glas.cli`)
- **Desktop GUI: live preview, camera/recording controls, experiment metadata, hardware status, analysis panel, dataset browser, log console** (`glas.gui`)
- **`glas gui`** (`glas.cli`)
- **Camera hardware triggering, waveform generator, shaker, oscilloscope, DAQ** (`glas.camera`, `glas.hardware`)
- **`glas trigger`/`glas waveform-gen`/`glas oscilloscope`/`glas shaker`/`glas daq`** (`glas.cli`)
- **Accelerometer import, vibration analysis, frame synchronization** (`glas.accelerometer`)
- **`glas accelerometer analyze`/`glas accelerometer sync`** (`glas.cli`)
- **Segregation index, mixing index, mixing entropy** (`glas.analysis.segregation`)
- **`glas segregation`** (`glas.cli`)
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
[`segregation.md`](segregation.md) for the segregation analysis,
[`accelerometer.md`](accelerometer.md) for accelerometer import and
synchronization, [`hardware.md`](hardware.md) for lab instrument
integration, [`gui.md`](gui.md) for the desktop GUI,
[`ai.md`](ai.md) for YOLO detection and SAM2 segmentation,
[`calibration.md`](calibration.md) for spatial calibration,
[`qa.md`](qa.md) for data-taking quality assurance,
[`publishing.md`](publishing.md) for publication-quality plots,
statistics, comparison, and reports, and
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
| 15 | v1.5 | Segregation analysis — segregation index, entropy, mixing (shipped as v1.5.0) |
| 16 | v1.6 | Accelerometer synchronization — PCB 352C22, Γ, frequency/amplitude (shipped as v1.6.0) |
| 17 | v1.7 | Hardware integration — function generators, shakers, DAQ (shipped as v1.7.0) |
| 18 | v2.0 | GUI (shipped as v2.0.0) |
| 19 | v2.5 | AI analysis — YOLO / SAM2 (shipped as v2.5.0) |
| 20 | v3.0 | Full research platform (this release, shipped as v3.0.0) |

Each phase's files, features, and acceptance criteria are tracked in the
project's development roadmap.

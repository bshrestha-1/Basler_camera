# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.1] - 2026-07-17

A full Phase 1-20 reproducibility audit found that `DatasetMetadata` did not
capture everything needed to reproduce how a recording was configured: the
frame rate cap, the ROI's offset (only its size was stored), and every
Phase 17 camera setting (gamma, binning, horizontal/vertical flip,
auto-exposure/auto-gain mode, whether the frame rate cap was enabled,
hardware trigger state). Fixed so every recording's `metadata.json` is now a
complete-enough snapshot to reproduce the capture configuration exactly, per
the project's reproducibility requirement.

### Added

- `DatasetMetadata.frame_rate_hz`, `.roi_offset_x`, `.roi_offset_y`, and
  `.camera_settings` (an open-ended dict for gamma/binning/flip/auto-modes/
  trigger state) fields, all with safe defaults so existing metadata files
  and datasets remain fully loadable.
- `glas.controller._capture_camera_settings()`: populates the new fields
  from the connected `Camera` at the start of every recording.
- `glas.report.generate_report()`'s metadata table now shows ROI offset,
  frame rate, and camera settings alongside exposure/gain, so the
  publishable HTML report itself demonstrates the recording is fully
  reproducible.

### Fixed

- `RecorderController.start_recording()` previously stored only
  `exposure_time_us`/`gain_db`/ROI size in a recording's metadata --
  frame rate, ROI offset, and every other camera setting were silently
  lost, meaning a recording could not be fully reproduced from its own
  metadata alone.

## [3.0.0] - 2026-07-17

Phase 20 — Full Research Platform. Closes out the 20-phase roadmap:
perfecting data taking, analysis, and publishable results. Adds spatial
calibration (pixel-to-millimeter conversion, two-point and checkerboard
methods), data-taking quality assurance (preflight checks before
recording, structural and scientific quality assessment after), a shared
publication-quality plotting style applied across every existing
analysis plot, proper statistics (confidence intervals via Student's t,
linear regression) for repeated-trial data, a generic multi-run
parameter-sweep comparison engine, and self-contained HTML experiment
reports. `scipy` and `cycler` become core dependencies (statistics and
publication plot styling respectively). Bumped to 3.0.0, matching the
roadmap.

### Added

- `glas.calibration` (new module): `SpatialCalibration`
  (`px_to_mm()`/`mm_to_px()`/`px_to_mm_area()`), `calibrate_from_known_distance()`
  (two-point, from a known real-world distance between two pixel points),
  `calibrate_from_checkerboard()` (from a checkerboard pattern of known
  square size, averaging many independent corner-spacing measurements),
  `save_calibration()`/`load_calibration()` (JSON persistence). Nothing
  elsewhere in GLAS requires a calibration to exist -- every analysis
  function continues to work in pixels with no changes.
- `glas.qa` (new module): `run_preflight_checks()` (disk space, camera
  connectivity, exposure/gain sanity, focus via variance of the
  Laplacian, calibration presence -- built for `glas doctor`, run before
  recording starts) and `assess_recording_quality()` (dropped frames and
  frame-rate jitter via `glas.timestamps.TimestampLog`, per-frame
  particle-count sanity via classical detection on a subsample -- built
  for `glas qa`, run after recording finishes, on top of
  `glas.dataset.validate_dataset()`'s structural/checksum validation).
- `glas.plotting` (new module): `PUBLICATION_PALETTE` (Okabe-Ito
  colorblind-safe qualitative palette), `apply_publication_style()`
  (consistent fonts, sizes, grid, 300 DPI), `style_axes()` (spine
  removal), `savefig_publication()`. Every existing `plot_*` function
  across `glas.analysis.brazil_nut`/`convection`/`packing`/`segregation`
  and `glas.accelerometer` now draws through this shared style; vector
  formats (`.pdf`/`.svg`) continue to work exactly as before via
  matplotlib's own extension-based format inference.
- `glas.stats` (new module): `describe()` (sample mean, standard
  deviation, standard error, confidence interval via Student's t
  distribution rather than a fixed z-score) and `linear_fit()` (ordinary
  least-squares regression: slope, intercept, standard errors, R²,
  p-value), both correctly handling the degenerate zero-variance case
  scipy itself returns as NaN for.
- `glas.analysis.comparison` (new module): `compare_runs()` (generic
  parameter-extractor/metric-extractor pattern grouping many recordings
  by a parameter value and summarizing a metric within each group via
  `glas.stats.describe()`, with an optional linear fit across group
  means; a recording whose metric extraction fails is skipped with a
  logged warning rather than aborting the whole sweep), `plot_parameter_sweep()`,
  `export_sweep_csv()`.
- `glas.report` (new module): `generate_report()` -- a self-contained
  HTML report per recording covering every analysis (tracking, Brazil
  nut, convection, packing, segregation, and optionally vibration) with
  summary statistics and a base64-embedded publication-styled plot per
  section; an individual analysis failing is shown as a skipped section
  rather than aborting the whole report.
- `glas.exceptions`: `CalibrationError`, `ReportError`.
- CLI: `glas doctor` (preflight checks), `glas qa` (post-recording
  quality, with `--strict` to exit nonzero on any warning), `glas report`,
  `glas compare` (`--parameter`/`--metric` over five built-in
  extractors each), `glas calibrate two-point`/`checkerboard`.
- GUI: the analysis panel gains a **Report** tab (dataset folder + output
  HTML path, backed by `glas.report.generate_report()` on the existing
  background-thread pattern).
- `src/glas/__init__.py`: exports every new public symbol above, plus
  `export_tracks_csv` (a pre-existing Phase 19 symbol that had been added
  to `glas.analysis.__all__` but not propagated to the top-level package).

### Fixed

- `glas.analysis.particle_tracking.export_tracks_csv()` is now
  re-exported from the top-level `glas` package (was previously only
  reachable via `glas.analysis`).
- `pyproject.toml`: added a `scipy.*` mypy override (no inline type
  stubs shipped).

## [2.5.0] - 2026-07-17

Phase 19 — AI Analysis: YOLO / SAM2. Adds AI-based particle detection,
classification, and pixel-exact segmentation alongside the existing
classical blob-detection pipeline: a trained YOLO model detects and
classifies every particle -- including automatic intruder identification
-- even under poor lighting or heavy overlap, and SAM2 refines each
detection into an exact pixel mask for area, perimeter, orientation,
aspect ratio, contact area between touching grains, packing fraction, and
void fraction. Both models support full training pipelines (dataset
preparation, annotation, configuration, training, validation, checkpoint
management, export) as well as inference-only use with pretrained
weights. `torch`/`ultralytics`/`sam2` stay an optional dependency group
(`pip install "glas[ai]"`) -- nothing outside `glas.ai` imports them, so
`import glas`, the CLI, and the GUI all work without them installed, and
every AI-backed CLI command/GUI tab shows a clear message naming exactly
which packages are missing rather than a raw import error. Bumped to
2.5.0, matching the roadmap.

### Added

- `glas.ai` (new subpackage):
  - `glas.ai.dependencies`: `import_torch()`/`import_ultralytics()`/
    `import_build_sam2()`/`import_sam2_image_predictor()` (lazy imports
    that raise `AIDependencyError` with an install hint instead of a raw
    `ImportError`), `missing_ai_packages()`/`describe_missing_ai_packages()`.
  - `glas.ai.yolo_detector`: `YoloDetection` (a
    `glas.analysis.tracking_utils.Detection` subclass carrying label,
    confidence, and intruder flag -- plugs directly into the existing
    `ParticleTracker` with no changes to it), `YoloParticleDetector`
    (wraps a trained `ultralytics` model for inference), and
    `track_dataset_yolo()` (the YOLO equivalent of
    `glas.analysis.track_dataset()`, same return shape).
  - `glas.ai.yolo_train`: `train_yolo()`, `validate_yolo()`,
    `export_yolo_model()`, and `YoloTrainingConfig`/`YoloTrainingResult`,
    wrapping `ultralytics`'s own training loop.
  - `glas.ai.annotation`: `auto_annotate_dataset()` (bootstraps YOLO
    training boxes from a recording via the existing classical blob
    detector) and `prepare_yolo_dataset()` (train/val split, writes a
    YOLO-format `data.yaml`).
  - `glas.ai.sam2_segmenter`: `Sam2Segmenter` (wraps a SAM2 image
    predictor for box-prompted mask segmentation), `ParticleSegment`
    (mask + score + shape metrics -- a deliberate non-Pydantic exception
    like `glas.frame.Frame`, for the same reason), `compute_shape_metrics()`
    (area, perimeter, centroid, orientation, aspect ratio),
    `compute_contact_area()` (shared boundary between two touching
    particles), and `compute_segmentation_summary()` (packing fraction,
    void fraction, pairwise contacts for a whole frame).
  - `glas.ai.sam2_train`: `train_sam2()` (lightweight fine-tuning --
    freezes the image encoder, trains only the prompt encoder and mask
    decoder -- on box-prompted ground-truth masks), `auto_annotate_masks()`
    (bootstraps masks via classical contour detection), and
    `prepare_sam2_dataset()` (writes a training manifest).
- `glas.analysis.particle_tracking`: `TrackedParticle` gains `label`,
  `confidence`, and `is_intruder` fields (populated from a `YoloDetection`
  via `getattr()`, left at their classical defaults --
  `None`/`None`/`False` -- for classical tracking); new
  `export_tracks_csv()` writes the same CSV format for both classical and
  YOLO-sourced tracking output.
- `glas.exceptions`: `AIError`, `AIDependencyError`, `AIModelError`,
  `AIDatasetError`.
- CLI: `glas ai` subcommand group -- `detect`, `prepare-yolo-dataset`,
  `train-yolo`, `segment`, `prepare-sam2-dataset`, `train-sam2`; `glas
  analyze` gains a `--csv` option using the new `export_tracks_csv()`.
- GUI: the analysis panel gains "Detection (YOLO)" and "Segmentation
  (SAM2)" tabs (each with a second input field -- YOLO weights path / SAM2
  model id, the latter defaulting to `facebook/sam2.1-hiera-large` so it
  works out of the box with nothing but a recording), backed by
  `AnalysisViewModel.run_detection()`/`run_segmentation()` and a new
  `ai_dependency_missing` signal that shows a modal dialog
  (`glas.gui.ai_dialog`) naming any missing AI package instead of
  starting a background run that would only fail. Only Histograms
  remains a disabled placeholder tab.
- `pyproject.toml`: new `ai` extra (`torch>=2.2`, `ultralytics>=8.3`,
  `sam2>=1.1`), also added to `dev` so the full test suite runs without
  extra setup.
- `docs/ai.md` (new): full design, quickstarts for inference and
  training, CLI/GUI usage, and integration with the existing tracking
  pipeline.

## [2.0.0] - 2026-07-16

Phase 18 — Desktop GUI. Adds a full PySide6/Qt6 desktop application
alongside the existing CLI, following an MVVM architecture: ViewModels
wrap the existing, Qt-free backend and translate it into Qt signals;
widgets contain no business logic and talk only to their ViewModel. The
GUI and CLI share the same backend end to end, so acquisition, recording,
and analysis logic exists only once. PySide6 stays an optional dependency
(`pip install glas[gui]`) -- nothing outside `glas.gui` imports it, so
`import glas` and every other CLI command work without it installed.
Bumped to 2.0.0 (a new top-level interface, not a breaking change to any
existing API).

### Added

- `glas.gui` (new subpackage):
  - `glas.gui.theme`: `dark_palette()`/`apply_theme()` -- a dark palette in
    the style of Qt Creator/Basler pylon Viewer, toggleable at runtime.
  - `glas.gui.logging_bridge.QtLogHandler`: bridges the existing
    `glas.logger` root logger into a Qt signal via composition (not
    multiple inheritance -- `logging.Handler.emit` and `QObject`'s own
    internal `emit` are genuinely incompatible), so every camera,
    recording, and export log line reaches the GUI with no per-module
    wiring.
  - `glas.gui.viewmodels` (six ViewModels, one per backend class/module):
    `CameraViewModel`, `RecordingViewModel` (adds GUI-only auto-stop by
    duration/frame-count, implemented the same way the CLI's own
    `glas record --duration` implements it -- polling and stopping once a
    target is reached, not a new `RecorderController` mode), `LiveFeedViewModel`,
    `HardwareStatusViewModel` (includes an open-ended `DeviceStatus`
    device registry so a future LabJack/NI DAQ/accelerometer/function-
    generator/amplifier integration can appear in the status panel by
    calling `register_device()`, with no changes to the ViewModel or
    widget), `DatasetViewModel`, and `AnalysisViewModel` (runs
    `glas.analysis`/`glas.accelerometer` calls on a background
    `QThread`, so a slow analysis never blocks the UI).
  - `glas.gui.widgets` (eight widgets assembled into `glas.gui.main_window.MainWindow`):
    `LivePreviewWidget` (zoom/pan/fit-to-window/100%, frame counter, FPS,
    histogram, crosshair, ROI selection, reference grid, wall-clock
    timestamp overlay -- rendered through the same `glas.display.render_frame`
    the CLI's own preview window uses), `CameraControlsWidget` (camera
    selection, pixel format, exposure/gain/gamma, frame rate, ROI,
    binning, image flip, auto exposure/auto gain, hardware trigger --
    every control's range/choices queried live from the connected
    camera), `RecordingControlsWidget` (start/stop/pause/resume, auto-stop
    by duration or frame count, output folder, progress bar, disk-space
    and estimated-remaining-time display, dropped-frame counter),
    `ExperimentMetadataWidget` (material, grain diameter/density,
    container geometry, fill depth, frequency, amplitude, target
    acceleration -- collected into the new `glas.experiment.PhysicalParameters`),
    `HardwareStatusWidget`, `AnalysisPanelWidget` (tabs wired to the real
    `track_dataset`/`analyze_brazil_nut`/`analyze_convection`/
    `analyze_packing`/`analyze_segregation`/`analyze_vibration` functions,
    each with a plot-export button where a `plot_*` function exists;
    Segmentation and Histograms are explicit, disabled placeholder tabs,
    since no backend function exists for either yet), `LogConsoleWidget`
    (level filtering, save to file), and `DatasetBrowserWidget` (search,
    thumbnail preview, metadata, export, delete, duplicate).
  - `glas.gui.main_window.MainWindow`: assembles every widget into
    dockable, movable, resizable panels around a central live preview,
    with a menu bar, status bar, dark-mode toggle, and `QSettings`-backed
    window layout save/restore. Owns the one piece of cross-widget
    orchestration in the GUI layer: a standalone live-preview-only
    `glas.acquisition.Acquisition` runs whenever the camera is connected
    but idle, and is synchronously released and handed off to
    `glas.recorder.Recorder`'s own ring buffer for the duration of a
    recording (`glas.preview.Preview` only ever `peek()`s, so this is
    always safe), switching back once it stops.
  - `glas.gui.app.main()`: constructs the `QApplication`, applies the
    saved theme, and shows `MainWindow`.
- `glas gui BASE_DIR` CLI command: launches the desktop GUI. Lazily
  imports `glas.gui.app`, printing an install hint (`pip install glas[gui]`)
  and exiting cleanly if PySide6 is not installed, rather than failing
  `import glas.cli` itself.
- `glas.experiment.PhysicalParameters`/`build_physical_parameters_extra()`/
  `get_physical_parameters()`: a fixed schema (experiment ID, operator,
  material, grain diameter/density, container geometry, fill depth,
  frequency, amplitude, target acceleration) for the scientific parameters
  `ExperimentMetadataWidget` collects, stored under a new reserved
  `DatasetMetadata.extra` key alongside the existing name/tags keys --
  usable from the CLI or any future caller, not GUI-specific.
- `glas.camera.Camera` gained introspection methods so callers (the GUI,
  or any future caller) can query valid ranges/choices without reaching
  into GenICam directly: `exposure_time_bounds_us()`, `gain_bounds_db()`,
  `gamma_bounds()`, `frame_rate_bounds_hz()`, `roi_bounds()`,
  `pixel_format_choices()`, `exposure_auto_choices()`,
  `gain_auto_choices()`, `trigger_source_choices()`,
  `trigger_activation_choices()`, and `temperature_celsius()` (`None` if
  the connected device does not expose a temperature sensor).
- `glas.preview.Preview.overlay_grid` and a matching `overlay_grid`/
  `timestamp_text` parameter on `glas.display.render_frame()`, for the
  live preview's reference-grid and timestamp overlays -- shared by the
  GUI and `glas.display.PreviewWindow` alike.
- `glas.controller.RecorderController.base_data_dir` (settable property):
  lets a long-lived controller (the GUI keeps one for its whole session)
  change where new experiment folders are created without reconstructing
  the controller.
- PySide6 (`gui` extra) and pytest-qt (dev dependency).

### Changed

- `glas.gui.widgets.recording_controls_widget.RecordingControlsWidget`
  accepts optional `extra_provider`/`before_start` hooks, used by
  `MainWindow` to merge `ExperimentMetadataWidget`'s parameters into every
  recording and to release the live-preview acquisition before one
  starts, without either widget needing a reference to the other.

## [1.7.0] - 2026-07-16

Phase 17 — Hardware Integration. Adds support for the lab equipment used
to drive and monitor a vibrated granular-material experiment: a Siglent
SDG1032X function generator, a Modal Shop 2025E shaker, LabJack and
National Instruments DAQ devices, a generic SCPI oscilloscope, and
Basler camera hardware triggering. Every class is built so its own
command-building and error-handling logic is unit-testable without
physical hardware attached.

### Added

- `glas.camera.Camera` gained hardware trigger support, extending the
  existing class rather than a separate module:
  - `enable_hardware_trigger()`/`disable_hardware_trigger()`/
    `is_hardware_triggered()`: configure the GenICam ``TriggerSelector``/
    ``TriggerSource``/``TriggerActivation``/``TriggerMode`` features Basler
    cameras expose for external triggering. Tested against pypylon's
    built-in camera emulator, exercising real GenICam node access.
- `glas.hardware` (new subpackage):
  - `glas.hardware.scpi`: `SCPITransport` (a minimal write/query/close
    protocol), `SocketSCPITransport` (a raw-TCP "SCPI-raw" transport,
    port 5025), and `SCPIInstrument` (the universal IEEE 488.2 commands
    -- identify, reset, clear status, self-test, operation-complete --
    shared by every SCPI-based class below). Every device-specific class
    is built on an injected transport, so command-building logic is
    unit-testable against a fake transport with no physical instrument
    or network access.
  - `glas.hardware.waveform_generator.SiglentSDG1032X`: `set_sine_wave()`/
    `set_frequency()`/`set_amplitude()`/`enable_output()`/
    `disable_output()`, using the `BSWV` command family documented in
    Siglent's SDG-X series programming guide.
  - `glas.hardware.oscilloscope.SCPIOscilloscope`: a generic wrapper
    adding only the universal IEEE 488.2 commands plus `query_float()`
    (a helper for the common "measurement query returns a bare number"
    pattern) -- deliberately not modeling any specific vendor's dialect,
    since no oscilloscope model was specified and SCPI syntax varies
    significantly between vendors even for basic operations.
  - `glas.hardware.shaker.ShakerCalibration`/`ShakerController`: drives a
    Modal Shop 2025E shaker by computing the drive voltage a measured
    calibration says will reach a target Gamma, then sending it to the
    waveform generator feeding the amplifier. The 2025E itself is an
    analog gain stage with no digital control interface, so this only
    ever talks to the generator -- documented explicitly rather than
    fabricating a protocol for hardware that has none.
  - `glas.hardware.daq.AnalogInputDAQ`/`LabJackDAQ`/`NiDAQ`: analog input
    for LabJack (`labjack-ljm`) and National Instruments (`nidaqmx`)
    DAQ devices. Neither vendor SDK is a hard GLAS dependency; each class
    defers importing its SDK until `connect()` (mirroring how
    `glas.camera` defers importing `pypylon`) and accepts the SDK module
    itself as an injectable constructor parameter, so tests can supply a
    fake module standing in for the real one.
- `glas trigger enable`/`disable`/`status`: camera hardware trigger
  control from the command line.
- `glas waveform-gen sine`: configure a waveform generator channel.
- `glas oscilloscope query`: send a raw SCPI query to an oscilloscope.
- `glas shaker set-gamma`: drive a shaker to a target Gamma.
- `glas daq read`: read a single analog input channel from a LabJack or
  NI DAQ.
- New exception types: `HardwareError` (base), `InstrumentConnectionError`,
  `InstrumentCommandError`.
- `docs/hardware.md` describing the design, including the
  test-without-hardware strategy used throughout the phase.

## [1.6.0] - 2026-07-16

Phase 16 — Accelerometer Synchronization. Imports a PCB 352C22
accelerometer recording (a CSV file exported by DAQ software), computes
the standard vibration diagnostics used in granular-material physics,
and aligns the recording's timeline with a recorded camera dataset's
frames.

### Added

- `glas.accelerometer` (new top-level module -- not under
  `glas.analysis`, since it's a new data source, not a frame-image
  analysis):
  - `import_accelerometer_csv()`: reads a CSV file with a header row,
    converting raw sensor voltage to acceleration in g via the
    accelerometer's sensitivity (configurable, since it varies by unit
    and is only correct when taken from the specific accelerometer's
    calibration certificate) or passing already-converted g values
    through unchanged. Raises `AccelerometerError` with a specific
    message for every malformed-input case: missing file, missing
    header, missing column, non-numeric value, too few rows, or
    non-increasing timestamps.
  - `compute_vibration_frequency()`: finds the dominant frequency via a
    real FFT of the (mean-removed) signal.
  - `compute_gamma()`: the dimensionless vibration intensity
    `Gamma = peak acceleration / g`, the standard control parameter for
    a vibrated granular bed -- equal to the peak measured acceleration
    expressed in g, since the accelerometer measures acceleration
    directly rather than displacement.
  - `compute_vibration_amplitude()`: recovers displacement amplitude, in
    meters, from peak acceleration and frequency, assuming sinusoidal
    motion.
  - `plot_vibration_signal()`: plots the time-domain acceleration
    signal, via matplotlib's non-interactive `"Agg"` backend.
  - `analyze_vibration()`: runs the whole frequency/amplitude/Gamma
    pipeline over a CSV file in one call, the same role
    `glas.analysis.analyze_packing()` plays for its own phase.
  - `synchronize_with_frames()`: finds the nearest accelerometer sample
    in time for each of a sequence of `Frame`s, via `numpy.searchsorted`
    (no new dependency). Assumes both recordings started at the same
    real-world moment by default; accepts an explicit `offset_s` to
    correct for a known difference. Exact hardware-triggered
    synchronization (a shared clock zero point) is Phase 17's concern --
    this is a best-effort software alignment for setups without that
    hardware yet.
  - `AccelerometerRecording`: the imported-recording result type. Like
    `glas.frame.Frame`, a plain dataclass rather than a Pydantic model,
    since its fields are numpy arrays.
  - `VibrationMetrics`: the frozen Pydantic result type bundling
    frequency, amplitude, Gamma, and peak acceleration.
- `glas accelerometer analyze`: runs the vibration analysis from the
  command line, with optional `--value-units`/`--sensitivity-mv-per-g`
  and `--plot`.
- `glas accelerometer sync`: synchronizes an accelerometer recording
  with a dataset's frames from the command line, writing one
  acceleration value per frame to a CSV.
- New exception type: `AccelerometerError`.
- `docs/accelerometer.md` describing the design, including the Gamma
  unit-cancellation reasoning and the software-vs-hardware
  synchronization distinction.

## [1.5.0] - 2026-07-16

Phase 15 — Segregation Analysis. Measures how separated two particle
populations are, on top of Phase 11's per-frame particle detection --
like packing, segregation is a per-frame spatial statistic, so no
tracking step is needed.

### Added

- `glas.analysis.segregation` (new module):
  - `compute_segregation_metrics()`: bins two particle populations
    ("large"/"small", classified by an equivalent-radius threshold) onto
    a shared coarse grid and compares each occupied cell's local
    composition against the bed's overall composition, computing:
    - Lacey's mixing index (the standard measure from powder/granular
      mixing literature): `0` for a fully segregated bed, `1` for a bed
      mixed as well as random chance allows.
    - Segregation index: `1 - mixing_index`, provided directly since
      "how segregated" and "how mixed" are both natural questions.
    - Mixing entropy: mean local Shannon entropy of composition,
      normalized by the entropy of the overall composition -- a
      genuinely different (nonlinear) measure from the two indices
      above, useful as a cross-check.
    Raises `SegregationError` when an occupied cell's average particle
    count is too low (at most 1 particle per cell) for the mixing index
    to statistically distinguish segregation from randomness, rather
    than returning a meaningless number.
  - `plot_segregation_summary()`: plots segregation/mixing index
    (overlaid, since they're complements) and mixing entropy over time,
    via matplotlib's non-interactive `"Agg"` backend.
  - `analyze_segregation()`: runs the whole pipeline (detection, size
    classification, and segregation metrics for every frame, optionally
    a summary plot) over a dataset folder in one call, the same role
    `track_dataset()` and `analyze_packing()` play for their own phases.
    Defaults the size-classification threshold to the median equivalent
    radius across every detection in the whole dataset -- computed once
    and applied consistently to every frame.
  - `SegregationMetrics`, `SegregationSummary`: frozen Pydantic result
    types.
- `glas segregation`: runs the segregation analysis from the command
  line, with optional `--size-threshold`, `--grid-spacing`, and `--plot`.
- New exception type: `SegregationError`.
- `docs/segregation.md` describing the design, including the Lacey mixing
  index math and the degenerate-grid-spacing failure mode.

## [1.4.0] - 2026-07-16

Phase 14 — Packing Analysis. Measures how densely particles fill the
frame, on top of Phase 11's per-frame particle detection -- packing
fraction, void fraction, and number density are per-frame statistics, so
no tracking step is needed.

### Added

- `glas.analysis.packing` (new module):
  - `compute_packing_metrics()`: computes packing fraction
    (`sum(particle areas) / roi_area`), void fraction
    (`1 - packing_fraction`), and number density
    (`particle_count / roi_area`) from a frame's detections. Deliberately
    does not clamp packing fraction to `[0, 1]` -- overlapping or merged
    blob detections can legitimately push it above 1.0, a valid computed
    result rather than an error condition.
  - `compute_packing_field()`: bins detections onto a coarse grid
    (`grid_spacing` pixels per cell) and computes packing fraction within
    each cell independently, for a spatial map of where a bed is dense or
    loose. Uses centroid-based binning (each detection's full area
    assigned to the cell containing its centroid); boundary cells are
    clipped to their actual pixel coverage rather than assumed to be a
    full `grid_spacing x grid_spacing` square. Centroid indices are
    defensively clamped into the grid, since `Detection.x`/`y` are
    unconstrained floats that could otherwise wrap around via numpy's
    negative indexing.
  - `plot_packing_heatmap()`: renders a `PackingField` as a color-coded
    heat map (`cmap="magma"`, fixed `0..1` scale) via matplotlib's
    non-interactive `"Agg"` backend.
  - `plot_packing_summary()`: plots packing fraction and particle count
    over time in two stacked panels.
  - `analyze_packing()`: runs the whole pipeline (detection + packing
    metrics for every frame, optionally a spatial field and heat map per
    frame) over a dataset folder in one call, the same role
    `track_dataset()` and `analyze_convection()` play for their own
    phases.
  - `PackingMetrics`, `PackingSummary`: frozen Pydantic result types.
  - `PackingField`: the per-frame spatial-field result type. Like
    `glas.analysis.convection.VelocityField`, a plain dataclass rather
    than a Pydantic model, since its numeric fields are numpy arrays.
- `glas packing`: runs the packing analysis from the command line, with
  optional `--roi-area`, `--field-grid-spacing`/`--field-dir`, and
  `--plot`.
- New exception type: `PackingError`.
- `docs/packing.md` describing the design, including the centroid-based
  binning tradeoff and the deliberate no-clamping behavior above 1.0.

## [1.3.0] - 2026-07-16

Phase 13 — Convection Analysis. Measures bulk flow in the granular bed
directly from pixel motion, via dense optical flow -- unlike Phase 11/12,
this doesn't need to detect or track individual particles.

### Added

- `glas.analysis.convection` (new module):
  - `compute_optical_flow()`: computes a dense velocity field between two
    consecutive mono frames via Farneback optical flow
    (`cv2.calcOpticalFlowFarneback`), downsampled to a regular grid
    (`grid_spacing` pixels apart) for a readable, size-bounded result.
    Converts pixel displacement to pixels/second using real elapsed time,
    the same real-time-not-assumed-frame-rate approach introduced in
    `glas.analysis.brazil_nut`.
  - `compute_vorticity()`: the curl of a velocity field
    (`d(vy)/dx - d(vx)/dy`, via finite differences) -- the classic
    diagnostic for convection rolls in vibrated granular media.
  - `total_circulation()`: the area-integrated vorticity across a whole
    velocity field, as a single scalar per frame pair -- consistent
    across different `grid_spacing` choices for the same physical region
    (Stokes' theorem).
  - `plot_velocity_heatmap()`: renders a velocity field as a color-coded
    heat map (`background="speed"` or `"vorticity"`) with an optional
    quiver overlay, via matplotlib's non-interactive `"Agg"` backend.
  - `analyze_convection()`: runs the whole pipeline (optical flow +
    circulation for every consecutive frame pair, optionally plotting
    each one) over a dataset folder in one call, the same role
    `track_dataset()` and `analyze_brazil_nut()` play for their own
    phases.
  - `VelocityField`: the per-frame-pair result type. Like `glas.frame.Frame`,
    a plain dataclass rather than a Pydantic model, since its numeric
    fields are numpy arrays (see `Frame`'s own docstring for why).
  - `ConvectionSummary`: the whole-recording result type (frozen Pydantic
    model).
- `glas convection`: runs the convection analysis from the command line,
  with an optional `--heatmap-dir` and `--heatmap-background`.
- New exception type: `ConvectionError`.
- `docs/convection.md` describing the design, including why circulation
  computed this way stays consistent across sampling resolutions.

## [1.2.0] - 2026-07-16

Phase 12 — Brazil Nut Analysis. Measures the classic Brazil nut effect on
top of Phase 11's tracked trajectories: automatic intruder identification,
height, rise time, velocity, and plots.

### Added

- `glas.analysis.brazil_nut` (new module):
  - `identify_brazil_nut()`: automatically identifies the intruder as the
    track with the largest mean equivalent radius across its whole
    trajectory -- the standard Brazil nut experiment setup seeds exactly
    one particle larger than the rest, so no manual selection is needed.
  - `compute_brazil_nut_trajectory()`: computes height (`frame_height - y`,
    so it increases as the particle rises), rise time (elapsed time to
    first reach `settle_fraction` of the frame height), and velocity
    (finite differences), all over *real* elapsed time derived from each
    observation's `host_timestamp_ns` -- not an assumed constant frame
    rate, since GLAS has no frame-rate concept anywhere else and real
    per-frame timestamps were already available in the pipeline. Raises
    `BrazilNutError` on non-increasing timestamps between consecutive
    observations, rather than crashing with a raw `ZeroDivisionError`.
  - `plot_brazil_nut_trajectory()`: saves a two-panel height/velocity PNG
    via matplotlib's non-interactive `"Agg"` backend (set at module
    import time) -- never requires or attempts to open a display.
  - `analyze_brazil_nut()`: runs the whole pipeline
    (`track_dataset()` -> `identify_brazil_nut()` ->
    `compute_brazil_nut_trajectory()`, optionally plotting) over a
    dataset folder in one call, the same role `track_dataset()` and
    `export_dataset()` play for their own phases.
  - `BrazilNutTrajectory`: the result type (frozen Pydantic model).
- `glas brazil-nut`: runs the Brazil nut analysis from the command line,
  with an optional `--plot` output.
- New exception type: `BrazilNutError`.
- `matplotlib` added as a runtime dependency (plots for this and future
  analysis phases).
- `docs/brazil-nut.md` describing the design.

### Changed

- `glas.analysis.TrackedParticle` gained a `host_timestamp_ns: int = 0`
  field (default preserves backward compatibility);
  `glas.analysis.ParticleTracker.update()` gained an optional
  `host_timestamp_ns` parameter that populates it;
  `glas.analysis.track_dataset()` now passes each `Frame`'s
  `host_timestamp_ns` through automatically. Existing callers that don't
  pass a timestamp are unaffected (it defaults to `0`).

## [1.1.0] - 2026-07-16

Phase 11 — Analysis Engine. The foundation every later analysis phase
(Brazil nut effect, convection, packing, segregation) builds its own
measurements on top of: particle detection and frame-to-frame tracking.

### Added

- `glas.analysis` (new subpackage):
  - `glas.analysis.tracking_utils.detect_particles()`: finds particle-like
    blobs in a mono frame -- Otsu or explicit thresholding, contour-based
    centroid and equivalent-radius (`sqrt(area / pi)`) sizing, `min_area`/
    `max_area` filters, and an `invert` flag for bright-on-dark vs.
    dark-on-bright particles.
  - `glas.analysis.tracking_utils.link_nearest()`: greedy nearest-neighbor
    matching between two frames' detections, within a `max_distance`.
    Deliberately not the Hungarian algorithm (no new `scipy` dependency);
    documented tradeoff in `docs/analysis.md`.
  - `glas.analysis.ParticleTracker`: links successive frames' detections
    into trajectories (`update()` called once per frame), spawning new
    tracks for unmatched detections and retiring tracks unmatched for
    more than `max_gap` frames. Track IDs are never reused.
  - `glas.analysis.track_dataset()`: runs the whole pipeline
    (`iter_frames()` -> `detect_particles()` -> `ParticleTracker`) over a
    finalized dataset folder in one call, the same role
    `glas.export.export_dataset()` plays for exporting.
- `glas analyze`: detects and tracks particles across a recorded dataset
  from the command line, printing a track-count/track-length summary.
- `tests/test_integration.py` extended to exercise `track_dataset()`
  against the same real (emulated) recording as every other phase.
- `docs/analysis.md` describing detection/tracking parameters and the
  greedy-vs-Hungarian tradeoff.

### Fixed

- `docs/development.md`'s module map had gone stale since Phase 5 --
  `preview.py`, `display.py`, `monitor.py`, `export.py`, and
  `experiment.py` were all missing. Brought up to date alongside the new
  `analysis/` subpackage.

## [1.0.0] - 2026-07-16

Phase 10 — Production Release. The version the roadmap calls "the version
used to collect scientific data." A hardening and integration release: no
new data-carrying module, just the audit and the glue connecting every
phase from 1 through 9 into one usable system.

### Added

- `glas record`: connects the camera, records (until Ctrl+C, or for a
  fixed `--duration`), and finalizes the dataset -- `--name`/`--tag`/
  `--notes` for experiment metadata, `--serial`/`--exposure-us`/
  `--gain-db` for camera setup, `--format` for the storage backend. Closes
  the gap `glas.cli`'s own module docstring had flagged since Phase 1
  ("Camera and recording commands are added in later development
  phases").
- `glas experiment list`/`glas experiment show`: browse recordings by
  name, tag, or camera model from the command line, wrapping
  `glas.experiment.ExperimentManager`.
- `glas export`: export a recorded dataset to TIFF/PNG/MP4/AVI/GIF from
  the command line, wrapping `glas.export.export_dataset()`.
- `tests/test_integration.py`: an end-to-end test proving every phase
  works together, not just in isolation -- record with a live preview and
  a performance monitor both attached concurrently to the same
  in-progress recording, stop, validate the dataset on disk, read every
  frame back, export to two formats, and find the recording again by tag.
- `docs/getting-started.md`: a lab-operator's workflow guide tying every
  phase together (install, record, preview, monitor, export, browse),
  distinct from the per-module API reference the rest of `docs/` provides.

### Changed

- `glas/__init__.py` now exports several previously-public-but-unexported
  symbols found during a full-repo audit ahead of this release:
  `load_metadata_json`/`save_metadata_json`, `NumericRange`/`ROIBounds`
  and the `validate_*` functions from `glas.camera_validator`, and
  `NAME_KEY`/`TAGS_KEY` from `glas.experiment`.
- `pyproject.toml` classifier updated from `Development Status :: 3 -
  Alpha` to `Development Status :: 5 - Production/Stable`.

### Audit

Every phase (1-9) was re-verified together before this release: full test
suite (348 tests before this phase's own additions), `ruff check`,
`ruff format --check`, and `mypy --strict` all clean; no TODO/FIXME/
placeholder code anywhere in `src/`; `version.py`/`pyproject.toml`/
`CHANGELOG.md` version numbers consistent; declared dependencies match
actual imports. See the "Changed" section above for the one real gap the
audit found and fixed.

## [0.10.0] - 2026-07-16

Phase 9 — Experiment Manager. Builds a searchable index across every
recording under a base data directory, and a convention for attaching a
human-readable name and tags to a recording at creation time.

### Added

- `glas.experiment.ExperimentManager`: `new_folder()` (thin wrapper around
  `glas.dataset.create_experiment_folder()`), `list_experiments()`
  (every finalized recording under a base directory, skipping folders
  with no or corrupt `metadata.json` rather than raising),
  `search_experiments(name_contains=..., tag=..., camera_model=...)`, and
  `get_experiment(run_id)`.
- `glas.experiment.ExperimentSummary`: a lightweight, searchable summary
  (`run_id`, `name`, `tags`, `notes`, `created_at_utc`, `frame_count`,
  `camera_model`, plus the full underlying `DatasetMetadata`).
- `glas.experiment.build_experiment_extra()`: builds a
  `DatasetMetadata.extra` dict carrying a reserved
  `experiment_name`/`experiment_tags` pair -- no schema change to
  `DatasetMetadata` itself, since `extra` has carried this exact
  forward-compatibility promise since Phase 4 (see its docstring).
- `RecorderController.start_recording()` gained `name`/`tags` parameters
  that call `build_experiment_extra()` internally, so a recording is
  discoverable through `ExperimentManager` without the caller having to
  import `glas.experiment` or hand-build the `extra` dict. Omitting both
  leaves `extra` exactly as before this phase (verified by a regression
  test against the existing `extra=` behavior).
- New exception type: `ExperimentNotFoundError`.
- `docs/experiment.md` describing the design and the reserved `extra` keys.

## [0.9.0] - 2026-07-15

Phase 8 — Export Engine. Turns a recorded dataset into common image and
video file formats for sharing, quick review, or external analysis tools.

### Added

- `glas.dataset.iter_frames(folder)`: reads a finalized dataset's frames
  back, in recorded order, as `Frame` objects -- supports both storage
  backends, streaming one frame at a time rather than loading the whole
  dataset into memory. A prerequisite for export, but useful generally
  (the dataset module previously had no read path at all, only write).
- `glas.frame.pixel_format_dtype()`: resolves a Basler mono pixel format
  name (`Mono8`/`Mono10`/`Mono12`/`Mono16`) to its numpy dtype -- needed
  because, unlike HDF5, raw binary storage never recorded a frame's dtype
  anywhere on disk.
- `glas.export.export_dataset(folder, output, format, ...)`: exports to
  `"tiff"`/`"png"` (one numbered file per frame via `cv2.imwrite`,
  preserving native pixel data and bit depth exactly), `"mp4"`/`"avi"`
  (via `cv2.VideoWriter`, reusing `glas.display`'s mono-to-BGR
  conversion), or `"gif"` (via Pillow, since OpenCV has no GIF encoder).
  Supports `start_frame`/`end_frame` range selection and an `overwrite`
  guard against clobbering existing output by accident. Returns an
  `ExportResult` (format, frame count, output path).
- New exception type: `ExportError`.
- `Pillow` added as a runtime dependency (GIF export only).
- `docs/export.md` describing each format's tradeoffs and the frame-range
  selection semantics; `docs/dataset.md` updated with a "Reading frames
  back" section for `iter_frames()`.

## [0.8.0] - 2026-07-14

Phase 7 — Performance Monitor. Continuous visibility into whether the
acquisition pipeline is keeping up, and whether the host machine is about
to become the bottleneck.

### Added

- `glas.monitor.PerformanceMonitor`: samples pipeline throughput and host
  resource usage (`sample() -> PerformanceSnapshot`). Reads only a
  `RingBuffer`'s lock-free `stats()` snapshot -- like `glas.preview.Preview`,
  never `pop()` -- so attaching a monitor to a buffer also being recorded
  from or previewed never affects either of them.
- `glas.monitor.PerformanceSnapshot`: `fps` (derived from
  `RingBufferStats.pushed`, a monotonic counter unaffected by what any
  consumer does with frames, averaged over a configurable window),
  `buffer_size`/`buffer_capacity`/`buffer_occupancy_percent`,
  `dropped_frame_count`, `cpu_percent`/`memory_used_mb`/`memory_percent`
  (this process's own footprint, via `psutil.Process`, not system-wide),
  and `disk_free_gb`/`disk_used_percent` (via `shutil.disk_usage` on a
  supplied data directory).
- `psutil` added as a runtime dependency; `types-psutil` for the dev extra.
- `docs/monitor.md` describing each field and the design choices (why
  CPU/RAM are per-process, why FPS is pushed-count-based rather than
  distinct-frame-based like `Preview.fps()`).

## [0.7.0] - 2026-07-13

Phase 6 — Live Preview. Watch a recording (or a camera before recording
starts) without ever competing with the dataset writer for frames.

### Added

- `glas.ringbuffer.RingBuffer.peek()`: a non-destructive, non-blocking read
  of the newest buffered frame, safe to call concurrently with
  `push()`/`pop()` from other threads. Sits alongside the existing
  destructive, ordered `pop()` -- a preview reading with `peek()` can never
  cause a dataset writer reading with `pop()` to lose or delay a frame.
- `glas.recorder.Recorder.buffer`: exposes the live acquisition ring buffer
  a preview can attach to while a recording is in progress.
- `glas.preview.Preview`: tracks the latest frame from a `RingBuffer`
  (`update()`), estimates live FPS from distinct frame timestamps (`fps()`),
  computes a pixel-intensity histogram (`histogram()`), and holds
  zoom/crosshair/ROI display state (`zoom`, `zoom_to()`, `reset_zoom()`,
  `crosshair`, `crosshair_position`, `show_roi`). Pure logic, no rendering,
  fully unit-testable without a display.
- `glas.preview.ZoomRegion` / `glas.preview.apply_zoom()`: a validated
  rectangular crop region and the pure function that applies it to an image.
- `glas.display`: OpenCV-backed rendering and windowing, split from
  `glas.preview` so only the final `cv2.imshow`/`cv2.waitKey` calls are
  untestable without a real display -- everything else (`render_frame()`,
  `render_histogram()`) is a pure function over numpy arrays. `PreviewWindow`
  wraps the actual OS window (`show_once()`, `wait_key()`, `run()`,
  context-manager support).
- New exception type: `DisplayError`, raised when a preview window cannot be
  shown.
- `opencv-python` added as a runtime dependency.
- `docs/preview.md` describing the peek-based non-competing design, the
  preview/display module split, and the headless-safety guard.

### Fixed

- Guarded every OS window call in `glas.display.PreviewWindow` behind a
  `_display_available()` pre-flight check. Without it, `cv2.imshow()` on the
  full `opencv-python` package *hangs indefinitely* (not raises) when no
  `DISPLAY`/`WAYLAND_DISPLAY` is set, unlike `opencv-python-headless`, which
  raises immediately -- a distinction that would otherwise only surface as a
  silent hang in a real headless deployment or CI run.

## [0.6.0] - 2026-07-13

A cross-cutting tooling upgrade, not a new roadmap phase: brings the project
in line with an updated tech-stack decision (Pydantic v2 for validation, Rich
for console logging, GitHub Actions for CI) on top of everything through
Phase 5.

### Changed

- **Every data-carrying type is now a validated Pydantic v2 model**
  (`model_config = ConfigDict(frozen=True)` in place of `@dataclass(frozen=True)`):
  `Settings`, `DatasetMetadata`, `CameraInfo`, `UsbDiagnostics`, `NumericRange`,
  `ROI`, `ROIBounds`, `RingBufferStats`, `AcquisitionStats`,
  `DatasetValidationResult`, `WriterStats`, `WallClockReference`, and
  `RecorderProgress`. `glas.frame.Frame` is the one deliberate exception --
  see its module docstring (hot path, never a validation boundary, holds a
  numpy array Pydantic can't meaningfully validate anyway).
- **`glas.config` is now purely a YAML loading/merging utility** --
  `validate_json()` and the `schema` parameter of `load_config()` are
  removed. Validation is now each Pydantic model's own job:
  `Settings.from_dict()` and `DatasetMetadata.from_dict()` translate a
  `pydantic.ValidationError` into `glas.exceptions.JSONValidationError` via
  the new `JSONValidationError.from_pydantic()` classmethod, preserving the
  same `.errors: list[str]` interface CLI and tests already relied on.
- **`glas.settings.CONFIG_SCHEMA` and `glas.metadata.METADATA_SCHEMA`
  (JSON Schema dicts) are removed**, replaced by the Pydantic models'
  own field constraints (`Literal` for enums, `Field(ge=..., min_length=...)`
  for ranges).
- **Pydantic model construction validates immediately**, unlike a plain
  dataclass. This surfaced one real, previously-latent bug: since
  `DatasetMetadata.dataset_format` only accepts a concrete `"hdf5"` or
  `"raw_binary"`, `RecorderController.start_recording()`'s
  `dataset_format="auto"` default could no longer be passed straight into
  `DatasetMetadata(...)` the way it silently was before (a plain dataclass
  never validated the intermediate, not-yet-resolved value).
  `glas.dataset.resolve_dataset_format()` is now a public function used to
  resolve `"auto"` to a concrete format *before* constructing metadata,
  in both `RecorderController` and `Dataset.create()` itself.
- **Console logging is now Rich-formatted** (`glas.logger`): colored,
  structured console output via `rich.logging.RichHandler`, with pretty
  tracebacks. File logging is unchanged (plain text via
  `RotatingFileHandler`) -- ANSI styling is noise once a log file is opened
  in an editor or grepped.
- `jsonschema` and `types-jsonschema` removed as dependencies; `pydantic` and
  `rich` added.

### Added

- `.github/workflows/ci.yml`: runs `pytest`, `ruff check`, `ruff format --check`,
  and `mypy` on every push and pull request against `main`, across Python
  3.10/3.11/3.12.
- 8 new tests covering the Pydantic migration and its bugfix, including a
  dedicated regression test proving `RecorderController.start_recording()`
  always resolves `dataset_format="auto"` to a concrete value before it
  reaches `DatasetMetadata`.

## [0.5.0] - 2026-07-13

Phase 5 — Recorder. The first usable release: actually records experiments,
end to end, with start/stop/pause/resume and graceful shutdown.

### Added

- `glas.recorder.Recorder`: orchestrates one recording session (an
  `Acquisition` and a `DatasetWriter` around an already-connected `Camera`
  and an already-created `Dataset`) with `start()`/`stop()`/`pause()`/`resume()`
  and a `RecorderState` state machine (`IDLE -> RECORDING -> PAUSED -> ... ->
  STOPPED`). `progress()` reports a live `RecorderProgress` snapshot: frame
  count, frames grabbed, dropped frames, bytes written, and elapsed recording
  time that excludes any time spent paused. Context-manager support finalizes
  the dataset on exit, including on an exception.
- `glas.controller.RecorderController`: the top-level entry point -- owns the
  camera connection, and `start_recording()` creates the experiment folder,
  builds `DatasetMetadata` from the camera's current settings, creates the
  `Dataset`, and starts a `Recorder` in one call. `graceful_shutdown()` is an
  opt-in context manager that installs SIGINT/SIGTERM handlers for its scope
  and finalizes any active recording safely on exit, however that exit
  happens (signal, exception, or normal completion), then restores whatever
  handlers were previously installed.
- New exception type: `RecorderError`.

### Changed

- `glas.acquisition.Acquisition`: frame numbering and `stats()` counters are
  now cumulative for the lifetime of an `Acquisition` instance rather than
  resetting on every `start()`. This was a real gap, not a style choice --
  without it, pausing and resuming a recording (`stop()` then `start()` on
  the same `Acquisition`, as `Recorder.pause()`/`resume()` do) would have
  restarted `frame_id` numbering from zero, so two different frames in the
  same dataset could both claim `frame_id=0`. Construct a new `Acquisition`
  if you genuinely want counters to reset.
- `docs/recorder.md` describing the Recorder/Controller API, the pause/resume
  frame-numbering guarantee, and graceful shutdown.

## [0.4.0] - 2026-07-13

Phase 4 — Dataset Writer. Saves experiments safely to disk:
`Camera -> Acquisition -> RingBuffer -> DatasetWriter -> Dataset`.

### Added

- `glas.metadata.DatasetMetadata`: the single canonical description of a recording
  (camera identity, pixel format, dimensions, timing, exposure/gain, frame count,
  free-text notes, and an open-ended `extra` dict for forward compatibility), with
  strict JSON Schema validation (`from_dict()`/`load_metadata_json()`) reusing
  `glas.config.validate_json` from Phase 1.
- `glas.dataset.Dataset`: HDF5 (primary) or raw-binary (fallback) frame storage,
  selected via `dataset_format="auto"|"hdf5"|"raw_binary"`. Frames stream to disk
  one at a time (`append_frame()`), never buffered in memory; `finalize()` writes
  `metadata.json` and SHA-256 `checksums.json` for every data file.
- `glas.dataset.create_experiment_folder()`: automatic `Run0001`, `Run0002`, ...
  folder numbering.
- `glas.dataset.validate_dataset()`: re-validates a dataset folder's checksums
  and structural consistency (frame counts, shapes) against its metadata,
  collecting every problem found rather than stopping at the first.
- `glas.writer.DatasetWriter`: a background thread that drains a `RingBuffer`
  into a `Dataset` without blocking the acquisition producer thread; `stop()`
  fully drains whatever is still buffered before finalizing, so no in-flight
  frame is silently discarded. `WriterStats` reports frames written, write
  errors, bytes written, and frames dropped upstream (detected via frame_id
  gaps in what was actually written).
- `glas.timestamps.TimestampLog`: incremental per-frame timestamp bookkeeping
  (O(1) gap detection per append) and `WallClockReference` for converting the
  monotonic `Frame.host_timestamp_ns` into an approximate wall-clock time.
- New exception types: `DatasetError`, `DatasetFormatError`, `DatasetIOError`,
  `WriterError`.
- `h5py` added as a runtime dependency.
- `docs/dataset.md` describing the on-disk dataset layout, both storage formats,
  and how checksums and validation work together.

## [0.3.0] - 2026-07-13

Phase 3 — Image Acquisition. Acquires frames from the camera into RAM as
fast as it can produce them; nothing is written to disk yet.

### Added

- `glas.frame.Frame`: a single acquired image (as a numpy array, decoupled from
  the camera driver's own buffer pool) plus its sequence number, host receive
  timestamp, and per-frame hardware timestamp.
- `glas.ringbuffer.RingBuffer`: a fixed-capacity, drop-oldest ring buffer of
  `Frame` objects. Built on `collections.deque`'s documented thread-safe
  `append()`/`popleft()` rather than an explicit lock, so the producer never
  blocks the hot path; `RingBufferStats` reports capacity, occupancy, and
  pushed/popped/dropped counters.
- `glas.acquisition.Acquisition`: runs a dedicated producer thread that grabs
  frames from an already-connected `Camera` and pushes them onto a
  `RingBuffer` (`Camera -> Acquisition -> RingBuffer`, all in memory).
  `start()`/`stop()`/`is_running`/`stats()` (`AcquisitionStats`: frames
  grabbed, grab errors, live buffer stats).
- `Camera` gained `start_grabbing()`, `stop_grabbing()`, `is_grabbing`, and
  `retrieve_frame()`, built on pylon's `GrabStrategy_OneByOne` so frame loss
  is only ever reported explicitly, never silently discarded by the driver.
- New exception type: `AcquisitionError`, for grab failures distinct from
  ordinary retrieval timeouts.
- `numpy` added as a runtime dependency.
- `docs/acquisition.md` describing the acquisition pipeline, ring buffer
  design tradeoffs, and how frame loss is tracked and reported.

## [0.2.0] - 2026-07-13

Phase 2 — Camera Layer. Talks to the Basler camera (detect, connect,
configure); nothing is saved to disk yet.

### Added

- `glas.camera_info`: `CameraInfo` / `UsbDiagnostics` dataclasses, `detect_cameras()`,
  and `get_usb_diagnostics()`, wrapping pypylon device enumeration and transport-layer
  link diagnostics.
- `glas.camera_validator`: pure, hardware-independent validation for exposure time,
  gain, pixel format, and region of interest (`ROI` / `ROIBounds`), collecting every
  violation rather than failing on the first.
- `glas.camera.Camera`: connect/disconnect (with context-manager support), validated
  `exposure_time_us` / `gain_db` / `roi` / `pixel_format` properties, `get_info()`,
  `get_usb_diagnostics()`, and hardware-timestamp support (`supports_hardware_timestamp`,
  `get_timestamp()`).
- New exception types: `CameraError`, `CameraDriverError`, `CameraNotFoundError`,
  `CameraConnectionError`, `CameraConfigurationError`, `CameraFeatureUnavailableError`.
- `pypylon` added as a runtime dependency; camera-touching tests run against pypylon's
  built-in emulated-camera transport layer (`PYLON_CAMEMU`), so the suite exercises
  real pypylon code paths without physical hardware, and skips gracefully if neither
  pypylon nor any camera (real or emulated) is available.
- `docs/camera.md` describing the camera API and testing approach.

## [0.1.0] - 2026-07-13

Phase 1 — Core Infrastructure. Establishes the project foundation; no camera
or acquisition code yet.

### Added

- Project scaffolding: `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE` (MIT).
- `glas.config`: YAML configuration loading, deep-merging, and JSON Schema validation.
- `glas.settings`: typed, validated `Settings` dataclass with default configuration and schema.
- `glas.logger`: rotating-file and console logging via `configure_logging` / `get_logger`.
- `glas.exceptions`: project-wide exception hierarchy rooted at `GLASError`.
- `glas.version`: single-source semantic version (`__version__`, `VERSION_INFO`).
- `glas.cli`: Typer-based command-line interface (`glas --version`, `glas config init|show|validate`).
- `pytest` unit test suite covering all Phase 1 modules.
- Developer and configuration documentation in `docs/`.

[Unreleased]: https://github.com/bshrestha-1/basler_camera/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/bshrestha-1/basler_camera/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/bshrestha-1/basler_camera/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/bshrestha-1/basler_camera/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bshrestha-1/basler_camera/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bshrestha-1/basler_camera/releases/tag/v0.1.0

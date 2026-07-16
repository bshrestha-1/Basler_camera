# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

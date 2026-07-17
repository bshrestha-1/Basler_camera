# Desktop GUI

Phase 18 adds a PySide6/Qt6 desktop application alongside the existing
CLI, for lab operators who prefer a windowed, dockable-panel workflow
over the terminal.

```
glas.<other modules>          (Model)      -- the existing, Qt-free GLAS backend
glas.gui.viewmodels           (ViewModel)  -- QObject subclasses wrapping backend
                                              classes, translating their plain
                                              Python API into Qt signals/slots
glas.gui.widgets               (View)      -- QWidget subclasses, no business logic
glas.gui.main_window.MainWindow             -- assembles every widget into docks
glas.gui.app.main()                         -- constructs QApplication, shows MainWindow
```

## Design principle: the GUI and CLI share one backend

Every ViewModel wraps exactly one backend class or module and holds no
logic of its own beyond adapting a synchronous Python API to Qt
signals/slots -- GenICam access, recording orchestration, and analysis
algorithms all stay exactly where the CLI already uses them
(`glas.camera`, `glas.controller`, `glas.preview`, `glas.monitor`,
`glas.experiment`, `glas.analysis`, `glas.accelerometer`). No widget
calls into the backend directly; each widget is constructed with (and
only talks to) one ViewModel. This means the GUI can never drift out of
sync with the CLI, and every ViewModel method is independently
unit-testable against the pypylon camera emulator with no display
required (`QT_QPA_PLATFORM=offscreen`, the default in `tests/conftest.py`).

PySide6 stays an optional dependency: nothing outside `glas.gui` imports
it, so `pip install glas` (without the `gui` extra) and every existing
CLI command work exactly as before. `glas.cli`'s `gui` command imports
`glas.gui.app` lazily, inside a `try`/`except ImportError`, printing an
install hint rather than making `import glas.cli` itself require Qt.

## Launching the GUI

```bash
pip install -e ".[gui,ai]"
glas gui ~/glas_data
```

`BASE_DIR` (the directory new experiment folders are created under) is
the GUI's only required argument, matching `glas record BASE_DIR`.

## Layout

The main window's central widget is the live camera preview -- the
largest panel, per the design brief. Every other panel is a dockable,
movable, resizable `QDockWidget`, arranged around it by default but
freely rearrangeable; the arrangement (and dark-mode setting) persists
across sessions via `QSettings` and can be reset from **View → Reset
Layout**.

### Live Preview (`LivePreviewWidget`)

Renders through `glas.display.render_frame()` -- the exact function
`glas.display.PreviewWindow` uses for the CLI's own preview window, so
overlays never look different between the two. A `QGraphicsView` handles
viewport zoom/pan/fit-to-window/100% (a pure display transform, distinct
from `Preview.zoom`'s server-side crop); toolbar toggles switch between
Pan, Crosshair (click to place), and Select ROI (drag) interaction
modes, plus independent Grid and Timestamp overlay toggles. Frame
counter, FPS, and a live histogram sit below the view.

### Camera Controls (`CameraControlsWidget`)

Camera selection/connect/disconnect, pixel format, exposure and gain
(with auto modes), gamma, frame rate, ROI (with a one-click "Full
Sensor" reset), binning, image flip, and hardware trigger source/
activation. Every control's valid range or choice list is queried live
from the connected camera (`Camera.exposure_time_bounds_us()`,
`pixel_format_choices()`, `trigger_source_choices()`, etc.) rather than
hard-coded, so the panel adapts to whatever device is actually
connected. "Test image mode" is a visible but permanently disabled
control with an explanatory tooltip: no GenICam `TestPattern` node is
exposed by the target hardware/emulator, and a control that looked
functional but silently did nothing would be worse than one that says
so.

### Recording Controls (`RecordingControlsWidget`)

Start/stop/pause/resume, optional auto-stop after a given duration or
frame count (implemented the same way the CLI's own `glas record
--duration` implements it -- polling and stopping once a target is
reached, not a new `RecorderController` mode), output folder, a name/
tags/notes identification block, a progress bar, live disk-free space,
an estimated-remaining-recording-time readout, a recording indicator,
and a dropped-frame counter.

### Experiment Metadata (`ExperimentMetadataWidget`)

Collects `glas.experiment.PhysicalParameters` (experiment ID, operator,
material, grain diameter/density, container geometry, fill depth,
frequency, amplitude, target acceleration) -- attached to every
recording alongside the existing name/tags/notes fields, which stay on
the Recording Controls panel rather than being duplicated here.

### Hardware Status (`HardwareStatusWidget`)

Camera connection, USB link speed, temperature (if the device exposes
one), live frame rate, exposure, gain, synchronization mode, ring buffer
occupancy, memory/CPU usage, storage remaining, and recorder state. An
"Other Devices" section renders whatever
`HardwareStatusViewModel.register_device()` has been told about -- a
future LabJack, NI DAQ, accelerometer, function generator, amplifier, or
environmental sensor integration appears here automatically, with no
changes to this widget.

### Analysis (`AnalysisPanelWidget`)

One tab per real analysis pipeline -- Tracking, Detection (YOLO), Brazil
Nut, Convection, Packing, Segregation, Segmentation (SAM2), Vibration,
Report -- each running on a background `QThread` so a slow analysis
never blocks the UI, with a plot-export button wherever a `plot_*`
function exists in the corresponding `glas.analysis`/`glas.accelerometer`
module. Detection and Segmentation take a second input field (YOLO
weights path / SAM2 model id) alongside the usual dataset-folder path and
are backed by `glas.ai` (see [`ai.md`](ai.md)) -- if
`torch`/`ultralytics`/`sam2` aren't installed, clicking Run shows a modal
dialog naming the missing packages instead of starting a background run
that would only fail. Report also takes a second field (the output HTML
path, defaulting to `report.html`) and is backed by `glas.report` (see
[`publishing.md`](publishing.md)) -- the report itself is the artifact,
so there is no plot-export button for that tab. Histograms is the one
remaining explicit, disabled placeholder tab: no backend function exists
for it yet.

### Dataset Browser (`DatasetBrowserWidget`)

Search/filter by name or tag, a thumbnail preview of an experiment's
first frame, full metadata (including any recorded `PhysicalParameters`),
and export/duplicate/delete (delete requires confirmation).

### Log Console (`LogConsoleWidget`)

Live INFO/WARNING/ERROR output from the entire backend -- camera,
recording, and export events all appear with no per-category wiring,
since every GLAS module already logs through the same `glas` root
logger `QtLogHandler` attaches to. Level filtering and save-to-file are
built in.

## Testing

Every ViewModel, widget, and `MainWindow` itself is covered by
pytest-qt tests (`tests/test_gui_*.py`), run offscreen
(`QT_QPA_PLATFORM=offscreen`) against the pypylon camera emulator --
no display or physical camera required. Run just the GUI suite with:

```bash
pytest tests/ -k gui
```

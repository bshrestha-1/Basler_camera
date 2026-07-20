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
pip install glas[gui]
glas gui ~/glas_data
```

`BASE_DIR` (the directory new experiment folders are created under) is
the GUI's only required argument, matching `glas record BASE_DIR`.

## Layout

The main window's central widget is the live camera preview, and by
design it dominates the window -- the imaging-software convention set by
pylon Viewer, ImageJ, Micro-Manager, NIS-Elements, and ZEN, where the
image is the primary focus and everything else is a narrow tool panel
around it. On a first launch the window defaults to 1600x1000 with the
preview claiming roughly 60% of the width and 65-70% of the height:
panels that are only glanced at occasionally (Experiment Metadata,
Hardware Status; Dataset Browser, Log Console) are tabified behind the
one most relevant during live operation (Recording Controls; Analysis)
rather than each claiming their own strip of space, and the bottom dock
group has a permanent height cap so it can never grow to compete with
the preview. Every panel is still a dockable, movable, resizable,
floatable `QDockWidget` -- reachable via its tab, `View`, or a drag to
float it -- and the *arrangement* (which panels are where, docked or
floating) is exactly as free as before; only the bottom group's maximum
height and the very first launch's default proportions are fixed. The
arrangement (and dark-mode setting) persists across sessions via
`QSettings` and can be reset from **View → Reset Layout**.

### Live Preview (`LivePreviewWidget`)

Renders through `glas.display.render_frame()` -- the exact function
`glas.display.PreviewWindow` uses for the CLI's own preview window, so
overlays never look different between the two. A `QGraphicsView` handles
viewport zoom/pan/fit-to-window/100% (a pure display transform, distinct
from `Preview.zoom`'s server-side crop); toolbar toggles switch between
Pan, Crosshair (click to place), and Select ROI (drag) interaction
modes, plus independent Grid and Timestamp overlay toggles. Frame
counter, FPS, and a live histogram sit below the view. Before a camera
has ever sent a frame (and again after a disconnect), the view area
shows an empty state -- a camera icon, "No Camera Connected", and "Click
Connect to begin acquisition." -- instead of a blank rectangle; it's
replaced by the live image the moment the first frame arrives.

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
so. The connection status is a colored dot (🔴 Not connected, 🟡
Connecting..., 🟢 Connected, 🔴 Error) rather than plain text -- see
[Status indicators and resource gauges](#status-indicators-and-resource-gauges).

### Recording Controls (`RecordingControlsWidget`)

Start/stop/pause/resume, optional auto-stop after a given duration or
frame count (implemented the same way the CLI's own `glas record
--duration` implements it -- polling and stopping once a target is
reached, not a new `RecorderController` mode), output folder, a name/
tags/notes identification block, a progress bar, live disk-free space,
an estimated-remaining-recording-time readout, a colored recording
indicator (gray ● Idle, red ● Recording, amber ● Paused), and a
dropped-frame counter.

### Experiment Metadata (`ExperimentMetadataWidget`)

Collects `glas.experiment.PhysicalParameters` (experiment ID, operator,
material, grain diameter/density, container geometry, fill depth,
frequency, amplitude, target acceleration) -- attached to every
recording alongside the existing name/tags/notes fields, which stay on
the Recording Controls panel rather than being duplicated here.

### Hardware Status (`HardwareStatusWidget`)

Camera connection (colored dot), USB link speed, temperature (if the
device exposes one), live frame rate, exposure, gain, synchronization
mode, ring buffer occupancy, memory/CPU usage, storage remaining, and
recorder state (colored dot). USB bandwidth, buffer occupancy, memory,
CPU, and storage are rendered as color-coded `QProgressBar` gauges
(green/amber/red by how full they are) rather than plain text -- see
[Status indicators and resource gauges](#status-indicators-and-resource-gauges)
below. An "Other Devices" section renders whatever
`HardwareStatusViewModel.register_device()` has been told about -- a
future LabJack, NI DAQ, accelerometer, function generator, amplifier, or
environmental sensor integration appears here automatically (each with
its own colored connected/disconnected dot), with no changes to this
widget.

### Status indicators and resource gauges

`glas.gui.status_indicators` is the one place colors and thresholds for
every status dot and resource gauge in the application are defined, so a
red dot always means the same thing everywhere it appears:

- `status_dot_html(color, text)` builds the rich text (`● text`, colored)
  used for every connection/recording/device status label -- green for
  healthy/connected, red for disconnected/error, amber for a
  transitional or cautionary state (connecting, paused), gray for idle.
- `update_resource_bar(bar, percent, display_text)` sets a
  `QProgressBar`'s fill level, color, and displayed text (e.g.
  `"14/256 (5%)"` instead of Qt's default `"5%"`) in one call. Colors
  come from `resource_bar_color()`: green below 70%, amber from 70-90%,
  red at 90% and above -- the same thresholds for every gauge (USB
  bandwidth, buffer occupancy, memory, CPU, storage), so a lab operator
  glancing at the panel reads red as "needs attention" consistently.

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

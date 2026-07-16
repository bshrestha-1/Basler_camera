# Recorder

Phase 5 is the first usable release: it actually records experiments,
end to end.

```
Camera --.
          v
Recorder -> Acquisition -> RingBuffer -> DatasetWriter -> Dataset
```

`RecorderController` is the entry point most code should use; `Recorder`
is the lower-level session object it creates and drives.

## Quickstart

```python
import time
from pathlib import Path
from glas.controller import RecorderController

controller = RecorderController(Path("~/glas_data").expanduser())
controller.connect()

controller.camera.exposure_time_us = 5000.0
controller.camera.gain_db = 6.0

controller.start_recording(notes="shaker at 60 Hz, 4g")
time.sleep(10.0)

controller.pause_recording()
time.sleep(2.0)  # swap something, take notes, whatever -- not recorded
controller.resume_recording()

time.sleep(10.0)
metadata = controller.stop_recording()
controller.disconnect()

print(f"Recorded {metadata.frame_count} frames to a dataset.")
```

`start_recording()` creates the next `RunNNNN` experiment folder (see
[`dataset.md`](dataset.md)), builds `DatasetMetadata` from the camera's
*current* settings (exposure, gain, ROI, pixel format), creates the
`Dataset`, and starts recording -- all in one call. Pass `name=`/`tags=`
to make the recording discoverable later through
`glas.experiment.ExperimentManager` (see [`experiment.md`](experiment.md)).

## Module layout

- `glas.recorder.Recorder` -- one recording session: start/stop/pause/resume,
  progress, a `RecorderState` state machine.
- `glas.controller.RecorderController` -- owns the camera connection, creates
  experiment folders and metadata, creates and tracks the current `Recorder`,
  and provides opt-in graceful shutdown.

## Pause and resume

Pausing stops the camera from grabbing (`Acquisition.stop()`); the dataset
stays open and the background writer keeps running, idle, until more frames
arrive. Resuming restarts acquisition into the *same* dataset.

Frame numbering picks up exactly where it left off -- pausing and resuming
never restarts `frame_id` at 0, so no two frames in a dataset ever share an
ID. (Before this phase, `Acquisition.start()` reset its frame counter every
time it was called; that was fine when acquisition only ever started once,
but pause/resume calls `stop()` then `start()` again on the *same*
`Acquisition`, so that reset had to go -- see the `[0.5.0]` entry in
`CHANGELOG.md`.)

`RecorderProgress.elapsed_seconds` tracks only time spent actively
recording -- time spent paused doesn't count.

## State machine

```
IDLE --start()--> RECORDING --stop()--> STOPPED
                     |    ^
                 pause()  resume()
                     v    |
                   PAUSED-'
                     |
                   stop()
                     v
                  STOPPED
```

Calling a method from the wrong state (e.g. `pause()` while `IDLE`, `start()`
twice) raises `RecorderError`. `stop()` works from either `RECORDING` or
`PAUSED`.

## Progress and the recording timer

```python
progress = controller.progress()  # None if nothing is recording
print(progress.state, progress.frame_count, progress.elapsed_seconds)
```

`RecorderProgress` is a live snapshot: `frame_count` (written to disk),
`frames_grabbed` (captured by the camera, may exceed `frame_count` if frames
are still in flight or were dropped), `dropped_frame_count`, `bytes_written`,
and `elapsed_seconds`.

## Graceful shutdown

A lab recording can run for a long time; someone *will* hit Ctrl+C
mid-experiment eventually. `RecorderController.graceful_shutdown()` is an
opt-in context manager -- it does not install signal handlers unless you
use it, since doing that unconditionally would be surprising behavior for
GLAS used as a library inside a larger application (a future GUI, for
instance):

```python
with controller.graceful_shutdown() as shutdown:
    controller.start_recording()
    while not shutdown.is_set():
        time.sleep(0.1)
# recording is already stopped and finalized here
```

The signal handler installed for the block's duration does the minimum
possible -- it only sets the yielded `threading.Event` -- and your own loop
is responsible for noticing it and returning control to the `with` block.
Whether the block exits because the event was set, because of an unrelated
exception, or just by completing normally, any recording still active when
it exits is stopped and its dataset finalized before the previous SIGINT/
SIGTERM handlers are restored -- so a script built this way never leaves a
half-written dataset behind because someone pressed Ctrl+C.

## Testing without physical hardware

As with earlier phases, `glas.recorder` and `glas.controller` tests run
against pypylon's emulated cameras (`PYLON_CAMEMU`, configured in
`tests/conftest.py`). The pause/resume frame-numbering guarantee and the
elapsed-time-excludes-pauses behavior are both exercised as real,
timing-based integration tests against a live (emulated) recording, not
mocked. Graceful shutdown is tested by actually raising `SIGINT` in-process
(`signal.raise_signal`) and confirming the handler fires, the event is set,
the recording stops, and the previous signal handlers are restored.

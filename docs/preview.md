# Live Preview

Phase 6 adds a way to *watch* a camera or an in-progress recording without
ever affecting it.

```
Camera --.
          v
Recorder -> Acquisition -> RingBuffer -> DatasetWriter -> Dataset
                                |
                                '-> Preview -> PreviewWindow (cv2.imshow)
```

`Preview` reads from the same `RingBuffer` a `DatasetWriter` is draining,
using `RingBuffer.peek()` instead of `RingBuffer.pop()`. That one choice is
the whole design: `peek()` never removes a frame, so a preview attached to
a live recording can never steal a frame the writer needs, slow the
recording down, or cause a drop on its behalf. If a preview is slow,
crashes, or is never attached at all, the recording pipeline is completely
unaffected.

## Quickstart

```python
import time
from glas.camera import Camera
from glas.acquisition import Acquisition
from glas.preview import Preview
from glas.display import PreviewWindow

with Camera() as camera:
    acquisition = Acquisition(camera, buffer_capacity=256)
    acquisition.start()

    preview = Preview(acquisition.buffer)
    window = PreviewWindow(preview)
    try:
        window.run()  # blocks until "q" is pressed
    finally:
        acquisition.stop()
```

Watching a live recording works the same way, reading from
`Recorder.buffer`:

```python
from glas.recorder import Recorder

recorder = Recorder(camera, dataset)
recorder.start()

preview = Preview(recorder.buffer)
window = PreviewWindow(preview)
window.run()  # the recording keeps writing to disk the entire time
```

## Module layout

- `glas.preview` -- pure logic, no rendering, no OS window: `Preview`
  (latest-frame tracking, FPS, histogram, zoom/crosshair/ROI state),
  `ZoomRegion`, `apply_zoom()`. Fully unit-testable without a display.
- `glas.display` -- OpenCV-backed rendering and windowing: `render_frame()`,
  `render_histogram()` (pure functions over numpy arrays, also testable
  without a display), and `PreviewWindow` (the one part of this phase that
  actually needs `cv2.imshow`/`cv2.waitKey`).

## Zoom, crosshair, ROI, FPS, histogram

```python
preview.zoom_to(factor=2.0, center=(320, 240), source_width=640, source_height=480)
preview.crosshair = True
preview.crosshair_position = (320, 240)
preview.show_roi = True
preview.reset_zoom()  # back to the full frame

print(preview.fps())  # frames per second, averaged over recent distinct frames

frame = preview.update()
counts = Preview.histogram(frame)  # pixel-intensity histogram, one bin per value
```

`fps()` only advances when `update()` sees a new `frame_id` -- calling
`update()` repeatedly against a buffer that hasn't produced a new frame
yet (a preview loop polling faster than the camera's frame rate) does not
skew the estimate.

## Why `PreviewWindow` checks for a display before ever calling `cv2.imshow`

`cv2.imshow()` behaves very differently depending on which OpenCV
distribution is installed when no display is available (no `DISPLAY` or
`WAYLAND_DISPLAY` on Linux, e.g. in CI or over SSH without X forwarding):

- `opencv-python-headless` has no GUI backend compiled in at all, so it
  raises a clean `cv2.error` immediately.
- `opencv-python` (the regular, full package -- what GLAS actually depends
  on, since a real preview window is the point of this phase) instead
  **hangs indefinitely with no exception**, because it has a GUI backend
  compiled in that then fails to find a display to attach to.

A `try/except` around `cv2.imshow()` only ever catches the headless
package's behavior, not the production one -- it would silently hang a
real deployment or a CI run. `PreviewWindow` checks `DISPLAY`/
`WAYLAND_DISPLAY` *before* every call that would touch the display
(`show_once()`, `wait_key()`, and therefore `run()`) and raises
`glas.exceptions.DisplayError` immediately instead. This is exercised as a
real, non-skipped test (`tests/test_display.py`), since a CI runner or this
sandbox guarantees no display is present.

## Testing

`glas.preview` and the pure functions in `glas.display`
(`render_frame()`, `render_histogram()`) are exercised as ordinary,
display-free unit tests -- no camera or display is needed. `PreviewWindow`
is tested for the one thing that's deterministic without a display: that
every method touching the window raises `DisplayError` immediately rather
than ever attempting (and hanging on) the underlying `cv2` call.

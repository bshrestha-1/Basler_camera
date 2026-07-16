# Image Acquisition

Phase 3 gets frames from the camera into RAM as fast as the camera can
produce them:

```
Camera -> Acquisition (producer thread) -> RingBuffer
```

Nothing is written to disk here. A dataset writer (Phase 4) or live
preview (Phase 6) will read from the ring buffer independently.

## Module layout

- `glas.frame.Frame` -- a single acquired image plus metadata.
- `glas.ringbuffer.RingBuffer` -- a fixed-capacity, drop-oldest buffer of
  frames.
- `glas.acquisition.Acquisition` -- the producer thread that ties a
  `Camera` to a `RingBuffer`.
- `Camera.start_grabbing()` / `stop_grabbing()` / `is_grabbing` /
  `retrieve_frame()` -- the pypylon-facing grab primitives `Acquisition`
  is built on (added to `glas.camera` in this phase).

## Quickstart

```python
import time
from glas.camera import Camera
from glas.acquisition import Acquisition

with Camera() as camera:
    acquisition = Acquisition(camera, buffer_capacity=256)
    acquisition.start()
    time.sleep(2.0)

    while True:
        frame = acquisition.buffer.pop(timeout=0)
        if frame is None:
            break
        print(frame.frame_id, frame.image.shape, frame.host_timestamp_ns)

    acquisition.stop()
    print(acquisition.stats())
```

`Acquisition` does not connect or disconnect the camera -- do that
yourself (or use `Camera` as a context manager, as above) before calling
`start()` and after calling `stop()`.

## Frame

Each `Frame` carries:

- `frame_id` -- a sequence number assigned by the producer, starting at
  0. A gap in `frame_id` values downstream means frames were dropped
  (see below).
- `image` -- a numpy array. It's copied out of the camera driver's own
  buffer immediately on retrieval, so it stays valid for as long as the
  `Frame` is referenced, independent of how many buffers the driver
  itself is cycling through.
- `pixel_format`, `width`, `height`, `nbytes`.
- `host_timestamp_ns` -- when this process received the frame
  (`time.perf_counter_ns()`); useful for measuring intervals, not
  wall-clock time.
- `device_timestamp_ticks` -- the camera's own per-frame timestamp, as
  reported by the driver. Whether this is meaningful depends on the
  camera/transport: if it never changes across frames, per-frame
  hardware timestamping isn't actually supported by the connected
  device, and only `host_timestamp_ns` should be trusted.

## Ring buffer

`RingBuffer` is a fixed-capacity, drop-oldest queue: when full, pushing a
new frame silently evicts the oldest buffered frame to make room, the
same way a live video ring buffer behaves. The producer thread must never
block on a slow consumer, so `push()` never waits.

**Why it doesn't use a lock.** `collections.deque` is documented by
CPython as thread-safe for `append()`/`popleft()` from opposite ends
without external locking, which is exactly the buffer's data path (one
producer calling `push`, one consumer calling `pop`). The only thing
`deque` doesn't report back is whether a given `append()` silently
evicted an item, which is needed for the `dropped` counter. `push()`
answers that with a length check immediately before the append; in rare
interleavings with a concurrent `pop()`, that check can be stale by the
time the append actually happens, making the `dropped` counter
imprecise by a small, self-correcting amount. This trades a small stats
imprecision for no lock contention on the hot path -- worthwhile at the
frame rates a fast camera can sustain. It never affects the buffer's
actual contents, ordering, or the accuracy of `frame_id` gaps.

`RingBuffer.pop(timeout=...)` blocks efficiently (via a
`threading.Event`, not polling) until a frame arrives or the timeout
elapses; `timeout=0` returns immediately, `timeout=None` waits
indefinitely.

## Tracking frame loss

Two independent counters distinguish *where* a frame was lost:

- `AcquisitionStats.grab_errors` -- the camera/driver failed to deliver a
  frame at all (e.g. an incomplete frame reported by the transport
  layer). Ordinary grab timeouts (no frame arrived within the configured
  window, nothing wrong) are not counted here.
- `AcquisitionStats.buffer.dropped` (via `RingBufferStats`) -- frames
  were successfully grabbed but evicted from the ring buffer because a
  consumer wasn't keeping up.

Both, together with `frame_id` gaps in whatever a consumer actually
receives, give a complete picture of frame loss for assessing whether a
recorded experiment is trustworthy.

## Memory management

`Acquisition` never grows without bound: the ring buffer has a fixed
`buffer_capacity`, and `Frame.image` arrays are independent copies rather
than references into the camera driver's own (much smaller) internal
buffer pool -- so a slow consumer causes old frames to be dropped, not
memory to grow unboundedly. Choose `buffer_capacity` based on how much
RAM you're willing to dedicate to smoothing over consumer stalls (a full
buffer of Mono8 frames at the camera's native resolution is
`width * height * buffer_capacity` bytes).

## Testing without physical hardware

As with Phase 2, `Acquisition` and `Camera`'s grab-related tests run
against pypylon's emulated cameras (`PYLON_CAMEMU`, configured in
`tests/conftest.py`) -- real producer-thread code, real pypylon grab
calls, no hardware required. `glas.frame` and `glas.ringbuffer` need
neither pypylon nor a camera at all; `RingBuffer`'s tests include genuine
multi-threaded producer/consumer scenarios to exercise its concurrency
behavior directly.

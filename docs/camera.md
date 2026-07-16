# Camera Layer

GLAS talks to a Basler camera (developed against an **ace acA640-750um**)
through [pypylon](https://github.com/basler/pypylon), Basler's official
Python binding for the Pylon Camera Software Suite. This phase covers
detection, connection, and parameter control only -- no frames are
acquired or saved. That's Phase 3 (acquisition) and Phase 4 (dataset
writer).

## Module layout

- `glas.camera_info` -- device discovery (`detect_cameras()`) and USB link
  diagnostics (`get_usb_diagnostics()`). No open camera required for
  discovery.
- `glas.camera_validator` -- pure validation logic (exposure time, gain,
  pixel format, region of interest) against explicit bounds. Has no
  dependency on pypylon, so it is trivially unit-testable.
- `glas.camera` -- the `Camera` class: connect/disconnect, and validated
  exposure/gain/ROI/pixel-format properties built on top of the two
  modules above.

## Quickstart

```python
from glas.camera import Camera
from glas.camera_validator import ROI

with Camera() as camera:
    info = camera.get_info()
    print(f"Connected to {info.model_name} (serial {info.serial_number})")

    camera.exposure_time_us = 5000.0   # microseconds
    camera.gain_db = 6.0                # dB
    camera.roi = ROI(width=640, height=480, offset_x=0, offset_y=0)
    camera.pixel_format = "Mono8"

    if camera.supports_hardware_timestamp:
        print(camera.get_timestamp())

    print(camera.get_usb_diagnostics())
# camera.disconnect() is called automatically on exit
```

Connecting to a specific camera by serial number (useful with more than
one camera attached):

```python
from glas.camera import Camera
from glas.camera_info import detect_cameras

for info in detect_cameras():
    print(info.serial_number, info.model_name)

camera = Camera()
camera.connect(serial_number="12345678")
```

## Validation

Every write to `exposure_time_us`, `gain_db`, `roi`, and `pixel_format`
first queries the *live* bounds from the connected device's GenICam node
map (min/max/step, or the supported pixel format list) and validates the
proposed value with `glas.camera_validator` before touching hardware.
Invalid values raise `glas.exceptions.CameraConfigurationError`; for
`roi`, every violated field is reported at once via the exception's
`errors` attribute, not just the first one found.

Bounds are queried dynamically rather than hardcoded, so the same code
works correctly against any Basler camera model, not just the
acA640-750um.

## Error handling

| Exception | Raised when |
|---|---|
| `CameraDriverError` | pypylon is not installed, or the transport layer itself fails |
| `CameraNotFoundError` | No camera is connected, or none matches a requested serial number |
| `CameraConnectionError` | Connecting while already connected, or using the camera before connecting |
| `CameraConfigurationError` | A proposed exposure/gain/ROI/pixel-format value is invalid |
| `CameraFeatureUnavailableError` | The connected device doesn't expose a requested feature (e.g. hardware timestamps) |

All inherit from `glas.exceptions.CameraError`, which itself inherits from
`GLASError`.

## Hardware timestamps

`Camera.supports_hardware_timestamp` reports whether the connected device
exposes a readable `Timestamp` node, or a `TimestampLatch` /
`TimestampLatchValue` command pair -- support and exact node names vary by
Basler model and firmware. `get_timestamp()` returns the raw device tick
count if supported, or raises `CameraFeatureUnavailableError` if not.

## USB diagnostics

`Camera.get_usb_diagnostics()` returns link speed, maximum bandwidth,
throughput limit, and USB speed mode where the device exposes them. Like
hardware timestamps, availability depends on the transport layer and
device model; unavailable fields come back as `None` rather than raising,
since USB diagnostics are informational.

## Testing without physical hardware

pypylon ships a built-in camera *emulation* transport layer: setting the
`PYLON_CAMEMU` environment variable to an integer *N* before pypylon's
transport-layer factory is first used exposes *N* virtual Basler cameras
that respond to the same GenICam node map calls as real hardware
(exposure, gain, ROI, pixel format, USB3-style throughput-limit nodes,
etc.).

`tests/conftest.py` sets `PYLON_CAMEMU=2` by default (unless already set),
so `pytest` exercises real pypylon code paths -- real device enumeration,
real GenICam validation errors from the SDK, real property round-trips --
without any camera plugged in. `glas.camera_validator` tests need no
camera or pypylon at all. If pypylon is not installed, or no camera (real
or emulated) is reachable, the hardware-dependent test modules skip
themselves with a clear reason instead of failing.

Emulated cameras do **not** support hardware timestamps or USB-specific
link diagnostics, so tests for those paths verify the graceful
"unsupported" behavior (`supports_hardware_timestamp is False`,
`get_timestamp()` raising `CameraFeatureUnavailableError`, USB diagnostic
fields coming back as `None`) -- exactly what would also happen on a real
camera or transport that lacks a particular feature.

"""The Frame data structure produced by the acquisition pipeline.

A :class:`Frame` pairs a single image (as a numpy array) with the
metadata needed to order and time-correlate it later: a monotonically
increasing sequence number assigned by the producer, the host-side time
it was retrieved, and the camera's per-frame hardware timestamp.

Unlike most other data-carrying types in GLAS, ``Frame`` is a plain
``dataclass`` rather than a Pydantic model. That's a deliberate,
narrow exception, not an oversight:

- It sits on the hottest path in the codebase -- one instance is
  constructed per grabbed frame, up to several hundred times a second
  for the target camera -- so it's worth avoiding validation overhead
  that buys nothing here.
- It's never a validation *boundary*. Every field is already
  correctly typed by construction (``image`` comes straight from a
  numpy array pypylon itself produced); nothing about a ``Frame`` is
  loaded from a file, a config, or other untrusted external input,
  which is where Pydantic earns its keep elsewhere in this codebase.
- Its one field that would be awkward for Pydantic --
  ``image: numpy.ndarray`` -- would need ``arbitrary_types_allowed``
  and still wouldn't get meaningful shape/dtype validation out of the
  box, for a field that's already correct by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class Frame:
    """A single acquired image plus acquisition metadata.

    Equality is identity-based (the default ``eq=False`` behavior), not
    field-based: comparing ``image`` arrays with ``==`` produces an
    element-wise array rather than a single bool, which would make an
    auto-generated ``__eq__`` unusable. Compare fields explicitly (e.g.
    ``numpy.array_equal(a.image, b.image)``) when you need content
    equality.

    Attributes
    ----------
    frame_id : int
        Sequence number assigned by the producer, starting at 0 and
        incrementing by one for every frame retrieved from the camera
        (including any later dropped from the ring buffer) -- a gap in
        ``frame_id`` values downstream indicates lost frames.
    image : numpy.ndarray
        Pixel data: shape ``(height, width)`` for mono formats, or
        ``(height, width, channels)`` for color formats. Owns its own
        memory (copied out of the camera driver's internal buffer), so
        it remains valid for as long as the ``Frame`` is referenced.
    pixel_format : str
        Pixel format of ``image``, e.g. ``"Mono8"``.
    host_timestamp_ns : int
        Host clock time the frame was retrieved, in nanoseconds since an
        unspecified reference point (:func:`time.perf_counter_ns`).
        Suitable for measuring intervals between frames, not wall-clock
        time.
    device_timestamp_ticks : int
        Camera hardware timestamp at frame capture, in device ticks, as
        reported by the driver. Tick frequency and epoch are
        device-specific; compare two frames from the same session to
        measure hardware-timed intervals independent of host scheduling
        jitter. If the connected device or transport layer does not
        support per-frame timestamping, this is typically a constant
        (often ``0``) for every frame rather than a meaningful value --
        check whether it varies across frames before relying on it.
    """

    frame_id: int
    image: NDArray[np.integer]
    pixel_format: str
    host_timestamp_ns: int
    device_timestamp_ticks: int

    @property
    def width(self) -> int:
        """Image width, in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Image height, in pixels."""
        return int(self.image.shape[0])

    @property
    def nbytes(self) -> int:
        """Size of :attr:`image` in bytes."""
        return int(self.image.nbytes)

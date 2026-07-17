"""Pure preview logic: latest-frame tracking, zoom, FPS, and histograms.

This module holds no OS window, no ``cv2.imshow`` call, and nothing that
requires a display to run -- see :mod:`glas.display` for that. Everything
here is plain data transformation over frames already sitting in memory,
which keeps it fully unit-testable in any environment, headless or not.

:class:`Preview` reads frames from a :class:`~glas.ringbuffer.RingBuffer`
using :meth:`~glas.ringbuffer.RingBuffer.peek` exclusively -- never
:meth:`~glas.ringbuffer.RingBuffer.pop` -- so a preview attached to a
buffer that a :class:`~glas.recorder.Recorder` is simultaneously writing
from can never steal a frame the dataset writer needs, slow the recording
down, or drop a frame on its behalf. If a ``Preview`` is slow, crashes, or
is never attached at all, the recording pipeline is completely unaffected.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from glas.frame import Frame
from glas.ringbuffer import RingBuffer

DEFAULT_FPS_WINDOW = 30


class ZoomRegion(BaseModel):
    """A rectangular crop region, in source-image pixel coordinates.

    Attributes
    ----------
    x, y : int
        Top-left corner of the region, in pixels.
    width, height : int
        Size of the region, in pixels. Both must be positive.
    """

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def clamped_to(self, image_width: int, image_height: int) -> ZoomRegion:
        """Return a copy of this region clipped to fit within an image.

        Useful when a zoom region was computed against one frame size but
        is about to be applied to a differently-sized one (e.g. after the
        camera's ROI changes mid-session).

        Parameters
        ----------
        image_width, image_height : int
            Size of the image this region will be applied to, in pixels.

        Returns
        -------
        ZoomRegion
            A region guaranteed to fit within ``(image_width, image_height)``.
            If the original region's top-left corner is already outside
            the image, the result has a minimal 1x1 size at the clamped
            corner rather than a negative or zero size.
        """
        x = min(self.x, max(image_width - 1, 0))
        y = min(self.y, max(image_height - 1, 0))
        width = min(self.width, image_width - x)
        height = min(self.height, image_height - y)
        return ZoomRegion(x=x, y=y, width=max(width, 1), height=max(height, 1))


def apply_zoom(image: NDArray[np.integer], region: ZoomRegion | None) -> NDArray[np.integer]:
    """Crop an image to a zoom region.

    Parameters
    ----------
    image : numpy.ndarray
        Source image, shape ``(height, width)`` or ``(height, width, channels)``.
    region : ZoomRegion, optional
        Region to crop to, clamped to ``image``'s actual size first. ``None``
        returns ``image`` unchanged.

    Returns
    -------
    numpy.ndarray
        A view into ``image`` (not a copy) covering ``region``, or ``image``
        itself if ``region`` is ``None``.
    """
    if region is None:
        return image

    height, width = image.shape[0], image.shape[1]
    clamped = region.clamped_to(width, height)
    return image[
        clamped.y : clamped.y + clamped.height,
        clamped.x : clamped.x + clamped.width,
    ]


class Preview:
    """Tracks the latest frame from a ring buffer for live viewing.

    Parameters
    ----------
    buffer : RingBuffer
        Buffer to read from, via :meth:`~glas.ringbuffer.RingBuffer.peek`
        only.
    fps_window : int, default 30
        Number of most recent distinct frames to base the :meth:`fps`
        estimate on.

    Attributes
    ----------
    zoom : ZoomRegion or None
        Region to crop rendered frames to. ``None`` shows the full frame.
    crosshair : bool
        Whether a crosshair should be drawn (rendering is
        :mod:`glas.display`'s job; this is just the flag it reads).
    crosshair_position : tuple of (int, int), optional
        Crosshair position, in source-image pixel coordinates.
    show_roi : bool
        Whether an ROI box should be drawn.
    overlay_grid : bool
        Whether an evenly-spaced reference grid should be drawn (see
        :func:`glas.display.render_frame`).

    Notes
    -----
    Not thread-safe against concurrent :meth:`update` calls from multiple
    threads -- intended for a single consumer (one preview/display loop)
    per instance, matching :meth:`~glas.ringbuffer.RingBuffer.peek`'s own
    single-reader-of-state expectations for FPS tracking (a second reader
    calling :meth:`update` concurrently would see its own new-frame
    detection interleaved with the first, corrupting the FPS estimate).
    """

    def __init__(self, buffer: RingBuffer, fps_window: int = DEFAULT_FPS_WINDOW) -> None:
        self._buffer = buffer
        self._last_frame_id: int | None = None
        self._recent_timestamps_ns: deque[int] = deque(maxlen=fps_window)

        self.zoom: ZoomRegion | None = None
        self.crosshair: bool = False
        self.crosshair_position: tuple[int, int] | None = None
        self.show_roi: bool = False
        self.overlay_grid: bool = False

    def update(self) -> Frame | None:
        """Fetch the current newest frame from the buffer.

        Returns
        -------
        Frame or None
            The newest buffered frame (possibly the same one returned by
            the previous call, if nothing new has arrived), or ``None``
            if the buffer is currently empty.
        """
        frame = self._buffer.peek()
        if frame is not None and frame.frame_id != self._last_frame_id:
            self._last_frame_id = frame.frame_id
            self._recent_timestamps_ns.append(frame.host_timestamp_ns)
        return frame

    def fps(self) -> float:
        """Estimate the current frame rate from recently seen distinct frames.

        Returns
        -------
        float
            Frames per second, averaged over the most recent
            ``fps_window`` distinct frames seen by :meth:`update`. ``0.0``
            if fewer than two distinct frames have been seen yet.
        """
        if len(self._recent_timestamps_ns) < 2:
            return 0.0
        span_ns = self._recent_timestamps_ns[-1] - self._recent_timestamps_ns[0]
        if span_ns <= 0:
            return 0.0
        intervals = len(self._recent_timestamps_ns) - 1
        return intervals * 1e9 / span_ns

    def zoom_to(
        self,
        factor: float,
        center: tuple[int, int],
        source_width: int,
        source_height: int,
    ) -> None:
        """Set :attr:`zoom` to a region centered on a point at a given magnification.

        Parameters
        ----------
        factor : float
            Zoom factor; ``2.0`` shows a region half the width and height
            of the source image. Must be greater than 0.
        center : tuple of (int, int)
            ``(x, y)`` point, in source-image pixel coordinates, the zoom
            region is centered on.
        source_width, source_height : int
            Size of the full (unzoomed) source image, in pixels.

        Raises
        ------
        ValueError
            If ``factor`` is not positive.
        """
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}.")

        width = max(int(source_width / factor), 1)
        height = max(int(source_height / factor), 1)
        center_x, center_y = center
        x = center_x - width // 2
        y = center_y - height // 2
        region = ZoomRegion(x=max(x, 0), y=max(y, 0), width=width, height=height)
        self.zoom = region.clamped_to(source_width, source_height)

    def reset_zoom(self) -> None:
        """Clear :attr:`zoom`, restoring the full, unzoomed frame."""
        self.zoom = None

    @staticmethod
    def histogram(frame: Frame, bins: int = 256) -> NDArray[np.int64]:
        """Compute a pixel-intensity histogram for a frame.

        Parameters
        ----------
        frame : Frame
            Frame to histogram.
        bins : int, default 256
            Number of histogram bins.

        Returns
        -------
        numpy.ndarray
            Integer bin counts, shape ``(bins,)``, covering the full
            representable range of ``frame.image``'s dtype (e.g.
            ``[0, 255]`` for 8-bit, ``[0, 65535]`` for 16-bit).
        """
        max_value = np.iinfo(frame.image.dtype).max
        counts, _ = np.histogram(frame.image, bins=bins, range=(0, max_value))
        return counts.astype(np.int64)

"""OpenCV-backed rendering and windowing for :class:`~glas.preview.Preview`.

Split deliberately from :mod:`glas.preview`: everything in this module up
to (but not including) the actual OS window calls is a pure function over
numpy arrays and is fully unit-testable without a display. Only
:class:`PreviewWindow` touches ``cv2.imshow``/``cv2.waitKey``, and even
those are guarded by :func:`_display_available` first.

Why the guard matters
----------------------
``cv2.imshow()`` behaves very differently depending on which OpenCV
distribution is installed when no display is available (no ``$DISPLAY``
or ``$WAYLAND_DISPLAY`` on Linux):

- ``opencv-python-headless`` has no GUI backend compiled in at all, so it
  raises a ``cv2.error`` immediately.
- ``opencv-python`` (the regular, full package -- what GLAS depends on,
  since a real preview window is the point of this module) instead
  *hangs indefinitely* with no exception, because it has a GUI backend
  compiled in that then fails to find a display to attach to.

Relying on a ``try/except`` around ``cv2.imshow()`` is therefore not
sufficient to fail safely -- it would only ever catch the headless
package's behavior, not the production one. :func:`_display_available`
checks the environment *before* any such call is made, so GLAS raises a
clean :class:`~glas.exceptions.DisplayError` instead of hanging, on both
distributions and in test suites and CI where no display exists.
"""

from __future__ import annotations

import os
import sys
from types import TracebackType
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from glas.camera_validator import ROI
from glas.exceptions import DisplayError
from glas.frame import Frame
from glas.preview import Preview, ZoomRegion, apply_zoom

DEFAULT_WINDOW_NAME = "GLAS Preview"

_CROSSHAIR_COLOR_BGR = (0, 255, 0)
_ROI_COLOR_BGR = (0, 0, 255)
_FPS_TEXT_COLOR_BGR = (0, 255, 255)
_GRID_COLOR_BGR = (80, 80, 80)
_GRID_SPACING_PX = 50
_TIMESTAMP_TEXT_COLOR_BGR = (255, 255, 255)
_HISTOGRAM_BAR_COLOR = 255
_HISTOGRAM_BACKGROUND = 0


def _display_available() -> bool:
    """Check whether a GUI display is available on this system.

    Returns
    -------
    bool
        ``True`` on non-Linux platforms (assumed to always have a
        windowing system available through the OS). On Linux, ``True``
        only if the ``DISPLAY`` or ``WAYLAND_DISPLAY`` environment
        variable is set.
    """
    if sys.platform != "linux":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _to_bgr_uint8(image: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Convert an arbitrary-dtype mono or color image to 8-bit BGR.

    Parameters
    ----------
    image : numpy.ndarray
        Source image, shape ``(height, width)`` (mono) or
        ``(height, width, 3)`` (already BGR/color).

    Returns
    -------
    numpy.ndarray
        ``uint8`` image, shape ``(height, width, 3)``, suitable for
        ``cv2.imshow`` and drawing functions.
    """
    if image.dtype != np.uint8:
        max_value = np.iinfo(image.dtype).max
        image = (image.astype(np.float64) * (255.0 / max_value)).astype(np.uint8)
    scaled = cast("NDArray[np.uint8]", image.astype(np.uint8, copy=False))

    if scaled.ndim == 2:
        return cast("NDArray[np.uint8]", cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR))
    return scaled


def render_frame(
    frame: Frame,
    *,
    zoom: ZoomRegion | None = None,
    crosshair: bool = False,
    crosshair_position: tuple[int, int] | None = None,
    roi: ROI | None = None,
    fps: float | None = None,
    overlay_grid: bool = False,
    timestamp_text: str | None = None,
) -> NDArray[np.uint8]:
    """Render a frame to a displayable BGR image, with optional overlays.

    Pure function: performs no I/O and opens no window, so it is fully
    unit-testable without a display.

    Parameters
    ----------
    frame : Frame
        Frame to render.
    zoom : ZoomRegion, optional
        If given, crop to this region (in the frame's original pixel
        coordinates) before drawing overlays. See :func:`glas.preview.apply_zoom`.
    crosshair : bool, default False
        Whether to draw a crosshair.
    crosshair_position : tuple of (int, int), optional
        Crosshair position, in the frame's original (pre-zoom) pixel
        coordinates. Ignored if ``crosshair`` is ``False`` or this is
        ``None``.
    roi : ROI, optional
        If given, draw a rectangle around this region, in the frame's
        original (pre-zoom) pixel coordinates.
    fps : float, optional
        If given, draw this value as on-screen FPS text.
    overlay_grid : bool, default False
        Whether to draw an evenly-spaced reference grid over the image.
    timestamp_text : str, optional
        If given, draw this exact string as on-screen text (bottom-left).
        Callers decide the formatting (e.g. a wall-clock time derived from
        :class:`~glas.timestamps.WallClockReference`) -- this function only
        draws whatever it is handed.

    Returns
    -------
    numpy.ndarray
        ``uint8`` BGR image ready to pass to ``cv2.imshow``.
    """
    cropped = apply_zoom(frame.image, zoom)
    image: NDArray[np.uint8] = _to_bgr_uint8(cropped).copy()

    clamped_zoom = zoom.clamped_to(frame.width, frame.height) if zoom is not None else None
    offset_x = clamped_zoom.x if clamped_zoom is not None else 0
    offset_y = clamped_zoom.y if clamped_zoom is not None else 0
    height, width = image.shape[0], image.shape[1]

    if overlay_grid:
        for x in range(0, width, _GRID_SPACING_PX):
            cv2.line(image, (x, 0), (x, height - 1), _GRID_COLOR_BGR, 1)
        for y in range(0, height, _GRID_SPACING_PX):
            cv2.line(image, (0, y), (width - 1, y), _GRID_COLOR_BGR, 1)

    if crosshair and crosshair_position is not None:
        x = crosshair_position[0] - offset_x
        y = crosshair_position[1] - offset_y
        if 0 <= x < width:
            cv2.line(image, (x, 0), (x, height - 1), _CROSSHAIR_COLOR_BGR, 1)
        if 0 <= y < height:
            cv2.line(image, (0, y), (width - 1, y), _CROSSHAIR_COLOR_BGR, 1)

    if roi is not None:
        top_left = (roi.offset_x - offset_x, roi.offset_y - offset_y)
        bottom_right = (top_left[0] + roi.width, top_left[1] + roi.height)
        cv2.rectangle(image, top_left, bottom_right, _ROI_COLOR_BGR, 1)

    if fps is not None:
        cv2.putText(
            image,
            f"{fps:.1f} fps",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _FPS_TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    if timestamp_text is not None:
        cv2.putText(
            image,
            timestamp_text,
            (5, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TIMESTAMP_TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    return image


def render_histogram(
    counts: NDArray[np.int64], width: int = 256, height: int = 100
) -> NDArray[np.uint8]:
    """Render histogram bin counts as a bar-chart image.

    Pure function: performs no I/O and opens no window.

    Parameters
    ----------
    counts : numpy.ndarray
        Bin counts, e.g. from :meth:`glas.preview.Preview.histogram`.
    width, height : int, default 256, 100
        Size of the rendered image, in pixels. If ``counts`` does not
        already have ``width`` entries, it is resampled (nearest-bin) to
        fit.

    Returns
    -------
    numpy.ndarray
        ``uint8`` grayscale image, shape ``(height, width)``: a black
        background with white vertical bars, one per (resampled) bin,
        scaled so the tallest bar reaches the top.
    """
    image = np.full((height, width), _HISTOGRAM_BACKGROUND, dtype=np.uint8)

    if counts.size == 0:
        return image

    if counts.size != width:
        indices = (np.arange(width) * counts.size / width).astype(np.int64)
        indices = np.clip(indices, 0, counts.size - 1)
        counts = counts[indices]

    peak = int(counts.max())
    if peak <= 0:
        return image

    bar_heights = (counts.astype(np.float64) / peak * height).astype(np.int64)
    for x, bar_height in enumerate(bar_heights):
        if bar_height > 0:
            image[height - bar_height : height, x] = _HISTOGRAM_BAR_COLOR

    return image


class PreviewWindow:
    """An OpenCV window showing a live-updating :class:`~glas.preview.Preview`.

    Parameters
    ----------
    preview : Preview
        Preview to render frames from.
    window_name : str, default "GLAS Preview"
        Title of the OS window.

    Notes
    -----
    Every method that would touch the display checks
    :func:`_display_available` first and raises
    :class:`~glas.exceptions.DisplayError` immediately rather than calling
    into ``cv2.imshow``/``cv2.waitKey`` without a display -- see the
    module docstring for why that check exists.
    """

    def __init__(self, preview: Preview, window_name: str = DEFAULT_WINDOW_NAME) -> None:
        self._preview = preview
        self._window_name = window_name
        self._open = False

    def show_once(self, roi: ROI | None = None) -> bool:
        """Render and display the current frame, once.

        Parameters
        ----------
        roi : ROI, optional
            If given, drawn as a rectangle overlay.

        Returns
        -------
        bool
            ``True`` if a frame was available and displayed, ``False``
            if the buffer was currently empty (nothing was shown).

        Raises
        ------
        DisplayError
            If no display is available.
        """
        if not _display_available():
            raise DisplayError(
                "Cannot show a preview window: no display available "
                "(DISPLAY/WAYLAND_DISPLAY is not set)."
            )

        frame = self._preview.update()
        if frame is None:
            return False

        image = render_frame(
            frame,
            zoom=self._preview.zoom,
            crosshair=self._preview.crosshair,
            crosshair_position=self._preview.crosshair_position,
            roi=roi if self._preview.show_roi else None,
            fps=self._preview.fps(),
            overlay_grid=self._preview.overlay_grid,
        )
        cv2.imshow(self._window_name, image)
        self._open = True
        return True

    def wait_key(self, delay_ms: int = 1) -> int:
        """Poll for a key press, pumping the window's event loop.

        Parameters
        ----------
        delay_ms : int, default 1
            Milliseconds to wait for a key press.

        Returns
        -------
        int
            The key code pressed, or ``-1`` if none was pressed within
            ``delay_ms``.

        Raises
        ------
        DisplayError
            If no display is available.
        """
        if not _display_available():
            raise DisplayError(
                "Cannot poll for input: no display available (DISPLAY/WAYLAND_DISPLAY is not set)."
            )
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        """Close the window, if currently open. Safe to call more than once."""
        if self._open:
            cv2.destroyWindow(self._window_name)
            self._open = False

    def run(self, roi: ROI | None = None, quit_key: str = "q") -> None:
        """Run a blocking display loop until ``quit_key`` is pressed.

        Parameters
        ----------
        roi : ROI, optional
            If given, drawn as a rectangle overlay on every frame.
        quit_key : str, default "q"
            Single character that stops the loop when pressed.

        Raises
        ------
        DisplayError
            If no display is available.
        """
        quit_code = ord(quit_key)
        try:
            while True:
                self.show_once(roi=roi)
                if self.wait_key(1) == quit_code:
                    break
        finally:
            self.close()

    def __enter__(self) -> PreviewWindow:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

"""Exports a recorded dataset to common image and video file formats.

Reads frames back from a finalized :class:`~glas.dataset.Dataset` via
:func:`glas.dataset.iter_frames` and writes them out as:

- a **TIFF or PNG image sequence** (one numbered file per frame), preserving
  each frame's native pixel data and bit depth exactly -- no color
  conversion, since the point of an image sequence export is downstream
  analysis, not viewing.
- an **MP4 or AVI video**, via ``cv2.VideoWriter``, reusing
  :func:`glas.display._to_bgr_uint8` to convert mono frames to the 8-bit
  BGR every common video codec expects (the same conversion
  :mod:`glas.display` already uses for on-screen preview rendering).
- an **animated GIF**, via Pillow -- OpenCV has no GIF encoder, so this is
  the one export path that depends on a different library. Pillow's GIF
  writer has no streaming API, so a GIF export buffers every frame in
  memory before writing; this is fine for a typical clip but means a very
  long recording exported as a GIF can use significant RAM, unlike the
  MP4/AVI/image-sequence paths, which stream frame by frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2
from PIL import Image
from pydantic import BaseModel, ConfigDict

from glas.dataset import iter_frames
from glas.display import _to_bgr_uint8
from glas.exceptions import ExportError
from glas.frame import Frame
from glas.logger import get_logger

logger = get_logger(__name__)

ExportFormat = Literal["tiff", "png", "mp4", "avi", "gif"]

_IMAGE_SEQUENCE_EXTENSIONS: dict[str, str] = {"tiff": "tif", "png": "png"}
_VIDEO_FOURCC: dict[str, str] = {"mp4": "mp4v", "avi": "XVID"}

DEFAULT_FPS = 30.0


class ExportResult(BaseModel):
    """Outcome of a successful :func:`export_dataset` call.

    Attributes
    ----------
    format : {"tiff", "png", "mp4", "avi", "gif"}
        Format that was exported.
    frame_count : int
        Number of frames written.
    output_path : pathlib.Path
        Where the export was written -- the image sequence directory, or
        the video/GIF file.
    """

    model_config = ConfigDict(frozen=True)

    format: ExportFormat
    frame_count: int
    output_path: Path


def export_dataset(
    folder: Path,
    output: Path,
    format: ExportFormat,
    *,
    fps: float = DEFAULT_FPS,
    start_frame: int | None = None,
    end_frame: int | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Export a recorded dataset to an image sequence, video, or GIF.

    Parameters
    ----------
    folder : pathlib.Path
        Dataset folder to read frames from (see
        :func:`glas.dataset.iter_frames`).
    output : pathlib.Path
        Destination. For ``"tiff"``/``"png"``, a directory that is
        created if missing, holding one numbered file per frame. For
        ``"mp4"``/``"avi"``/``"gif"``, the destination file path.
    format : {"tiff", "png", "mp4", "avi", "gif"}
        Output format.
    fps : float, default 30.0
        Playback frame rate for video/GIF formats. Ignored for image
        sequences. Must be positive.
    start_frame, end_frame : int, optional
        If given, export only frames whose position in dataset order
        (0-based, *not* necessarily ``Frame.frame_id``) falls in
        ``[start_frame, end_frame)``. ``None`` exports from the
        beginning / through the end.
    overwrite : bool, default False
        For image sequences, whether to proceed if ``output`` already
        exists and is non-empty. For video/GIF, whether to overwrite an
        existing ``output`` file. ``False`` raises instead.

    Returns
    -------
    ExportResult

    Raises
    ------
    ExportError
        If ``format`` is unrecognized, ``fps`` is not positive, the
        destination already exists and ``overwrite`` is ``False``, the
        selected frame range is empty, or (for video formats) the
        installed OpenCV build cannot open the requested codec.
    DatasetError, DatasetFormatError, DatasetIOError
        Propagated from :func:`glas.dataset.iter_frames` if the source
        dataset cannot be read.
    """
    frames = _select_frames(iter_frames(folder), start_frame, end_frame)

    if format in _IMAGE_SEQUENCE_EXTENSIONS:
        count = _export_image_sequence(frames, output, format, overwrite=overwrite)
    elif format == "gif":
        count = _export_gif(frames, output, fps=fps, overwrite=overwrite)
    elif format in _VIDEO_FOURCC:
        count = _export_video(frames, output, format, fps=fps, overwrite=overwrite)
    else:
        expected = sorted({*_IMAGE_SEQUENCE_EXTENSIONS, "gif", *_VIDEO_FOURCC})
        raise ExportError(f"Unknown export format {format!r}; expected one of {expected}.")

    logger.info("Exported %d frame(s) from %s to %s (%s).", count, folder, output, format)
    return ExportResult(format=format, frame_count=count, output_path=output)


def _select_frames(
    frames: Iterator[Frame], start_frame: int | None, end_frame: int | None
) -> Iterator[Frame]:
    for index, frame in enumerate(frames):
        if start_frame is not None and index < start_frame:
            continue
        if end_frame is not None and index >= end_frame:
            return
        yield frame


def _export_image_sequence(
    frames: Iterator[Frame], output: Path, format: ExportFormat, *, overwrite: bool
) -> int:
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise ExportError(
            f"{output} already exists and is not empty; pass overwrite=True to replace "
            "its contents."
        )
    output.mkdir(parents=True, exist_ok=True)

    extension = _IMAGE_SEQUENCE_EXTENSIONS[format]
    count = 0
    for index, frame in enumerate(frames):
        path = output / f"frame_{index:06d}.{extension}"
        if not cv2.imwrite(str(path), frame.image):
            raise ExportError(f"Failed to write {path}.")
        count += 1

    if count == 0:
        raise ExportError("No frames to export: the requested range is empty.")
    return count


def _export_video(
    frames: Iterator[Frame], output: Path, format: ExportFormat, *, fps: float, overwrite: bool
) -> int:
    _check_video_destination(output, fps=fps, overwrite=overwrite)

    fourcc = cv2.VideoWriter.fourcc(*_VIDEO_FOURCC[format])
    writer: cv2.VideoWriter | None = None
    count = 0
    try:
        for frame in frames:
            bgr = _to_bgr_uint8(frame.image)
            if writer is None:
                height, width = bgr.shape[0], bgr.shape[1]
                writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise ExportError(
                        f"Could not open a video writer for {output} (format={format!r}); "
                        "the installed OpenCV build may be missing this codec."
                    )
            writer.write(bgr)
            count += 1
    finally:
        if writer is not None:
            writer.release()

    if count == 0:
        raise ExportError("No frames to export: the requested range is empty.")
    return count


def _export_gif(frames: Iterator[Frame], output: Path, *, fps: float, overwrite: bool) -> int:
    _check_video_destination(output, fps=fps, overwrite=overwrite)

    images: list[Image.Image] = []
    for frame in frames:
        bgr = _to_bgr_uint8(frame.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))

    if not images:
        raise ExportError("No frames to export: the requested range is empty.")

    duration_ms = round(1000.0 / fps)
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return len(images)


def _check_video_destination(output: Path, *, fps: float, overwrite: bool) -> None:
    if fps <= 0:
        raise ExportError(f"fps must be positive, got {fps}.")
    if output.is_file() and not overwrite:
        raise ExportError(f"{output} already exists; pass overwrite=True to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

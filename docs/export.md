# Export Engine

Phase 8 turns a recorded dataset into common image and video file formats
for sharing, quick review, or feeding into external analysis tools.

```
Dataset (frames.h5 or frames.bin) -> iter_frames() -> export_dataset()
                                                          |
                                          TIFF/PNG sequence, MP4/AVI, or GIF
```

`glas.dataset.iter_frames()` reads a finalized dataset's frames back, in
recorded order, streaming one frame at a time regardless of dataset size
(the same never-buffer-more-than-one-frame discipline
`Dataset.append_frame()` already has on the write side). `glas.export`
consumes that stream and writes it out in the requested format.

## Quickstart

```python
from glas.export import export_dataset

# One TIFF file per frame, preserving native pixel data exactly.
export_dataset(dataset.folder, output=Path("frames_tiff"), format="tiff")

# An MP4 at 30 fps.
export_dataset(dataset.folder, output=Path("recording.mp4"), format="mp4", fps=30.0)

# Just frames 100-199, as an animated GIF.
export_dataset(
    dataset.folder,
    output=Path("clip.gif"),
    format="gif",
    fps=15.0,
    start_frame=100,
    end_frame=200,
)
```

`export_dataset()` returns an `ExportResult` (`format`, `frame_count`,
`output_path`). Pass `overwrite=True` to replace an existing destination
(a non-empty output directory for image sequences, or an existing file
for video/GIF) -- the default is to raise `ExportError` rather than
silently overwrite or silently merge into existing output.

## Format notes

- **TIFF / PNG** (`format="tiff"` / `"png"`): one numbered file per frame
  (`frame_000000.tif`, `frame_000001.tif`, ...) written via
  `cv2.imwrite`, with the frame's native pixel data and bit depth
  preserved exactly -- no color conversion. This is deliberately
  different from `glas.display`'s rendering path, which converts
  everything to 8-bit BGR for on-screen viewing: an image sequence
  export is for downstream analysis, where losing bit depth to make an
  image "look right" on screen would throw away real data.
- **MP4 / AVI** (`format="mp4"` / `"avi"`): written via `cv2.VideoWriter`,
  reusing `glas.display`'s `_to_bgr_uint8()` conversion (the same one
  used for live preview rendering) since video codecs expect 8-bit BGR
  frames. Streams frame by frame; memory use doesn't grow with dataset
  size.
- **GIF** (`format="gif"`): written via [Pillow](https://python-pillow.org/),
  since OpenCV has no GIF encoder. Pillow's GIF writer has no streaming
  API, so a GIF export buffers every frame in memory before writing --
  fine for a typical short clip, but a very long recording exported as a
  GIF can use significant RAM, unlike the other three formats.

## Frame range selection

`start_frame`/`end_frame` select a `[start_frame, end_frame)` range by
**position in dataset order** (0-based), not by `Frame.frame_id` --
usually the same thing, but if a recording was ever paused and resumed
with frames dropped in between, position and `frame_id` can diverge, and
range selection follows position for predictable "give me frames 100
through 200" behavior. An empty resulting range (e.g. `start_frame`
beyond the dataset's frame count) raises `ExportError` rather than
silently producing an empty file.

## Reading raw binary datasets back

HDF5 records each frame's dtype as part of the file format; raw binary
storage only ever wrote pixel bytes (see `docs/dataset.md`), so reading
raw binary frames back requires knowing each pixel's byte width, which
`glas.frame.pixel_format_dtype()` resolves from the pixel format name
recorded in `metadata.json`. Only mono pixel formats are supported
(`Mono8`, `Mono10`, `Mono12`, `Mono16`), matching the Basler ace
acA640-750um this project targets -- a mono camera, so color/Bayer
formats are out of scope for raw binary reading.

## Testing

Every export path is tested end to end against real datasets (both HDF5
and raw binary sources) in a `tmp_path`, then read back and verified: TIFF/
PNG output via `cv2.imread(..., cv2.IMREAD_UNCHANGED)`, MP4/AVI via
`cv2.VideoCapture` (actually reading every frame back, not just trusting a
reported frame count), and GIF via `PIL.Image.open()` (`n_frames`,
per-frame `duration`). Overwrite guards, frame-range selection, and error
paths (unknown format, non-positive fps, empty range) are all exercised
directly, no mocking.

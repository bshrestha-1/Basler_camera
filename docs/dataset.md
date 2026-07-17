# Dataset Writer

Phase 4 saves experiments to disk safely:

```
Camera -> Acquisition (producer thread) -> RingBuffer -> DatasetWriter (writer thread) -> Dataset
```

The writer thread never blocks the producer thread, and never silently
drops a frame it has already accepted -- everything currently buffered
gets written before a stop completes.

## Module layout

- `glas.metadata.DatasetMetadata` -- the canonical description of a recording.
- `glas.timestamps.TimestampLog` -- per-frame timestamp bookkeeping and gap detection.
- `glas.dataset.Dataset` -- on-disk frame storage (HDF5 or raw binary), plus
  `create_experiment_folder()` and `validate_dataset()`.
- `glas.writer.DatasetWriter` -- the background thread tying a `RingBuffer` to a `Dataset`.

## Quickstart

```python
import time
from glas.camera import Camera
from glas.acquisition import Acquisition
from glas.dataset import Dataset, create_experiment_folder, resolve_dataset_format
from glas.metadata import DatasetMetadata
from glas.writer import DatasetWriter
from pathlib import Path

with Camera() as camera:
    info = camera.get_info()
    folder = create_experiment_folder(Path("~/glas_data").expanduser())

    # DatasetMetadata.dataset_format only accepts a concrete "hdf5" or
    # "raw_binary" (it validates on construction); resolve "auto" first.
    dataset_format = resolve_dataset_format("auto")
    metadata = DatasetMetadata(
        dataset_format=dataset_format,
        camera_model=info.model_name,
        camera_serial=info.serial_number,
        pixel_format=camera.pixel_format,
        width=camera.roi.width,
        height=camera.roi.height,
        created_at_utc="2026-07-13T12:00:00+00:00",
        exposure_time_us=camera.exposure_time_us,
        gain_db=camera.gain_db,
        frame_rate_hz=camera.frame_rate_hz,
        roi_offset_x=camera.roi.offset_x,
        roi_offset_y=camera.roi.offset_y,
        notes="shaker at 60 Hz, 4g",
        extra={"operator": "bijay", "experiment": "brazil-nut-01"},
    )

    dataset = Dataset.create(folder, metadata, dataset_format=dataset_format)
    acquisition = Acquisition(camera, buffer_capacity=256)
    writer = DatasetWriter(acquisition.buffer, dataset)

    writer.start()
    acquisition.start()
    time.sleep(10.0)
    acquisition.stop()
    writer.stop()  # drains the buffer, then finalizes the dataset

    print(acquisition.stats())
    print(writer.stats())
```

`Acquisition` and `DatasetWriter` are independent objects connected only
by the ring buffer -- `Acquisition` doesn't know a writer exists, and
`DatasetWriter` doesn't know frames come from a camera. Stop the producer
before the writer so the writer's final drain has nothing new arriving
while it empties the buffer.

## On-disk layout

Every dataset is one folder containing:

```
Run0001/
  metadata.json      # DatasetMetadata, human-readable
  checksums.json      # SHA-256 of every data file below
  frames.h5            # HDF5 format (dataset_format="hdf5")
  # -- or --
  frames.bin            # raw binary format (dataset_format="raw_binary")
  frames_index.csv
```

### HDF5 format (`frames.h5`)

- `frames`: resizable dataset, shape `(N, height, width[, channels])`,
  gzip-compressed, chunked one frame at a time.
- `frame_ids`, `host_timestamps_ns`, `device_timestamps_ticks`: parallel
  `int64` datasets, one entry per frame, in the same order as `frames`.
- Attribute `metadata_json`: the same JSON as `metadata.json`, embedded
  directly in the file so it's self-describing even in isolation.

### Raw binary format (`frames.bin` + `frames_index.csv`)

- `frames.bin`: every frame's raw bytes (`image.tobytes()`), concatenated
  in write order. Frame size and dtype aren't stored in the file itself --
  they're derived from `metadata.json`'s `width`/`height`/`pixel_format`.
- `frames_index.csv`: one row per frame -- `frame_id, host_timestamp_ns,
  device_timestamp_ticks` -- in the same order as `frames.bin`.

### Choosing a format

`Dataset.create(..., dataset_format=...)`:

- `"auto"` (default) -- HDF5 if `h5py` is installed, otherwise raw binary.
  Use this unless you have a specific reason not to.
- `"hdf5"` -- explicit. Raises `DatasetFormatError` if `h5py` isn't
  installed, rather than silently substituting a different format.
- `"raw_binary"` -- explicit, regardless of whether `h5py` is available.
  Useful for environments where the compiled HDF5 library can't be
  installed, or for maximum-simplicity post-processing with tools that
  don't speak HDF5.

## Metadata

`DatasetMetadata` is a strict Pydantic model (`extra="forbid"`): every field
must be present, and no unrecognized top-level fields are allowed -- put
anything not covered by a named field into `extra` instead. Loading from disk
(`load_metadata_json()` / `DatasetMetadata.from_dict()`) translates Pydantic's
`ValidationError` into `glas.exceptions.JSONValidationError` (via
`JSONValidationError.from_pydantic()`), listing every violation rather than
just the first. Note that this means direct, in-code construction
(`DatasetMetadata(...)`) is also validated immediately -- Pydantic models
validate on construction, not only when loaded from a file -- so `Dataset.create()`
always builds `DatasetMetadata` with a concrete `dataset_format` ("hdf5" or
"raw_binary"), never the placeholder `"auto"` accepted by `Dataset.create()`'s
own `dataset_format` parameter. Use `glas.dataset.resolve_dataset_format()` if
you need to resolve `"auto"` to a concrete value before constructing metadata
yourself (see `glas.controller.RecorderController.start_recording()` for an
example).

### Reproducibility

Every recording's metadata is a complete-enough snapshot to reproduce how it
was captured: camera model/serial, pixel format, ROI (both size --
`width`/`height` -- and offset -- `roi_offset_x`/`roi_offset_y`), exposure
time, gain, frame rate cap (`frame_rate_hz`), the timestamp it was created
(`created_at_utc`), the GLAS version that created it (`glas_version`), and
every other camera setting that affects capture (gamma, binning, horizontal/
vertical flip, auto-exposure/auto-gain mode, whether the frame rate cap was
enabled, hardware trigger state) in the open-ended `camera_settings` dict.
`glas.controller.RecorderController.start_recording()` populates all of this
automatically from the connected `Camera` -- readers of a dataset's
`metadata.json` never need to guess how it was configured.

## Checksums and validation

`Dataset.finalize()` computes a SHA-256 checksum of every data file it wrote
and records them in `checksums.json`. `validate_dataset(folder)` re-reads a
dataset from disk and checks, collecting every problem found rather than
stopping at the first:

1. `metadata.json` exists and parses against the schema.
2. Every file listed in `checksums.json` exists and its checksum still matches.
3. The data file's structure matches `metadata.frame_count` -- HDF5 dataset
   lengths and frame shape, or raw binary file size and index row count.

```python
from glas.dataset import validate_dataset

result = validate_dataset(folder)
if not result.valid:
    for error in result.errors:
        print(error)
```

An empty recording (`frame_count == 0`, e.g. started and immediately
stopped) writes only `metadata.json` -- no data file, no checksums entry --
and validates cleanly.

## Reading frames back

`glas.dataset.iter_frames(folder)` reads a finalized dataset's frames back,
in recorded order, as `Frame` objects -- supporting both storage backends,
and streaming one frame at a time rather than loading the whole dataset
into memory:

```python
from glas.dataset import iter_frames

for frame in iter_frames(folder):
    print(frame.frame_id, frame.image.shape)
```

This is what `glas.export.export_dataset()` (Phase 8, see `docs/export.md`)
builds on to turn a dataset into a TIFF/PNG sequence or an MP4/AVI/GIF.
Reading a raw binary dataset back requires knowing each pixel's byte
width, which isn't recorded anywhere in `frames.bin`/`frames_index.csv`
itself -- `glas.frame.pixel_format_dtype()` resolves it from
`metadata.json`'s `pixel_format`, supporting the mono formats the target
camera produces (`Mono8`, `Mono10`, `Mono12`, `Mono16`).

## Frame loss tracking

`WriterStats.dropped_frame_count` reports frames that were successfully
grabbed by `Acquisition` but never reached the writer -- typically because
the ring buffer overflowed under a slow disk. It's derived from gaps in the
`frame_id` sequence the writer actually observes
(`glas.timestamps.TimestampLog`), which in steady state matches
`RingBufferStats.dropped` from the same run. Combined with
`AcquisitionStats.grab_errors` (frames the camera driver failed to deliver
at all), this gives a complete picture of where, if anywhere, frames were
lost between the camera and disk.

## Testing without physical hardware

`glas.metadata`, `glas.timestamps`, and `glas.dataset` need neither pypylon
nor a camera -- their tests build `Frame` objects directly with `numpy` and
exercise real HDF5/raw-binary I/O against `tmp_path`. `glas.writer`'s tests
combine a real `RingBuffer` with a real background thread and real disk I/O,
including a genuine concurrency test that floods the buffer faster than the
writer can drain it to verify drop accounting under real timing pressure.

"""On-disk dataset storage: HDF5 (primary) or raw binary (fallback).

A :class:`Dataset` owns everything about one recorded experiment folder:

- the frame data itself (``frames.h5``, or ``frames.bin`` +
  ``frames_index.csv``)
- ``metadata.json``, the canonical description of the recording
  (:class:`glas.metadata.DatasetMetadata`)
- ``checksums.json``, SHA-256 checksums of the data file(s), for later
  integrity validation via :func:`validate_dataset`

Frames are appended one at a time via :meth:`Dataset.append_frame`;
nothing is buffered beyond what the backend itself streams to disk
immediately, so a ``Dataset`` never accumulates unbounded memory
regardless of how long a recording runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from glas.exceptions import DatasetError, DatasetFormatError, DatasetIOError, JSONValidationError
from glas.frame import Frame
from glas.logger import get_logger
from glas.metadata import DatasetMetadata, load_metadata_json, save_metadata_json

logger = get_logger(__name__)

try:
    import h5py
except ImportError:  # pragma: no cover - exercised in environments without h5py
    h5py = None

_METADATA_FILENAME = "metadata.json"
_CHECKSUMS_FILENAME = "checksums.json"
_HDF5_FILENAME = "frames.h5"
_RAW_FRAMES_FILENAME = "frames.bin"
_RAW_INDEX_FILENAME = "frames_index.csv"


def create_experiment_folder(base_dir: Path, prefix: str = "Run", width: int = 4) -> Path:
    """Create the next automatically numbered experiment folder.

    Scans ``base_dir`` for existing ``{prefix}{NNNN}`` folders and
    creates the next one in sequence (``Run0001``, ``Run0002``, ...).

    Parameters
    ----------
    base_dir : pathlib.Path
        Directory experiment folders live under. Created if it doesn't
        already exist.
    prefix : str, default "Run"
        Folder name prefix.
    width : int, default 4
        Zero-padded width of the run number.

    Returns
    -------
    pathlib.Path
        The newly created, empty experiment folder.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    highest = 0
    for entry in base_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith(prefix):
            continue
        suffix = entry.name[len(prefix) :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))

    folder = base_dir / f"{prefix}{highest + 1:0{width}d}"
    folder.mkdir()
    return folder


def resolve_dataset_format(requested: str) -> Literal["hdf5", "raw_binary"]:
    """Resolve a requested dataset format to a concrete ``"hdf5"``/``"raw_binary"``.

    Useful for callers (like :class:`glas.controller.RecorderController`)
    that need a concrete format to build :class:`~glas.metadata.DatasetMetadata`
    with *before* calling :meth:`Dataset.create`, which cannot hold
    ``"auto"`` since it isn't a real storage backend.

    Parameters
    ----------
    requested : {"hdf5", "raw_binary", "auto"}
        ``"auto"`` resolves to ``"hdf5"`` if ``h5py`` is installed,
        otherwise ``"raw_binary"``. This does not check ``h5py``
        availability for an explicit ``"hdf5"`` request -- that is
        enforced when the backend is actually constructed, in
        :meth:`Dataset.create`.

    Returns
    -------
    Literal["hdf5", "raw_binary"]

    Raises
    ------
    DatasetFormatError
        If ``requested`` is not one of ``"hdf5"``, ``"raw_binary"``, or
        ``"auto"``.
    """
    if requested == "auto":
        return "hdf5" if h5py is not None else "raw_binary"
    if requested == "hdf5":
        return "hdf5"
    if requested == "raw_binary":
        return "raw_binary"
    raise DatasetFormatError(
        f"Unknown dataset format {requested!r}; expected 'hdf5', 'raw_binary', or 'auto'."
    )


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Backend(Protocol):
    def append(self, frame: Frame, metadata: DatasetMetadata | None = None) -> None: ...
    def close(self, metadata: DatasetMetadata) -> None: ...
    @property
    def count(self) -> int: ...
    def data_files(self) -> list[Path]: ...


def _validate_frame_shape(frame: Frame, metadata: DatasetMetadata) -> None:
    if frame.height != metadata.height or frame.width != metadata.width:
        raise DatasetIOError(
            f"Frame {frame.frame_id} has shape {frame.height}x{frame.width}, "
            f"expected {metadata.height}x{metadata.width}."
        )


class _Hdf5Backend:
    """Appends frames to a single resizable HDF5 file."""

    def __init__(self, path: Path) -> None:
        if h5py is None:
            raise DatasetFormatError(
                "h5py is not installed. Install it (`pip install h5py`) to use the "
                "'hdf5' dataset format, or use 'raw_binary' / 'auto' instead."
            )
        self._path = path
        self._file: Any = None
        self._frames_ds: Any = None
        self._frame_ids_ds: Any = None
        self._host_ts_ds: Any = None
        self._device_ts_ds: Any = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def data_files(self) -> list[Path]:
        return [self._path]

    def _initialize(self, frame: Frame) -> None:
        shape = frame.image.shape
        self._file = h5py.File(self._path, "w")
        self._frames_ds = self._file.create_dataset(
            "frames",
            shape=(0, *shape),
            maxshape=(None, *shape),
            dtype=frame.image.dtype,
            chunks=(1, *shape),
            compression="gzip",
            compression_opts=4,
        )
        self._frame_ids_ds = self._file.create_dataset(
            "frame_ids", shape=(0,), maxshape=(None,), dtype="int64"
        )
        self._host_ts_ds = self._file.create_dataset(
            "host_timestamps_ns", shape=(0,), maxshape=(None,), dtype="int64"
        )
        self._device_ts_ds = self._file.create_dataset(
            "device_timestamps_ticks", shape=(0,), maxshape=(None,), dtype="int64"
        )

    def append(self, frame: Frame, metadata: DatasetMetadata | None = None) -> None:
        if metadata is not None:
            _validate_frame_shape(frame, metadata)
        if self._file is None:
            self._initialize(frame)

        n = self._count + 1
        for dataset, value in (
            (self._frames_ds, frame.image),
            (self._frame_ids_ds, frame.frame_id),
            (self._host_ts_ds, frame.host_timestamp_ns),
            (self._device_ts_ds, frame.device_timestamp_ticks),
        ):
            dataset.resize(n, axis=0)
            dataset[self._count] = value
        self._count = n

    def close(self, metadata: DatasetMetadata) -> None:
        if self._file is not None:
            self._file.attrs["metadata_json"] = json.dumps(
                metadata.replace(frame_count=self._count).to_dict()
            )
            self._file.close()
            self._file = None


class _RawBinaryBackend:
    """Appends raw frame bytes to a flat binary file, with a CSV index."""

    def __init__(self, folder: Path) -> None:
        self._frames_path = folder / _RAW_FRAMES_FILENAME
        self._index_path = folder / _RAW_INDEX_FILENAME
        self._frames_file: Any = None
        self._index_rows: list[tuple[int, int, int]] = []
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def data_files(self) -> list[Path]:
        return [self._frames_path, self._index_path]

    def append(self, frame: Frame, metadata: DatasetMetadata | None = None) -> None:
        if metadata is not None:
            _validate_frame_shape(frame, metadata)

        if self._frames_file is None:
            try:
                self._frames_file = self._frames_path.open("wb")
            except OSError as exc:
                raise DatasetIOError(f"Could not open {self._frames_path}: {exc}") from exc

        try:
            self._frames_file.write(frame.image.tobytes())
        except OSError as exc:
            raise DatasetIOError(f"Could not write frame {frame.frame_id}: {exc}") from exc

        self._index_rows.append(
            (frame.frame_id, frame.host_timestamp_ns, frame.device_timestamp_ticks)
        )
        self._count += 1

    def close(self, metadata: DatasetMetadata) -> None:
        if self._frames_file is not None:
            self._frames_file.flush()
            self._frames_file.close()
            self._frames_file = None

        with self._index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_id", "host_timestamp_ns", "device_timestamp_ticks"])
            writer.writerows(self._index_rows)


class Dataset:
    """A recorded (or in-progress) experiment dataset on disk.

    Create with :meth:`create`, append frames with :meth:`append_frame`,
    and call :meth:`finalize` (or use as a context manager) exactly once
    when done.

    Attributes
    ----------
    folder : pathlib.Path
        The experiment folder this dataset lives in.
    """

    def __init__(self, folder: Path, metadata: DatasetMetadata, backend: _Backend) -> None:
        self.folder = folder
        self._metadata = metadata
        self._backend = backend
        self._finalized = False

    @classmethod
    def create(
        cls,
        folder: Path,
        metadata: DatasetMetadata,
        dataset_format: str = "auto",
    ) -> Dataset:
        """Create a new dataset in ``folder``.

        Parameters
        ----------
        folder : pathlib.Path
            Destination directory. Created if it doesn't exist.
        metadata : DatasetMetadata
            Metadata describing the recording. Its ``dataset_format`` is
            overwritten to match the backend actually selected.
        dataset_format : {"hdf5", "raw_binary", "auto"}, default "auto"
            Storage backend to use. ``"auto"`` uses HDF5 if ``h5py`` is
            installed, otherwise falls back to raw binary. Requesting
            ``"hdf5"`` explicitly without ``h5py`` installed raises
            rather than silently substituting a different format.

        Returns
        -------
        Dataset

        Raises
        ------
        DatasetFormatError
            If ``dataset_format`` is unrecognized, or ``"hdf5"`` is
            requested without ``h5py`` installed.
        """
        folder.mkdir(parents=True, exist_ok=True)

        resolved_format = resolve_dataset_format(dataset_format)
        metadata = metadata.replace(dataset_format=resolved_format)

        backend: _Backend
        if resolved_format == "hdf5":
            backend = _Hdf5Backend(folder / _HDF5_FILENAME)
        else:
            backend = _RawBinaryBackend(folder)

        logger.info("Created %s dataset at %s.", resolved_format, folder)
        return cls(folder, metadata, backend)

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata this dataset was created with (``frame_count`` is stale until finalized)."""
        return self._metadata

    @property
    def frame_count(self) -> int:
        """Number of frames appended so far."""
        return self._backend.count

    def append_frame(self, frame: Frame) -> None:
        """Append one frame to the dataset.

        Parameters
        ----------
        frame : Frame
            Must have the same width/height the dataset was created with.

        Raises
        ------
        DatasetError
            If the dataset has already been finalized.
        DatasetIOError
            If the frame's shape doesn't match the dataset's declared
            width/height, or the underlying write fails.
        """
        if self._finalized:
            raise DatasetError("Cannot append to a dataset that has already been finalized.")
        self._backend.append(frame, self._metadata)

    def finalize(self) -> DatasetMetadata:
        """Close the dataset: finalize frame data, write metadata and checksums.

        Safe to call more than once; only the first call has an effect.
        If no frames were ever appended, no data file is created -- only
        ``metadata.json`` (with ``frame_count=0``).

        Returns
        -------
        DatasetMetadata
            The final metadata, with ``frame_count`` set to the number
            of frames actually written.
        """
        if self._finalized:
            return self._metadata

        self._backend.close(self._metadata)
        self._metadata = self._metadata.replace(frame_count=self._backend.count)
        save_metadata_json(self._metadata, self.folder / _METADATA_FILENAME)

        checksums: dict[str, str] = {}
        if self._backend.count > 0:
            for data_file in self._backend.data_files():
                if data_file.is_file():
                    checksums[data_file.name] = _sha256_of_file(data_file)
        (self.folder / _CHECKSUMS_FILENAME).write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self._finalized = True
        logger.info("Finalized dataset at %s (%d frames).", self.folder, self._backend.count)
        return self._metadata

    def __enter__(self) -> Dataset:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finalize()


class DatasetValidationResult(BaseModel):
    """Outcome of validating a dataset folder on disk.

    Attributes
    ----------
    valid : bool
        ``True`` if every check passed.
    errors : list of str
        Every problem found; empty if ``valid`` is ``True``.
    metadata : DatasetMetadata or None
        Parsed metadata, if ``metadata.json`` was readable.
    """

    model_config = ConfigDict(frozen=True)

    valid: bool
    errors: list[str]
    metadata: DatasetMetadata | None


def validate_dataset(folder: Path) -> DatasetValidationResult:
    """Validate a dataset folder's structural and checksum integrity.

    Checks that ``metadata.json`` parses, that the data file(s) implied
    by its ``dataset_format`` exist and structurally match
    ``frame_count`` (HDF5 dataset lengths, or raw binary file size and
    index row count), and that every file listed in ``checksums.json``
    still matches its recorded SHA-256 checksum.

    Parameters
    ----------
    folder : pathlib.Path
        Dataset folder to validate.

    Returns
    -------
    DatasetValidationResult
        Collects every problem found rather than stopping at the first.
    """
    errors: list[str] = []

    metadata_path = folder / _METADATA_FILENAME
    try:
        metadata = load_metadata_json(metadata_path)
    except (DatasetError, JSONValidationError) as exc:
        errors.append(f"{_METADATA_FILENAME}: {exc}")
        return DatasetValidationResult(valid=False, errors=errors, metadata=None)

    checksums_path = folder / _CHECKSUMS_FILENAME
    if checksums_path.is_file():
        try:
            recorded_checksums: dict[str, str] = json.loads(
                checksums_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{_CHECKSUMS_FILENAME}: could not read ({exc}).")
            recorded_checksums = {}

        for filename, expected_checksum in recorded_checksums.items():
            data_path = folder / filename
            if not data_path.is_file():
                errors.append(f"{filename}: file listed in {_CHECKSUMS_FILENAME} is missing.")
                continue
            if _sha256_of_file(data_path) != expected_checksum:
                errors.append(f"{filename}: checksum mismatch (data has changed or is corrupt).")
    elif metadata.frame_count > 0:
        errors.append(f"{_CHECKSUMS_FILENAME} is missing.")

    if metadata.frame_count > 0:
        if metadata.dataset_format == "hdf5":
            errors.extend(_validate_hdf5_structure(folder, metadata))
        elif metadata.dataset_format == "raw_binary":
            errors.extend(_validate_raw_binary_structure(folder, metadata))

    return DatasetValidationResult(valid=not errors, errors=errors, metadata=metadata)


def _validate_hdf5_structure(folder: Path, metadata: DatasetMetadata) -> list[str]:
    errors: list[str] = []
    path = folder / _HDF5_FILENAME
    if h5py is None:
        errors.append("h5py is not installed; cannot validate the HDF5 data file's structure.")
        return errors
    if not path.is_file():
        errors.append(f"{_HDF5_FILENAME} is missing.")
        return errors

    try:
        with h5py.File(path, "r") as handle:
            for name in ("frames", "frame_ids", "host_timestamps_ns", "device_timestamps_ticks"):
                if name not in handle:
                    errors.append(f"{_HDF5_FILENAME}: missing dataset {name!r}.")
                    continue
                if len(handle[name]) != metadata.frame_count:
                    errors.append(
                        f"{_HDF5_FILENAME}: {name} has {len(handle[name])} entries, "
                        f"expected {metadata.frame_count}."
                    )
            if "frames" in handle:
                frame_shape = handle["frames"].shape[1:3]
                expected_shape = (metadata.height, metadata.width)
                if frame_shape != expected_shape:
                    errors.append(
                        f"{_HDF5_FILENAME}: frame shape {frame_shape} does not match "
                        f"metadata {expected_shape}."
                    )
    except OSError as exc:
        errors.append(f"{_HDF5_FILENAME}: could not open ({exc}).")

    return errors


def _validate_raw_binary_structure(folder: Path, metadata: DatasetMetadata) -> list[str]:
    errors: list[str] = []
    frames_path = folder / _RAW_FRAMES_FILENAME
    index_path = folder / _RAW_INDEX_FILENAME

    if not frames_path.is_file():
        errors.append(f"{_RAW_FRAMES_FILENAME} is missing.")
    if not index_path.is_file():
        errors.append(f"{_RAW_INDEX_FILENAME} is missing.")
    if errors:
        return errors

    with index_path.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.reader(handle)) - 1  # minus header

    if row_count != metadata.frame_count:
        errors.append(
            f"{_RAW_INDEX_FILENAME} has {row_count} entries, expected {metadata.frame_count}."
        )

    return errors

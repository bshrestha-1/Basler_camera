"""Dataset metadata: a single canonical description of a recorded dataset.

:class:`DatasetMetadata` is the source of truth that both the JSON
metadata sidecar file and the embedded HDF5 attribute (see
:mod:`glas.dataset`) are derived from, so the two never disagree about
what a recording contains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from glas.exceptions import DatasetError, JSONValidationError
from glas.version import __version__ as _glas_version


class DatasetMetadata(BaseModel):
    """A complete description of a recorded (or in-progress) dataset.

    Attributes
    ----------
    dataset_format : str
        Storage backend used: ``"hdf5"`` or ``"raw_binary"``.
    camera_model, camera_serial : str
        Identifying information for the camera that captured the data.
    pixel_format : str
        Pixel format of every stored frame, e.g. ``"Mono8"``.
    width, height : int
        Frame dimensions, in pixels.
    created_at_utc : str
        ISO 8601 UTC timestamp the recording started.
    glas_version : str
        GLAS version that created the dataset.
    frame_count : int
        Number of frames stored. ``0`` until the writer finalizes the
        dataset.
    exposure_time_us, gain_db : float or None
        Camera settings at the start of the recording, if known.
    frame_rate_hz : float or None
        Acquisition frame rate cap at the start of the recording, if known.
        See :attr:`glas.camera.Camera.frame_rate_hz`.
    roi_offset_x, roi_offset_y : int
        Region-of-interest offset, in pixels, of the ``width`` x ``height``
        crop relative to the sensor's top-left corner. Zero for a
        full-sensor capture.
    camera_settings : dict
        Every other camera setting that affects how a frame is captured
        and therefore matters for reproducing the recording: gamma,
        binning, horizontal/vertical flip, auto-exposure/auto-gain mode,
        whether the frame rate cap is enabled, and hardware trigger
        state. Populated by :meth:`glas.controller.RecorderController.start_recording`
        from :class:`~glas.camera.Camera`; open-ended (a plain dict, not
        named fields) so a future camera setting doesn't require a schema
        change here.
    notes : str
        Free-text operator notes.
    extra : dict
        Open-ended additional fields (e.g. experiment name, operator),
        forward-compatible with later phases (the Experiment Manager,
        Phase 9) without requiring schema changes here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_format: Literal["hdf5", "raw_binary"]
    camera_model: str
    camera_serial: str
    pixel_format: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    created_at_utc: str = Field(min_length=1)
    glas_version: str = _glas_version
    frame_count: int = Field(default=0, ge=0)
    exposure_time_us: float | None = None
    gain_db: float | None = None
    frame_rate_hz: float | None = None
    roi_offset_x: int = Field(default=0, ge=0)
    roi_offset_y: int = Field(default=0, ge=0)
    camera_settings: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this metadata as a plain, JSON-serializable dict."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        """Build validated :class:`DatasetMetadata` from a plain dict.

        Parameters
        ----------
        data : dict
            Must match this model exactly (every field required except
            those with defaults, no unrecognized keys -- put anything
            not covered by a named field into ``extra`` instead).

        Raises
        ------
        JSONValidationError
            If ``data`` does not match the expected structure.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise JSONValidationError.from_pydantic(exc, context="Dataset metadata") from exc

    def replace(self, **changes: Any) -> DatasetMetadata:
        """Return a copy of this metadata with ``changes`` applied.

        Intended for internally-controlled updates (e.g. setting the
        final ``frame_count`` on finalize); unlike :meth:`from_dict`,
        the result is not re-validated.
        """
        return self.model_copy(update=changes)


def save_metadata_json(metadata: DatasetMetadata, path: Path) -> None:
    """Write ``metadata`` to ``path`` as pretty-printed JSON.

    Raises
    ------
    DatasetError
        If the file cannot be written.
    """
    try:
        path.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise DatasetError(f"Could not write metadata file {path}: {exc}") from exc


def load_metadata_json(path: Path) -> DatasetMetadata:
    """Read and validate a :class:`DatasetMetadata` from a JSON file.

    Raises
    ------
    DatasetError
        If the file does not exist or cannot be read or parsed as JSON.
    JSONValidationError
        If the parsed data does not match the expected structure.
    """
    if not path.is_file():
        raise DatasetError(f"Metadata file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read metadata file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise JSONValidationError(
            f"Metadata file {path} must contain a JSON object at the top level, "
            f"got {type(data).__name__}."
        )
    return DatasetMetadata.from_dict(data)

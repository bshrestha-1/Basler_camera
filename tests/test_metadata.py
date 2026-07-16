"""Tests for glas.metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from glas.exceptions import DatasetError, JSONValidationError
from glas.metadata import DatasetMetadata, load_metadata_json, save_metadata_json


def _make_metadata(**overrides: object) -> DatasetMetadata:
    defaults = dict(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=640,
        height=480,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)  # type: ignore[arg-type]


def test_to_dict_round_trips_through_from_dict() -> None:
    metadata = _make_metadata(notes="test run", extra={"operator": "bijay"})
    rebuilt = DatasetMetadata.from_dict(metadata.to_dict())
    assert rebuilt == metadata


def test_from_dict_rejects_missing_required_field() -> None:
    data = _make_metadata().to_dict()
    del data["camera_model"]
    with pytest.raises(JSONValidationError):
        DatasetMetadata.from_dict(data)


def test_from_dict_rejects_unknown_field() -> None:
    data = _make_metadata().to_dict()
    data["totally_unexpected_field"] = 1
    with pytest.raises(JSONValidationError):
        DatasetMetadata.from_dict(data)


def test_from_dict_rejects_invalid_dataset_format() -> None:
    data = _make_metadata().to_dict()
    data["dataset_format"] = "not_a_format"
    with pytest.raises(JSONValidationError):
        DatasetMetadata.from_dict(data)


def test_from_dict_rejects_non_positive_width() -> None:
    data = _make_metadata().to_dict()
    data["width"] = 0
    with pytest.raises(JSONValidationError):
        DatasetMetadata.from_dict(data)


def test_replace_returns_new_instance_with_changes() -> None:
    metadata = _make_metadata()
    updated = metadata.replace(frame_count=100)
    assert updated.frame_count == 100
    assert metadata.frame_count == 0
    assert updated.camera_model == metadata.camera_model


def test_defaults_are_applied() -> None:
    metadata = _make_metadata()
    assert metadata.frame_count == 0
    assert metadata.exposure_time_us is None
    assert metadata.gain_db is None
    assert metadata.notes == ""
    assert metadata.extra == {}


def test_save_and_load_metadata_json_round_trip(tmp_path: Path) -> None:
    metadata = _make_metadata(frame_count=42, exposure_time_us=5000.0, gain_db=6.0)
    path = tmp_path / "metadata.json"
    save_metadata_json(metadata, path)

    loaded = load_metadata_json(path)
    assert loaded == metadata


def test_load_metadata_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_metadata_json(tmp_path / "does_not_exist.json")


def test_load_metadata_json_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text("{not valid json")
    with pytest.raises(DatasetError):
        load_metadata_json(path)


def test_load_metadata_json_non_object_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(JSONValidationError):
        load_metadata_json(path)

"""Tests for glas.dataset."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from glas.dataset import (
    Dataset,
    create_experiment_folder,
    iter_frames,
    resolve_dataset_format,
    validate_dataset,
)
from glas.exceptions import DatasetError, DatasetFormatError, DatasetIOError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _make_metadata(**overrides: object) -> DatasetMetadata:
    # dataset_format defaults to "hdf5" as a placeholder: Dataset.create()
    # always overwrites it with the format it actually resolved to, so the
    # seed value here is never what ends up on disk.
    defaults = dict(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=8,
        height=4,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)  # type: ignore[arg-type]


def _make_frame(frame_id: int, width: int = 8, height: int = 4, value: int = 0) -> Frame:
    return Frame(
        frame_id=frame_id,
        image=np.full((height, width), value, dtype=np.uint8),
        pixel_format="Mono8",
        host_timestamp_ns=frame_id * 1000,
        device_timestamp_ticks=frame_id,
    )


class TestCreateExperimentFolder:
    def test_creates_first_run_folder(self, tmp_path: Path) -> None:
        folder = create_experiment_folder(tmp_path)
        assert folder == tmp_path / "Run0001"
        assert folder.is_dir()

    def test_increments_past_existing_runs(self, tmp_path: Path) -> None:
        (tmp_path / "Run0001").mkdir()
        (tmp_path / "Run0003").mkdir()
        folder = create_experiment_folder(tmp_path)
        assert folder == tmp_path / "Run0004"

    def test_ignores_non_matching_entries(self, tmp_path: Path) -> None:
        (tmp_path / "Run0001").mkdir()
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "RunXYZ").mkdir()
        folder = create_experiment_folder(tmp_path)
        assert folder == tmp_path / "Run0002"

    def test_creates_base_dir_if_missing(self, tmp_path: Path) -> None:
        base = tmp_path / "experiments"
        folder = create_experiment_folder(base)
        assert folder == base / "Run0001"

    def test_custom_prefix_and_width(self, tmp_path: Path) -> None:
        folder = create_experiment_folder(tmp_path, prefix="Session", width=2)
        assert folder == tmp_path / "Session01"


class TestHdf5Dataset:
    def test_append_and_finalize_writes_expected_files(self, tmp_path: Path) -> None:
        metadata = _make_metadata(dataset_format="hdf5")
        dataset = Dataset.create(tmp_path, metadata, dataset_format="hdf5")

        for i in range(3):
            dataset.append_frame(_make_frame(i, value=i))

        final_metadata = dataset.finalize()

        assert final_metadata.frame_count == 3
        assert (tmp_path / "frames.h5").is_file()
        assert (tmp_path / "metadata.json").is_file()
        assert (tmp_path / "checksums.json").is_file()

    def test_written_frame_data_is_correct(self, tmp_path: Path) -> None:
        metadata = _make_metadata(dataset_format="hdf5")
        dataset = Dataset.create(tmp_path, metadata, dataset_format="hdf5")
        for i in range(3):
            dataset.append_frame(_make_frame(i, value=i * 10))
        dataset.finalize()

        with h5py.File(tmp_path / "frames.h5", "r") as handle:
            frames = handle["frames"][:]
            assert frames.shape == (3, 4, 8)
            assert frames[1].mean() == 10
            np.testing.assert_array_equal(handle["frame_ids"][:], [0, 1, 2])
            assert "metadata_json" in handle.attrs
            embedded = json.loads(handle.attrs["metadata_json"])
            assert embedded["frame_count"] == 3

    def test_append_after_finalize_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        dataset.append_frame(_make_frame(0))
        dataset.finalize()
        with pytest.raises(DatasetError):
            dataset.append_frame(_make_frame(1))

    def test_finalize_is_idempotent(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        dataset.append_frame(_make_frame(0))
        first = dataset.finalize()
        second = dataset.finalize()
        assert first == second

    def test_wrong_shape_frame_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(width=8, height=4), dataset_format="hdf5")
        dataset.append_frame(_make_frame(0, width=8, height=4))
        with pytest.raises(DatasetIOError):
            dataset.append_frame(_make_frame(1, width=16, height=4))

    def test_empty_dataset_writes_metadata_but_no_data_file(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        metadata = dataset.finalize()

        assert metadata.frame_count == 0
        assert not (tmp_path / "frames.h5").is_file()
        assert (tmp_path / "metadata.json").is_file()
        checksums = json.loads((tmp_path / "checksums.json").read_text())
        assert checksums == {}

    def test_context_manager_finalizes_on_exit(self, tmp_path: Path) -> None:
        with Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5") as dataset:
            dataset.append_frame(_make_frame(0))
        assert (tmp_path / "metadata.json").is_file()

    def test_explicit_hdf5_request_is_respected_in_metadata(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        assert dataset.metadata.dataset_format == "hdf5"


class TestRawBinaryDataset:
    def test_append_and_finalize_writes_expected_files(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="raw_binary")
        for i in range(3):
            dataset.append_frame(_make_frame(i, value=i))
        metadata = dataset.finalize()

        assert metadata.frame_count == 3
        assert metadata.dataset_format == "raw_binary"
        assert (tmp_path / "frames.bin").is_file()
        assert (tmp_path / "frames_index.csv").is_file()

    def test_raw_binary_file_size_matches_frame_count(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(width=8, height=4), "raw_binary")
        for i in range(5):
            dataset.append_frame(_make_frame(i, width=8, height=4))
        dataset.finalize()

        expected_bytes = 5 * 8 * 4  # frame_count * width * height * 1 byte/px (Mono8)
        assert (tmp_path / "frames.bin").stat().st_size == expected_bytes

    def test_index_csv_has_expected_rows(self, tmp_path: Path) -> None:
        import csv

        dataset = Dataset.create(tmp_path, _make_metadata(), "raw_binary")
        for i in range(3):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        with (tmp_path / "frames_index.csv").open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["frame_id", "host_timestamp_ns", "device_timestamp_ticks"]
        assert len(rows) == 4  # header + 3 frames

    def test_wrong_shape_frame_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(width=8, height=4), "raw_binary")
        dataset.append_frame(_make_frame(0, width=8, height=4))
        with pytest.raises(DatasetIOError):
            dataset.append_frame(_make_frame(1, width=8, height=99))


class TestIterFrames:
    def test_hdf5_round_trip_preserves_order_and_content(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        for i in range(3):
            dataset.append_frame(_make_frame(i, value=i * 10))
        dataset.finalize()

        frames = list(iter_frames(tmp_path))
        assert [f.frame_id for f in frames] == [0, 1, 2]
        assert [int(f.image.mean()) for f in frames] == [0, 10, 20]
        assert frames[1].host_timestamp_ns == 1000
        assert frames[1].device_timestamp_ticks == 1
        assert frames[0].pixel_format == "Mono8"

    def test_raw_binary_round_trip_preserves_order_and_content(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="raw_binary")
        for i in range(3):
            dataset.append_frame(_make_frame(i, value=i * 10))
        dataset.finalize()

        frames = list(iter_frames(tmp_path))
        assert [f.frame_id for f in frames] == [0, 1, 2]
        assert [int(f.image.mean()) for f in frames] == [0, 10, 20]
        assert frames[1].host_timestamp_ns == 1000
        assert frames[1].device_timestamp_ticks == 1

    def test_raw_binary_round_trip_with_16_bit_pixel_format(self, tmp_path: Path) -> None:
        metadata = _make_metadata(pixel_format="Mono16")
        dataset = Dataset.create(tmp_path, metadata, dataset_format="raw_binary")
        image = np.full((4, 8), 4000, dtype=np.uint16)
        frame = Frame(
            frame_id=0,
            image=image,
            pixel_format="Mono16",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
        dataset.append_frame(frame)
        dataset.finalize()

        frames = list(iter_frames(tmp_path))
        assert frames[0].image.dtype == np.uint16
        np.testing.assert_array_equal(frames[0].image, image)

    def test_empty_dataset_yields_nothing(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        dataset.finalize()
        assert list(iter_frames(tmp_path)) == []

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError):
            list(iter_frames(tmp_path))

    def test_missing_hdf5_data_file_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")
        dataset.append_frame(_make_frame(0))
        dataset.finalize()
        (tmp_path / "frames.h5").unlink()

        with pytest.raises(DatasetIOError):
            list(iter_frames(tmp_path))

    def test_missing_raw_binary_data_file_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="raw_binary")
        dataset.append_frame(_make_frame(0))
        dataset.finalize()
        (tmp_path / "frames.bin").unlink()

        with pytest.raises(DatasetIOError):
            list(iter_frames(tmp_path))

    def test_truncated_raw_binary_file_raises(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="raw_binary")
        dataset.append_frame(_make_frame(0))
        dataset.append_frame(_make_frame(1))
        dataset.finalize()

        frames_path = tmp_path / "frames.bin"
        frames_path.write_bytes(frames_path.read_bytes()[:4])  # truncate mid-frame

        with pytest.raises(DatasetIOError):
            list(iter_frames(tmp_path))


class TestResolveDatasetFormat:
    def test_auto_resolves_to_hdf5_when_available(self) -> None:
        assert resolve_dataset_format("auto") == "hdf5"

    def test_auto_resolves_to_raw_binary_when_h5py_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("glas.dataset.h5py", None)
        assert resolve_dataset_format("auto") == "raw_binary"

    def test_explicit_hdf5_passes_through(self) -> None:
        assert resolve_dataset_format("hdf5") == "hdf5"

    def test_explicit_raw_binary_passes_through(self) -> None:
        assert resolve_dataset_format("raw_binary") == "raw_binary"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(DatasetFormatError):
            resolve_dataset_format("not_a_format")


class TestDatasetFormatSelection:
    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError):
            Dataset.create(tmp_path, _make_metadata(), dataset_format="not_a_format")

    def test_auto_selects_hdf5_when_available(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="auto")
        assert dataset.metadata.dataset_format == "hdf5"

    def test_auto_falls_back_to_raw_binary_when_h5py_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("glas.dataset.h5py", None)
        dataset = Dataset.create(tmp_path, _make_metadata(), dataset_format="auto")
        assert dataset.metadata.dataset_format == "raw_binary"

        dataset.append_frame(_make_frame(0))
        dataset.finalize()
        assert (tmp_path / "frames.bin").is_file()

    def test_explicit_hdf5_without_h5py_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("glas.dataset.h5py", None)
        with pytest.raises(DatasetFormatError):
            Dataset.create(tmp_path, _make_metadata(), dataset_format="hdf5")


class TestValidateDataset:
    def test_valid_hdf5_dataset_passes(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "hdf5")
        for i in range(4):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        result = validate_dataset(tmp_path)
        assert result.valid, result.errors
        assert result.errors == []
        assert result.metadata is not None
        assert result.metadata.frame_count == 4

    def test_valid_raw_binary_dataset_passes(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "raw_binary")
        for i in range(4):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        result = validate_dataset(tmp_path)
        assert result.valid, result.errors

    def test_valid_empty_dataset_passes(self, tmp_path: Path) -> None:
        Dataset.create(tmp_path, _make_metadata(), "hdf5").finalize()
        result = validate_dataset(tmp_path)
        assert result.valid, result.errors

    def test_missing_metadata_file_is_invalid(self, tmp_path: Path) -> None:
        result = validate_dataset(tmp_path)
        assert not result.valid
        assert result.metadata is None
        assert any("metadata.json" in error for error in result.errors)

    def test_corrupted_data_file_fails_checksum(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "hdf5")
        for i in range(3):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        with (tmp_path / "frames.h5").open("ab") as handle:
            handle.write(b"corruption")

        result = validate_dataset(tmp_path)
        assert not result.valid
        assert any("checksum mismatch" in error for error in result.errors)

    def test_missing_data_file_is_invalid(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "hdf5")
        for i in range(3):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        (tmp_path / "frames.h5").unlink()

        result = validate_dataset(tmp_path)
        assert not result.valid

    def test_missing_checksums_file_is_invalid_when_frames_exist(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "hdf5")
        dataset.append_frame(_make_frame(0))
        dataset.finalize()

        (tmp_path / "checksums.json").unlink()

        result = validate_dataset(tmp_path)
        assert not result.valid
        assert any("checksums.json" in error for error in result.errors)

    def test_tampered_frame_count_in_metadata_is_invalid(self, tmp_path: Path) -> None:
        dataset = Dataset.create(tmp_path, _make_metadata(), "hdf5")
        for i in range(3):
            dataset.append_frame(_make_frame(i))
        dataset.finalize()

        metadata_path = tmp_path / "metadata.json"
        data = json.loads(metadata_path.read_text())
        data["frame_count"] = 999
        metadata_path.write_text(json.dumps(data))

        result = validate_dataset(tmp_path)
        assert not result.valid
        assert any("999" in error or "entries" in error for error in result.errors)

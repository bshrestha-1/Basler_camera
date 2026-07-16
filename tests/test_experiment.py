"""Tests for glas.experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glas.dataset import Dataset
from glas.exceptions import ExperimentNotFoundError
from glas.experiment import (
    NAME_KEY,
    TAGS_KEY,
    ExperimentManager,
    ExperimentSummary,
    build_experiment_extra,
)
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _make_experiment(
    manager: ExperimentManager,
    *,
    name: str = "",
    tags: list[str] | None = None,
    notes: str = "",
    camera_model: str = "acA640-750um",
    finalize: bool = True,
) -> Path:
    folder = manager.new_folder()
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model=camera_model,
        camera_serial="12345678",
        pixel_format="Mono8",
        width=4,
        height=4,
        created_at_utc="2026-07-13T00:00:00+00:00",
        notes=notes,
        extra=build_experiment_extra(name=name, tags=tags),
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    dataset.append_frame(
        Frame(
            frame_id=0,
            image=np.zeros((4, 4), dtype=np.uint8),
            pixel_format="Mono8",
            host_timestamp_ns=0,
            device_timestamp_ticks=0,
        )
    )
    if finalize:
        dataset.finalize()
    return folder


class TestBuildExperimentExtra:
    def test_empty_name_and_tags_produce_no_reserved_keys(self) -> None:
        assert build_experiment_extra() == {}

    def test_name_only(self) -> None:
        assert build_experiment_extra(name="run A") == {NAME_KEY: "run A"}

    def test_tags_only(self) -> None:
        assert build_experiment_extra(tags=["a", "b"]) == {TAGS_KEY: ["a", "b"]}

    def test_merges_with_existing_extra(self) -> None:
        result = build_experiment_extra(name="run A", extra={"operator": "bijay"})
        assert result == {"operator": "bijay", NAME_KEY: "run A"}

    def test_empty_tags_sequence_is_omitted(self) -> None:
        assert build_experiment_extra(tags=[]) == {}


class TestNewFolder:
    def test_creates_first_run_folder(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        folder = manager.new_folder()
        assert folder == tmp_path / "Run0001"
        assert folder.is_dir()

    def test_creates_base_dir_if_missing(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path / "data")
        folder = manager.new_folder()
        assert folder == tmp_path / "data" / "Run0001"


class TestListExperiments:
    def test_empty_base_dir_returns_empty_list(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        assert manager.list_experiments() == []

    def test_nonexistent_base_dir_returns_empty_list(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path / "does_not_exist")
        assert manager.list_experiments() == []

    def test_lists_finalized_experiments_in_order(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="first")
        _make_experiment(manager, name="second")

        summaries = manager.list_experiments()
        assert [s.name for s in summaries] == ["first", "second"]
        assert [s.run_id for s in summaries] == ["Run0001", "Run0002"]

    def test_skips_folders_without_metadata_json(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="finalized")
        (tmp_path / "Run0002").mkdir()  # in-progress: no metadata.json yet

        summaries = manager.list_experiments()
        assert [s.run_id for s in summaries] == ["Run0001"]

    def test_skips_non_directory_entries(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="finalized")
        (tmp_path / "notes.txt").write_text("hello")

        summaries = manager.list_experiments()
        assert [s.run_id for s in summaries] == ["Run0001"]

    def test_skips_folders_with_corrupt_metadata_json(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="finalized")
        broken = tmp_path / "Run0002"
        broken.mkdir()
        (broken / "metadata.json").write_text("not json")

        summaries = manager.list_experiments()
        assert [s.run_id for s in summaries] == ["Run0001"]

    def test_experiment_without_name_or_tags_has_empty_defaults(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager)

        summary = manager.list_experiments()[0]
        assert summary.name == ""
        assert summary.tags == []

    def test_summary_surfaces_expected_fields(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="run A", tags=["x"], notes="hello", camera_model="acA640")

        summary = manager.list_experiments()[0]
        assert isinstance(summary, ExperimentSummary)
        assert summary.notes == "hello"
        assert summary.camera_model == "acA640"
        assert summary.frame_count == 1
        assert summary.created_at_utc == "2026-07-13T00:00:00+00:00"
        assert summary.metadata.camera_model == "acA640"


class TestSearchExperiments:
    def test_filters_by_name_contains_case_insensitively(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="Shaker Sweep")
        _make_experiment(manager, name="Convection Test")

        results = manager.search_experiments(name_contains="shaker")
        assert [s.name for s in results] == ["Shaker Sweep"]

    def test_filters_by_tag(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="a", tags=["brazil-nut", "60hz"])
        _make_experiment(manager, name="b", tags=["convection"])

        results = manager.search_experiments(tag="brazil-nut")
        assert [s.name for s in results] == ["a"]

    def test_filters_by_camera_model(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="a", camera_model="acA640-750um")
        _make_experiment(manager, name="b", camera_model="other-camera")

        results = manager.search_experiments(camera_model="other-camera")
        assert [s.name for s in results] == ["b"]

    def test_combining_filters_is_an_and(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="match", tags=["keep"])
        _make_experiment(manager, name="match", tags=["drop"])

        results = manager.search_experiments(name_contains="match", tag="keep")
        assert len(results) == 1
        assert results[0].tags == ["keep"]

    def test_no_filters_returns_everything(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="a")
        _make_experiment(manager, name="b")

        assert len(manager.search_experiments()) == 2


class TestGetExperiment:
    def test_returns_matching_summary(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="only one")

        summary = manager.get_experiment("Run0001")
        assert summary.name == "only one"

    def test_missing_run_id_raises(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        with pytest.raises(ExperimentNotFoundError):
            manager.get_experiment("Run9999")

    def test_unfinalized_run_id_raises(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        manager.new_folder()  # created but never finalized
        with pytest.raises(ExperimentNotFoundError):
            manager.get_experiment("Run0001")


class TestExperimentSummary:
    def test_is_frozen(self, tmp_path: Path) -> None:
        manager = ExperimentManager(tmp_path)
        _make_experiment(manager, name="x")
        summary = manager.list_experiments()[0]

        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            summary.name = "y"  # type: ignore[misc]

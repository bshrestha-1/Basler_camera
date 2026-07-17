"""Tests for glas.ai.annotation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from glas.ai.annotation import (
    FrameAnnotation,
    ParticleAnnotation,
    auto_annotate_dataset,
    collect_class_names,
    prepare_yolo_dataset,
)
from glas.dataset import Dataset
from glas.exceptions import AIDatasetError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


def _make_blob_dataset(tmp_path: Path, frame_count: int = 4, blobs_per_frame: int = 2) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=100,
        height=100,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        image = np.zeros((100, 100), dtype=np.uint8)
        for b in range(blobs_per_frame):
            cv2.circle(image, (20 + b * 30, 20 + i * 5), 6, 255, -1)
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=image,
                pixel_format="Mono8",
                host_timestamp_ns=i * 1000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


class TestParticleAnnotation:
    def test_rejects_non_positive_width(self) -> None:
        with pytest.raises(ValidationError):
            ParticleAnnotation(label="particle", x_center=1, y_center=1, width=0, height=1)

    def test_rejects_non_positive_height(self) -> None:
        with pytest.raises(ValidationError):
            ParticleAnnotation(label="particle", x_center=1, y_center=1, width=1, height=-1)

    def test_is_frozen(self) -> None:
        annotation = ParticleAnnotation(label="particle", x_center=1, y_center=1, width=2, height=2)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            annotation.label = "other"  # type: ignore[misc]


class TestFrameAnnotation:
    def test_defaults_to_empty_annotations(self) -> None:
        frame_annotation = FrameAnnotation(
            image_path=Path("a.png"), image_width=10, image_height=10
        )
        assert frame_annotation.annotations == []

    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValidationError):
            FrameAnnotation(image_path=Path("a.png"), image_width=0, image_height=10)


class TestAutoAnnotateDataset:
    def test_writes_one_image_per_frame(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=3)
        output_dir = tmp_path / "annotated"
        annotations = auto_annotate_dataset(folder, output_dir)

        assert len(annotations) == 3
        for frame_annotation in annotations:
            assert frame_annotation.image_path.exists()

    def test_detects_expected_number_of_particles(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=2)
        annotations = auto_annotate_dataset(folder, tmp_path / "annotated")

        assert len(annotations[0].annotations) == 2

    def test_applies_the_given_label(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=1)
        annotations = auto_annotate_dataset(folder, tmp_path / "annotated", label="glass_bead")

        assert annotations[0].annotations[0].label == "glass_bead"

    def test_min_area_filters_out_small_blobs(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=2)
        annotations = auto_annotate_dataset(folder, tmp_path / "annotated", min_area=10_000)

        assert annotations[0].annotations == []

    def test_boxes_are_centered_on_the_blob(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=1)
        annotations = auto_annotate_dataset(folder, tmp_path / "annotated")

        box = annotations[0].annotations[0]
        assert box.x_center == pytest.approx(20, abs=1.5)
        assert box.y_center == pytest.approx(20, abs=1.5)


class TestCollectClassNames:
    def test_returns_sorted_unique_labels(self) -> None:
        annotations = [
            FrameAnnotation(
                image_path=Path("a.png"),
                image_width=10,
                image_height=10,
                annotations=[
                    ParticleAnnotation(
                        label="steel_ball", x_center=1, y_center=1, width=2, height=2
                    ),
                    ParticleAnnotation(
                        label="glass_bead", x_center=2, y_center=2, width=2, height=2
                    ),
                ],
            ),
            FrameAnnotation(
                image_path=Path("b.png"),
                image_width=10,
                image_height=10,
                annotations=[
                    ParticleAnnotation(
                        label="glass_bead", x_center=3, y_center=3, width=2, height=2
                    )
                ],
            ),
        ]
        assert collect_class_names(annotations) == ["glass_bead", "steel_ball"]

    def test_empty_annotations_yield_no_classes(self) -> None:
        assert collect_class_names([]) == []


class TestPrepareYoloDataset:
    def _make_annotations(self, tmp_path: Path, count: int = 10) -> list[FrameAnnotation]:
        annotations = []
        for i in range(count):
            image_path = tmp_path / "images" / f"frame_{i:03d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.zeros((50, 50), dtype=np.uint8))
            annotations.append(
                FrameAnnotation(
                    image_path=image_path,
                    image_width=50,
                    image_height=50,
                    annotations=[
                        ParticleAnnotation(
                            label="particle", x_center=25, y_center=25, width=10, height=10
                        )
                    ],
                )
            )
        return annotations

    def test_rejects_empty_annotations(self, tmp_path: Path) -> None:
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset([], ["particle"], tmp_path / "out")

    def test_rejects_empty_class_names(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path)
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset(annotations, [], tmp_path / "out")

    def test_rejects_val_fraction_out_of_range(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path)
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset(annotations, ["particle"], tmp_path / "out", val_fraction=0.0)
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset(annotations, ["particle"], tmp_path / "out", val_fraction=1.0)

    def test_rejects_unknown_label(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path)
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset(annotations, ["not_particle"], tmp_path / "out")

    def test_rejects_too_few_frames_for_split(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path, count=1)
        with pytest.raises(AIDatasetError):
            prepare_yolo_dataset(annotations, ["particle"], tmp_path / "out", val_fraction=0.2)

    def test_writes_data_yaml_with_class_names(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path)
        output_dir = tmp_path / "out"
        data_yaml_path = prepare_yolo_dataset(annotations, ["particle"], output_dir)

        data_yaml = yaml.safe_load(data_yaml_path.read_text())
        assert data_yaml["names"] == {0: "particle"}
        assert data_yaml["train"] == "images/train"
        assert data_yaml["val"] == "images/val"

    def test_splits_into_train_and_val(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path, count=10)
        output_dir = tmp_path / "out"
        prepare_yolo_dataset(annotations, ["particle"], output_dir, val_fraction=0.3, seed=1)

        train_images = list((output_dir / "images" / "train").iterdir())
        val_images = list((output_dir / "images" / "val").iterdir())
        assert len(train_images) == 7
        assert len(val_images) == 3
        assert len(train_images) + len(val_images) == 10

    def test_writes_a_label_file_per_image(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path, count=5)
        output_dir = tmp_path / "out"
        prepare_yolo_dataset(annotations, ["particle"], output_dir, val_fraction=0.2)

        train_labels = list((output_dir / "labels" / "train").iterdir())
        val_labels = list((output_dir / "labels" / "val").iterdir())
        assert len(train_labels) + len(val_labels) == 5

    def test_label_file_content_is_normalized_yolo_format(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path, count=2)
        output_dir = tmp_path / "out"
        prepare_yolo_dataset(annotations, ["particle"], output_dir, val_fraction=0.5, seed=2)

        label_files = list((output_dir / "labels").rglob("*.txt"))
        assert len(label_files) == 2
        for label_file in label_files:
            class_id, x, y, w, h = label_file.read_text().strip().split()
            assert class_id == "0"
            assert 0.0 <= float(x) <= 1.0
            assert 0.0 <= float(y) <= 1.0
            assert 0.0 <= float(w) <= 1.0
            assert 0.0 <= float(h) <= 1.0

    def test_deterministic_split_given_same_seed(self, tmp_path: Path) -> None:
        annotations = self._make_annotations(tmp_path, count=10)
        first = prepare_yolo_dataset(annotations, ["particle"], tmp_path / "out1", seed=42)
        second = prepare_yolo_dataset(annotations, ["particle"], tmp_path / "out2", seed=42)

        first_train = sorted(p.name for p in (first.parent / "images" / "train").iterdir())
        second_train = sorted(p.name for p in (second.parent / "images" / "train").iterdir())
        assert first_train == second_train

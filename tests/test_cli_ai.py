"""Tests for the `glas ai` CLI subcommand group.

Heavy AI operations (YOLO inference/training, SAM2 inference/training) are
monkeypatched at the `glas.cli` module level -- the same names `cli.py`
imported them under -- so these tests exercise the CLI's own argument
wiring, error handling, and output formatting without needing real model
weights or GPU time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import glas.cli as glas_cli
from glas.ai.annotation import FrameAnnotation, ParticleAnnotation
from glas.ai.sam2_segmenter import ParticleSegment, compute_shape_metrics
from glas.ai.sam2_train import Sam2Example, Sam2TrainingResult
from glas.ai.yolo_train import YoloTrainingResult
from glas.analysis.particle_tracking import TrackedParticle
from glas.analysis.tracking_utils import Detection
from glas.cli import app
from glas.dataset import Dataset
from glas.exceptions import AIDatasetError, AIDependencyError, AIModelError
from glas.frame import Frame
from glas.metadata import DatasetMetadata

runner = CliRunner()


def _make_single_frame_dataset(tmp_path: Path, frame_count: int = 1) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=64,
        height=64,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        dataset.append_frame(
            Frame(
                frame_id=i,
                image=np.zeros((64, 64), dtype=np.uint8),
                pixel_format="Mono8",
                host_timestamp_ns=i * 1000,
                device_timestamp_ticks=i,
            )
        )
    dataset.finalize()
    return folder


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius**2


class _FakeYoloDetector:
    def __init__(self, weights: str, **kwargs: object) -> None:
        self.weights = weights
        self.detections: list[Detection] = [Detection(x=32, y=32, radius=8, area=200)]

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detections


class _RaisingYoloDetector:
    exception_cls: type[Exception] = AIDependencyError

    def __init__(self, weights: str, **kwargs: object) -> None:
        raise self.exception_cls("missing dependency")


class _FakeSam2Segmenter:
    def __init__(self, **kwargs: object) -> None:
        mask = _circle_mask((64, 64), (32, 32), 8)
        self.segments: list[ParticleSegment] = [
            ParticleSegment(mask=mask, score=0.9, metrics=compute_shape_metrics(mask))
        ]

    def segment_frame(
        self, image: np.ndarray, detections: list[Detection]
    ) -> list[ParticleSegment]:
        return self.segments


class TestAiDetect:
    def test_reports_tracked_particles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        history = {
            0: [
                TrackedParticle(
                    track_id=0,
                    frame_id=0,
                    x=1,
                    y=1,
                    radius=1,
                    area=1,
                    label="glass_bead",
                    confidence=0.9,
                    is_intruder=False,
                )
            ]
        }
        monkeypatch.setattr(glas_cli, "track_dataset_yolo", lambda *a, **k: history)

        result = runner.invoke(app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 0
        assert "Tracked 1 particle(s)" in result.output
        assert "glass_bead: 1" in result.output

    def test_reports_intruder_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        history = {
            0: [
                TrackedParticle(
                    track_id=0,
                    frame_id=0,
                    x=1,
                    y=1,
                    radius=1,
                    area=1,
                    label="intruder",
                    confidence=0.9,
                    is_intruder=True,
                )
            ]
        }
        monkeypatch.setattr(glas_cli, "track_dataset_yolo", lambda *a, **k: history)

        result = runner.invoke(app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 0
        assert "Intruder(s) detected: 1" in result.output

    def test_reports_no_particles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(glas_cli, "track_dataset_yolo", lambda *a, **k: {})

        result = runner.invoke(app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 0
        assert "No particles detected." in result.output

    def test_writes_csv_when_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        history = {0: [TrackedParticle(track_id=0, frame_id=0, x=1, y=1, radius=1, area=1)]}
        monkeypatch.setattr(glas_cli, "track_dataset_yolo", lambda *a, **k: history)
        csv_path = tmp_path / "out.csv"

        result = runner.invoke(
            app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt", "--csv", str(csv_path)]
        )
        assert result.exit_code == 0
        assert csv_path.exists()
        assert "Wrote 1 row(s)" in result.output

    def test_dependency_error_shows_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def raise_missing(*args: object, **kwargs: object) -> None:
            raise AIDependencyError("ultralytics is required")

        monkeypatch.setattr(glas_cli, "track_dataset_yolo", raise_missing)
        monkeypatch.setattr(glas_cli, "missing_ai_packages", lambda: ["ultralytics"])

        result = runner.invoke(app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 1
        assert "ultralytics" in result.output
        assert "pip install" in result.output

    def test_model_error_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def raise_model_error(*args: object, **kwargs: object) -> None:
            raise AIModelError("bad weights file")

        monkeypatch.setattr(glas_cli, "track_dataset_yolo", raise_model_error)

        result = runner.invoke(app, ["ai", "detect", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 1
        assert "Detection failed" in result.output


class TestAiPrepareYoloDataset:
    def test_writes_data_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        annotation = FrameAnnotation(
            image_path=tmp_path / "a.png",
            image_width=10,
            image_height=10,
            annotations=[
                ParticleAnnotation(label="particle", x_center=5, y_center=5, width=2, height=2)
            ],
        )
        monkeypatch.setattr(glas_cli, "auto_annotate_dataset", lambda *a, **k: [annotation])
        data_yaml_path = tmp_path / "out" / "data.yaml"
        monkeypatch.setattr(glas_cli, "prepare_yolo_dataset", lambda *a, **k: data_yaml_path)

        result = runner.invoke(
            app, ["ai", "prepare-yolo-dataset", str(tmp_path / "ds"), str(tmp_path / "out")]
        )
        assert result.exit_code == 0
        assert f"Wrote {data_yaml_path}" in result.output

    def test_dataset_error_fails_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(glas_cli, "auto_annotate_dataset", lambda *a, **k: [])

        def raise_dataset_error(*args: object, **kwargs: object) -> None:
            raise AIDatasetError("empty annotation set")

        monkeypatch.setattr(glas_cli, "prepare_yolo_dataset", raise_dataset_error)

        result = runner.invoke(
            app, ["ai", "prepare-yolo-dataset", str(tmp_path / "ds"), str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert "Dataset preparation failed" in result.output


class TestAiTrainYolo:
    def test_reports_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result_obj = YoloTrainingResult(
            best_weights=tmp_path / "best.pt",
            last_weights=tmp_path / "last.pt",
            save_dir=tmp_path / "run",
            metrics={"metrics/mAP50(B)": 0.8},
        )
        monkeypatch.setattr(glas_cli, "train_yolo", lambda config: result_obj)

        result = runner.invoke(app, ["ai", "train-yolo", str(tmp_path / "data.yaml")])
        assert result.exit_code == 0
        assert "Best weights:" in result.output
        assert "metrics/mAP50(B): 0.8000" in result.output

    def test_dependency_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(config: object) -> None:
            raise AIDependencyError("ultralytics is required")

        monkeypatch.setattr(glas_cli, "train_yolo", raise_missing)
        monkeypatch.setattr(glas_cli, "missing_ai_packages", lambda: ["ultralytics"])

        result = runner.invoke(app, ["ai", "train-yolo", str(tmp_path / "data.yaml")])
        assert result.exit_code == 1
        assert "ultralytics" in result.output

    def test_model_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_model_error(config: object) -> None:
            raise AIModelError("training exploded")

        monkeypatch.setattr(glas_cli, "train_yolo", raise_model_error)

        result = runner.invoke(app, ["ai", "train-yolo", str(tmp_path / "data.yaml")])
        assert result.exit_code == 1
        assert "Training failed" in result.output


class TestAiSegment:
    def test_reports_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        folder = _make_single_frame_dataset(tmp_path)
        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _FakeYoloDetector)
        monkeypatch.setattr(glas_cli, "Sam2Segmenter", _FakeSam2Segmenter)

        result = runner.invoke(app, ["ai", "segment", str(folder), "weights.pt"])
        assert result.exit_code == 0
        assert "Segmented 1 particle(s)." in result.output
        assert "Packing fraction:" in result.output

    def test_writes_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        folder = _make_single_frame_dataset(tmp_path)
        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _FakeYoloDetector)
        monkeypatch.setattr(glas_cli, "Sam2Segmenter", _FakeSam2Segmenter)
        csv_path = tmp_path / "shapes.csv"

        result = runner.invoke(
            app, ["ai", "segment", str(folder), "weights.pt", "--csv", str(csv_path)]
        )
        assert result.exit_code == 0
        assert csv_path.exists()
        assert "Wrote 1 row(s)" in result.output

    def test_no_frames_reports_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = _make_single_frame_dataset(tmp_path, frame_count=0)
        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _FakeYoloDetector)
        monkeypatch.setattr(glas_cli, "Sam2Segmenter", _FakeSam2Segmenter)

        result = runner.invoke(app, ["ai", "segment", str(folder), "weights.pt"])
        assert result.exit_code == 0
        assert "Dataset has no frames." in result.output

    def test_no_particles_reports_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = _make_single_frame_dataset(tmp_path)

        class _EmptySegmenter(_FakeSam2Segmenter):
            def __init__(self, **kwargs: object) -> None:
                self.segments = []

        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _FakeYoloDetector)
        monkeypatch.setattr(glas_cli, "Sam2Segmenter", _EmptySegmenter)

        result = runner.invoke(app, ["ai", "segment", str(folder), "weights.pt"])
        assert result.exit_code == 0
        assert "No particles detected." in result.output

    def test_dependency_error_shows_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RaisingDetector(_RaisingYoloDetector):
            exception_cls = AIDependencyError

        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _RaisingDetector)
        monkeypatch.setattr(glas_cli, "missing_ai_packages", lambda: ["torch", "sam2"])

        result = runner.invoke(app, ["ai", "segment", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 1
        assert "torch" in result.output

    def test_model_load_error_fails_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RaisingDetector(_RaisingYoloDetector):
            exception_cls = AIModelError

        monkeypatch.setattr(glas_cli, "YoloParticleDetector", _RaisingDetector)

        result = runner.invoke(app, ["ai", "segment", str(tmp_path / "ds"), "weights.pt"])
        assert result.exit_code == 1
        assert "Could not load model(s)" in result.output


class TestAiPrepareSam2Dataset:
    def test_writes_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        example = Sam2Example(
            image_path=tmp_path / "a.png", mask_path=tmp_path / "a_mask.png", box=(0, 0, 5, 5)
        )
        monkeypatch.setattr(glas_cli, "auto_annotate_masks", lambda *a, **k: [example])
        manifest_path = tmp_path / "out" / "manifest.json"
        monkeypatch.setattr(glas_cli, "prepare_sam2_dataset", lambda *a, **k: manifest_path)

        result = runner.invoke(
            app, ["ai", "prepare-sam2-dataset", str(tmp_path / "ds"), str(tmp_path / "out")]
        )
        assert result.exit_code == 0
        assert f"Wrote {manifest_path}" in result.output

    def test_dataset_error_fails_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(glas_cli, "auto_annotate_masks", lambda *a, **k: [])

        def raise_dataset_error(*args: object, **kwargs: object) -> None:
            raise AIDatasetError("empty example set")

        monkeypatch.setattr(glas_cli, "prepare_sam2_dataset", raise_dataset_error)

        result = runner.invoke(
            app, ["ai", "prepare-sam2-dataset", str(tmp_path / "ds"), str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert "Dataset preparation failed" in result.output


class TestAiTrainSam2:
    def test_reports_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result_obj = Sam2TrainingResult(
            checkpoint_path=tmp_path / "finetuned.pt",
            epochs=5,
            metrics={"train_loss": 0.123, "val_loss": 0.2, "val_mean_iou": 0.7},
        )
        monkeypatch.setattr(glas_cli, "train_sam2", lambda config: result_obj)

        result = runner.invoke(
            app,
            [
                "ai",
                "train-sam2",
                str(tmp_path / "manifest.json"),
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                str(tmp_path / "base.pt"),
                str(tmp_path / "out.pt"),
            ],
        )
        assert result.exit_code == 0
        assert "Fine-tuned checkpoint:" in result.output
        assert "train_loss: 0.1230" in result.output

    def test_dependency_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(config: object) -> None:
            raise AIDependencyError("sam2 is required")

        monkeypatch.setattr(glas_cli, "train_sam2", raise_missing)
        monkeypatch.setattr(glas_cli, "missing_ai_packages", lambda: ["sam2"])

        result = runner.invoke(
            app,
            [
                "ai",
                "train-sam2",
                str(tmp_path / "manifest.json"),
                "cfg.yaml",
                str(tmp_path / "base.pt"),
                str(tmp_path / "out.pt"),
            ],
        )
        assert result.exit_code == 1
        assert "sam2" in result.output

    def test_dataset_or_model_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_model_error(config: object) -> None:
            raise AIModelError("could not load base model")

        monkeypatch.setattr(glas_cli, "train_sam2", raise_model_error)

        result = runner.invoke(
            app,
            [
                "ai",
                "train-sam2",
                str(tmp_path / "manifest.json"),
                "cfg.yaml",
                str(tmp_path / "base.pt"),
                str(tmp_path / "out.pt"),
            ],
        )
        assert result.exit_code == 1
        assert "Training failed" in result.output

"""Tests for glas.ai.yolo_detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from glas.ai.yolo_detector import (
    YoloDetection,
    YoloParticleDetector,
    track_dataset_yolo,
)
from glas.dataset import Dataset
from glas.exceptions import AIDependencyError, AIModelError
from glas.frame import Frame
from glas.metadata import DatasetMetadata


class _FakeBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[float]) -> None:
        self.xyxy = np.array(xyxy, dtype=np.float64)
        self.conf = np.array(conf, dtype=np.float64)
        self.cls = np.array(cls, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.conf)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None) -> None:
        self.boxes = boxes


class _FakeYoloModel:
    def __init__(self, weights: str) -> None:
        if weights == "raise_on_load":
            raise RuntimeError("bad weights")
        self.weights = weights
        self.names = {0: "glass_bead", 1: "intruder"}
        self.next_results: list[_FakeResult] = [
            _FakeResult(_FakeBoxes([[10.0, 10.0, 20.0, 20.0]], [0.9], [0.0]))
        ]

    def predict(self, **kwargs: Any) -> list[_FakeResult]:
        if kwargs.get("source") is None:
            raise RuntimeError("no source")
        if isinstance(kwargs.get("source"), str) and kwargs["source"] == "raise_on_predict":
            raise RuntimeError("inference exploded")
        return self.next_results


class _RaisingPredictModel(_FakeYoloModel):
    def predict(self, **kwargs: Any) -> list[_FakeResult]:
        raise RuntimeError("inference exploded")


class _FakeUltralyticsModule:
    YOLO = _FakeYoloModel


class _RaisingUltralyticsModule:
    @staticmethod
    def YOLO(weights: str) -> Any:  # noqa: N802 -- matches ultralytics's own API
        raise RuntimeError("cannot load")


def _make_dataset(tmp_path: Path, frame_count: int = 3) -> Path:
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


class TestYoloDetection:
    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            YoloDetection(x=1, y=1, radius=1, area=1, label="a", confidence=1.5)

    def test_is_a_detection(self) -> None:
        from glas.analysis.tracking_utils import Detection

        detection = YoloDetection(x=1, y=1, radius=1, area=1, label="a", confidence=0.5)
        assert isinstance(detection, Detection)


class TestYoloParticleDetectorConstruction:
    def test_rejects_bad_confidence_threshold(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            YoloParticleDetector(
                "weights.pt", confidence_threshold=1.5, _ultralytics=_FakeUltralyticsModule()
            )

    def test_rejects_bad_iou_threshold(self) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            YoloParticleDetector(
                "weights.pt", iou_threshold=-0.1, _ultralytics=_FakeUltralyticsModule()
            )

    def test_raises_ai_dependency_error_when_ultralytics_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "ultralytics":
                raise ImportError("no ultralytics")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(AIDependencyError):
            YoloParticleDetector("weights.pt")

    def test_raises_ai_model_error_when_weights_fail_to_load(self) -> None:
        with pytest.raises(AIModelError):
            YoloParticleDetector("raise_on_load", _ultralytics=_FakeUltralyticsModule())

    def test_class_names_property(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        assert detector.class_names == {0: "glass_bead", 1: "intruder"}


class TestYoloParticleDetectorDetect:
    def test_returns_one_detection_per_box(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detections = detector.detect(np.zeros((64, 64), dtype=np.uint8))
        assert len(detections) == 1
        assert detections[0].label == "glass_bead"
        assert detections[0].confidence == pytest.approx(0.9)

    def test_computes_centroid_and_radius_from_box(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detections = detector.detect(np.zeros((64, 64), dtype=np.uint8))
        detection = detections[0]
        assert detection.x == pytest.approx(15.0)
        assert detection.y == pytest.approx(15.0)
        assert detection.area == pytest.approx(100.0)

    def test_flags_intruder_label_case_insensitively(self) -> None:
        module = _FakeUltralyticsModule()
        detector = YoloParticleDetector(
            "weights.pt", intruder_label="Intruder", _ultralytics=module
        )
        detector._model.next_results = [
            _FakeResult(_FakeBoxes([[0.0, 0.0, 10.0, 10.0]], [0.8], [1.0]))
        ]
        detections = detector.detect(np.zeros((64, 64), dtype=np.uint8))
        assert detections[0].is_intruder is True

    def test_non_intruder_label_not_flagged(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detections = detector.detect(np.zeros((64, 64), dtype=np.uint8))
        assert detections[0].is_intruder is False

    def test_no_results_returns_empty_list(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detector._model.next_results = []
        assert detector.detect(np.zeros((64, 64), dtype=np.uint8)) == []

    def test_no_boxes_returns_empty_list(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detector._model.next_results = [_FakeResult(None)]
        assert detector.detect(np.zeros((64, 64), dtype=np.uint8)) == []

    def test_empty_boxes_returns_empty_list(self) -> None:
        detector = YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule())
        detector._model.next_results = [_FakeResult(_FakeBoxes([], [], []))]
        assert detector.detect(np.zeros((64, 64), dtype=np.uint8)) == []

    def test_raises_ai_model_error_on_inference_failure(self) -> None:
        module = _FakeUltralyticsModule()
        module.YOLO = _RaisingPredictModel  # type: ignore[assignment]
        detector = YoloParticleDetector("weights.pt", _ultralytics=module)
        with pytest.raises(AIModelError):
            detector.detect(np.zeros((64, 64), dtype=np.uint8))


class TestTrackDatasetYolo:
    def test_tracks_across_frames(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=3)
        history = track_dataset_yolo(
            folder,
            "weights.pt",
            _detector=YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule()),
        )
        assert len(history) == 1
        observations = next(iter(history.values()))
        assert len(observations) == 3
        assert all(obs.label == "glass_bead" for obs in observations)
        assert all(obs.confidence == pytest.approx(0.9) for obs in observations)

    def test_constructs_its_own_detector_when_none_injected(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=1)
        history = track_dataset_yolo(
            folder,
            "weights.pt",
            _detector=YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule()),
        )
        assert len(history) == 1

    def test_empty_dataset_yields_empty_history(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=0)
        history = track_dataset_yolo(
            folder,
            "weights.pt",
            _detector=YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule()),
        )
        assert history == {}

    def test_is_a_dropin_for_classical_track_dataset_shape(self, tmp_path: Path) -> None:
        folder = _make_dataset(tmp_path, frame_count=2)
        history = track_dataset_yolo(
            folder,
            "weights.pt",
            _detector=YoloParticleDetector("weights.pt", _ultralytics=_FakeUltralyticsModule()),
        )
        for observations in history.values():
            for obs in observations:
                assert hasattr(obs, "track_id")
                assert hasattr(obs, "frame_id")
                assert hasattr(obs, "label")

"""Tests for glas.ai.yolo_train."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from glas.ai.yolo_train import (
    YoloTrainingConfig,
    export_yolo_model,
    train_yolo,
    validate_yolo,
)
from glas.exceptions import AIModelError


class _FakeMetrics:
    def __init__(self, results_dict: dict[str, float] | None) -> None:
        self.results_dict = results_dict


class _FakeTrainer:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.best = save_dir / "weights" / "best.pt"
        self.last = save_dir / "weights" / "last.pt"
        self.metrics = _FakeMetrics({"metrics/mAP50(B)": 0.75, "fitness": 0.8})


class _FakeYoloModel:
    def __init__(self, weights: str, *, tmp_path: Path | None = None) -> None:
        self.weights = weights
        self.trainer: _FakeTrainer | None = None
        self._tmp_path = tmp_path

    def train(self, **kwargs: Any) -> None:
        if kwargs.get("data") == "raise_on_train":
            raise RuntimeError("training exploded")
        save_dir = (
            Path(kwargs["project"]) / kwargs["name"]
            if kwargs.get("project")
            else Path("runs/detect/glas_yolo")
        )
        self.trainer = _FakeTrainer(save_dir)

    def val(self, **kwargs: Any) -> _FakeMetrics:
        if kwargs.get("data") == "raise_on_val":
            raise RuntimeError("val exploded")
        return _FakeMetrics({"metrics/mAP50(B)": 0.6})

    def export(self, **kwargs: Any) -> str:
        if kwargs.get("format") == "raise_on_export":
            raise RuntimeError("export exploded")
        return f"exported.{kwargs['format']}"


class _FakeUltralyticsModule:
    def YOLO(self, weights: str) -> _FakeYoloModel:  # noqa: N802 -- matches ultralytics API
        return _FakeYoloModel(weights)


class TestYoloTrainingConfig:
    def test_rejects_non_positive_epochs(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            YoloTrainingConfig(data_yaml=tmp_path / "data.yaml", epochs=0)

    def test_rejects_negative_patience(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):  # noqa: B017
            YoloTrainingConfig(data_yaml=tmp_path / "data.yaml", patience=-1)

    def test_defaults(self, tmp_path: Path) -> None:
        config = YoloTrainingConfig(data_yaml=tmp_path / "data.yaml")
        assert config.base_weights == "yolo11n.pt"
        assert config.epochs == 100
        assert config.resume is False


class TestTrainYolo:
    def test_returns_result_with_weights_and_metrics(self, tmp_path: Path) -> None:
        config = YoloTrainingConfig(
            data_yaml=tmp_path / "data.yaml", project=tmp_path / "runs", name="run1", epochs=1
        )
        result = train_yolo(config, _ultralytics=_FakeUltralyticsModule())
        assert result.best_weights == tmp_path / "runs" / "run1" / "weights" / "best.pt"
        assert result.last_weights == tmp_path / "runs" / "run1" / "weights" / "last.pt"
        assert result.metrics["metrics/mAP50(B)"] == pytest.approx(0.75)

    def test_raises_ai_model_error_on_training_failure(self, tmp_path: Path) -> None:
        config = YoloTrainingConfig(data_yaml=Path("raise_on_train"), epochs=1)
        with pytest.raises(AIModelError):
            train_yolo(config, _ultralytics=_FakeUltralyticsModule())

    def test_missing_metrics_yields_empty_dict(self, tmp_path: Path) -> None:
        class _NoMetricsModule:
            def YOLO(self, weights: str) -> Any:  # noqa: N802 -- matches ultralytics's own API
                model = _FakeYoloModel(weights)

                def train(**kwargs: Any) -> None:
                    model.trainer = _FakeTrainer(tmp_path / "runs" / "run2")
                    model.trainer.metrics = _FakeMetrics(None)

                model.train = train  # type: ignore[method-assign]
                return model

        config = YoloTrainingConfig(data_yaml=tmp_path / "data.yaml", epochs=1)
        result = train_yolo(config, _ultralytics=_NoMetricsModule())
        assert result.metrics == {}


class TestValidateYolo:
    def test_returns_metrics(self, tmp_path: Path) -> None:
        metrics = validate_yolo(
            "weights.pt", tmp_path / "data.yaml", _ultralytics=_FakeUltralyticsModule()
        )
        assert metrics["metrics/mAP50(B)"] == pytest.approx(0.6)

    def test_raises_ai_model_error_on_failure(self, tmp_path: Path) -> None:
        with pytest.raises(AIModelError):
            validate_yolo("weights.pt", Path("raise_on_val"), _ultralytics=_FakeUltralyticsModule())


class TestExportYoloModel:
    def test_returns_exported_path(self) -> None:
        path = export_yolo_model("weights.pt", "onnx", _ultralytics=_FakeUltralyticsModule())
        assert path == Path("exported.onnx")

    def test_raises_ai_model_error_on_failure(self) -> None:
        with pytest.raises(AIModelError):
            export_yolo_model(
                "weights.pt", "raise_on_export", _ultralytics=_FakeUltralyticsModule()
            )

    def test_default_format_is_onnx(self) -> None:
        path = export_yolo_model("weights.pt", _ultralytics=_FakeUltralyticsModule())
        assert path == Path("exported.onnx")

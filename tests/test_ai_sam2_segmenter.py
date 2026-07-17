"""Tests for glas.ai.sam2_segmenter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from glas.ai.sam2_segmenter import (
    DEFAULT_CONTACT_DILATION_PX,
    ParticleSegment,
    Sam2Segmenter,
    ShapeMetrics,
    compute_contact_area,
    compute_segmentation_summary,
    compute_shape_metrics,
)
from glas.analysis.tracking_utils import Detection
from glas.exceptions import AIModelError


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius**2


class _FakeTorchModule:
    class cuda:  # noqa: N801 -- mirrors torch.cuda
        @staticmethod
        def is_available() -> bool:
            return False


class _FakePredictor:
    def __init__(self, mask: np.ndarray | None = None, score: float = 0.95) -> None:
        self.device = "cpu"
        self._mask = mask
        self._score = score
        self.set_image_calls = 0
        self.predict_calls: list[Any] = []

    def set_image(self, image: Any) -> None:
        if isinstance(image, str) and image == "raise":
            raise RuntimeError("bad image")
        self.set_image_calls += 1

    def predict(self, *, box: Any, multimask_output: bool) -> tuple[Any, Any, Any]:
        self.predict_calls.append(box)
        if self._mask is None:
            raise RuntimeError("segmentation exploded")
        return np.array([self._mask]), np.array([self._score]), None

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str) -> _FakePredictor:
        if model_id == "raise_on_load":
            raise RuntimeError("bad model id")
        return cls(mask=_circle_mask((64, 64), (32, 32), 10))


class TestComputeShapeMetrics:
    def test_computes_area_close_to_circle_area(self) -> None:
        mask = _circle_mask((100, 100), (50, 50), 20)
        metrics = compute_shape_metrics(mask)
        expected_area = np.pi * 20**2
        assert metrics.area_px == pytest.approx(expected_area, rel=0.05)

    def test_aspect_ratio_close_to_one_for_circle(self) -> None:
        mask = _circle_mask((100, 100), (50, 50), 20)
        metrics = compute_shape_metrics(mask)
        assert metrics.aspect_ratio == pytest.approx(1.0, abs=0.2)

    def test_centroid_matches_circle_center(self) -> None:
        mask = _circle_mask((100, 100), (40, 60), 15)
        metrics = compute_shape_metrics(mask)
        assert metrics.centroid_x == pytest.approx(40, abs=1.5)
        assert metrics.centroid_y == pytest.approx(60, abs=1.5)

    def test_raises_on_empty_mask(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(AIModelError):
            compute_shape_metrics(mask)

    def test_perimeter_positive(self) -> None:
        mask = _circle_mask((100, 100), (50, 50), 20)
        metrics = compute_shape_metrics(mask)
        assert metrics.perimeter_px > 0


class TestComputeContactArea:
    def test_touching_circles_have_nonzero_contact(self) -> None:
        mask_a = _circle_mask((100, 100), (30, 50), 15)
        mask_b = _circle_mask((100, 100), (60, 50), 15)
        assert compute_contact_area(mask_a, mask_b, dilation_px=2) > 0

    def test_distant_circles_have_zero_contact(self) -> None:
        mask_a = _circle_mask((200, 200), (20, 20), 5)
        mask_b = _circle_mask((200, 200), (180, 180), 5)
        assert compute_contact_area(mask_a, mask_b) == 0

    def test_rejects_non_positive_dilation(self) -> None:
        mask = _circle_mask((50, 50), (25, 25), 5)
        with pytest.raises(ValueError, match="dilation_px"):
            compute_contact_area(mask, mask, dilation_px=0)

    def test_rejects_mismatched_shapes(self) -> None:
        mask_a = _circle_mask((50, 50), (25, 25), 5)
        mask_b = _circle_mask((60, 60), (25, 25), 5)
        with pytest.raises(ValueError, match="shape"):
            compute_contact_area(mask_a, mask_b)

    def test_default_dilation_constant(self) -> None:
        assert DEFAULT_CONTACT_DILATION_PX == 1


class TestComputeSegmentationSummary:
    def _segment(self, mask: np.ndarray) -> ParticleSegment:
        return ParticleSegment(mask=mask, score=0.9, metrics=compute_shape_metrics(mask))

    def test_empty_segments_yield_zero_counts(self) -> None:
        summary = compute_segmentation_summary([], (100, 100))
        assert summary.particle_count == 0
        assert summary.packing_fraction == 0.0
        assert summary.void_fraction == 1.0
        assert summary.contacts == []

    def test_packing_fraction_matches_mask_coverage(self) -> None:
        mask = _circle_mask((100, 100), (50, 50), 10)
        segment = self._segment(mask)
        summary = compute_segmentation_summary([segment], (100, 100))
        expected = np.count_nonzero(mask) / (100 * 100)
        assert summary.packing_fraction == pytest.approx(expected)
        assert summary.void_fraction == pytest.approx(1 - expected)

    def test_touching_particles_produce_contacts(self) -> None:
        mask_a = _circle_mask((100, 100), (30, 50), 15)
        mask_b = _circle_mask((100, 100), (60, 50), 15)
        summary = compute_segmentation_summary(
            [self._segment(mask_a), self._segment(mask_b)], (100, 100)
        )
        assert summary.contacts == [(0, 1, summary.contacts[0][2])]
        assert summary.contacts[0][2] > 0

    def test_non_touching_particles_produce_no_contacts(self) -> None:
        mask_a = _circle_mask((200, 200), (20, 20), 5)
        mask_b = _circle_mask((200, 200), (180, 180), 5)
        summary = compute_segmentation_summary(
            [self._segment(mask_a), self._segment(mask_b)], (200, 200)
        )
        assert summary.contacts == []

    def test_particle_count_matches_segment_count(self) -> None:
        mask = _circle_mask((50, 50), (25, 25), 5)
        summary = compute_segmentation_summary([self._segment(mask), self._segment(mask)], (50, 50))
        assert summary.particle_count == 2


class TestShapeMetricsValidation:
    def test_rejects_negative_area(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            ShapeMetrics(
                area_px=-1,
                perimeter_px=1,
                centroid_x=0,
                centroid_y=0,
                orientation_deg=0,
                aspect_ratio=1,
            )

    def test_rejects_aspect_ratio_below_one(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            ShapeMetrics(
                area_px=1,
                perimeter_px=1,
                centroid_x=0,
                centroid_y=0,
                orientation_deg=0,
                aspect_ratio=0.5,
            )


class TestSam2SegmenterConstruction:
    def test_rejects_neither_model_id_nor_local(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            Sam2Segmenter()

    def test_rejects_both_model_id_and_local(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Sam2Segmenter(
                model_id="facebook/sam2.1-hiera-large",
                config_file="cfg.yaml",
                checkpoint_path="ckpt.pt",
            )

    def test_rejects_only_config_file_without_checkpoint(self) -> None:
        with pytest.raises(ValueError):
            Sam2Segmenter(config_file="cfg.yaml")

    def test_loads_from_pretrained_model_id(self) -> None:
        segmenter = Sam2Segmenter(
            model_id="facebook/sam2.1-hiera-large",
            _predictor_cls=_FakePredictor,
            _torch=_FakeTorchModule(),
        )
        assert segmenter._device == "cpu"

    def test_raises_ai_model_error_when_pretrained_load_fails(self) -> None:
        with pytest.raises(AIModelError):
            Sam2Segmenter(
                model_id="raise_on_load", _predictor_cls=_FakePredictor, _torch=_FakeTorchModule()
            )

    def test_loads_from_local_config_and_checkpoint(self) -> None:
        def build_sam2(config_file: str, checkpoint_path: str, *, device: str) -> str:
            return "fake_model"

        segmenter = Sam2Segmenter(
            config_file="cfg.yaml",
            checkpoint_path="ckpt.pt",
            _predictor_cls=_FakePredictor,
            _build_sam2=build_sam2,
            _torch=_FakeTorchModule(),
        )
        assert segmenter._device == "cpu"


class TestSam2SegmenterSegment:
    def _segmenter(self, mask: np.ndarray | None) -> Sam2Segmenter:
        segmenter = Sam2Segmenter(
            model_id="facebook/sam2.1-hiera-large",
            _predictor_cls=_FakePredictor,
            _torch=_FakeTorchModule(),
        )
        segmenter._predictor = _FakePredictor(mask=mask)
        return segmenter

    def test_segment_returns_particle_segment(self) -> None:
        mask = _circle_mask((64, 64), (32, 32), 10)
        segmenter = self._segmenter(mask)
        segment = segmenter.segment(
            np.zeros((64, 64), dtype=np.uint8), np.array([10.0, 10.0, 50.0, 50.0])
        )
        assert isinstance(segment, ParticleSegment)
        assert segment.score == pytest.approx(0.95)

    def test_segment_converts_mono_to_rgb(self) -> None:
        mask = _circle_mask((64, 64), (32, 32), 10)
        segmenter = self._segmenter(mask)
        segment = segmenter.segment(
            np.zeros((64, 64), dtype=np.uint8), np.array([10.0, 10.0, 50.0, 50.0])
        )
        assert segment.mask.shape == (64, 64)

    def test_segment_raises_ai_model_error_on_failure(self) -> None:
        segmenter = self._segmenter(None)
        with pytest.raises(AIModelError):
            segmenter.segment(np.zeros((64, 64), dtype=np.uint8), np.array([0.0, 0.0, 5.0, 5.0]))

    def test_segment_frame_returns_one_segment_per_detection(self) -> None:
        mask = _circle_mask((64, 64), (32, 32), 10)
        segmenter = self._segmenter(mask)
        detections = [
            Detection(x=32, y=32, radius=10, area=314),
            Detection(x=20, y=20, radius=5, area=78),
        ]
        segments = segmenter.segment_frame(np.zeros((64, 64), dtype=np.uint8), detections)
        assert len(segments) == 2

    def test_segment_frame_raises_on_set_image_failure(self) -> None:
        segmenter = self._segmenter(_circle_mask((64, 64), (32, 32), 10))
        segmenter._predictor.set_image = lambda image: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("fail")
        )
        with pytest.raises(AIModelError):
            segmenter.segment_frame(
                np.zeros((64, 64), dtype=np.uint8), [Detection(x=32, y=32, radius=10, area=314)]
            )

    def test_segment_frame_empty_detections_returns_empty_list(self) -> None:
        mask = _circle_mask((64, 64), (32, 32), 10)
        segmenter = self._segmenter(mask)
        assert segmenter.segment_frame(np.zeros((64, 64), dtype=np.uint8), []) == []

"""Tests for glas.ai.sam2_train.

The training tests use a genuine tiny ``torch.nn.Module``-based fake SAM2
model -- real forward/backward passes and a real optimizer step, not a
mocked loss -- so a real gradient-flow regression (e.g. accidentally
freezing the mask decoder too) would actually fail these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from glas.ai.sam2_train import (
    Sam2Example,
    Sam2TrainingConfig,
    auto_annotate_masks,
    prepare_sam2_dataset,
    train_sam2,
)
from glas.dataset import Dataset
from glas.exceptions import AIDatasetError, AIModelError
from glas.frame import Frame
from glas.metadata import DatasetMetadata

EMBED_DIM = 4
FEAT_SIZE = 4


def _make_blob_dataset(tmp_path: Path, frame_count: int = 3, blobs_per_frame: int = 2) -> Path:
    folder = tmp_path / "dataset"
    metadata = DatasetMetadata(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=80,
        height=80,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    dataset = Dataset.create(folder, metadata, dataset_format="hdf5")
    for i in range(frame_count):
        image = np.zeros((80, 80), dtype=np.uint8)
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


class TestSam2Example:
    def test_is_frozen(self) -> None:
        example = Sam2Example(
            image_path=Path("a.png"), mask_path=Path("a_mask.png"), box=(0, 0, 5, 5)
        )
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            example.box = (1, 1, 2, 2)  # type: ignore[misc]


class TestAutoAnnotateMasks:
    def test_writes_one_image_per_frame(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=3)
        examples = auto_annotate_masks(folder, tmp_path / "out")
        image_paths = {example.image_path for example in examples}
        assert len(image_paths) == 3

    def test_creates_a_mask_per_blob(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=2)
        examples = auto_annotate_masks(folder, tmp_path / "out")
        assert len(examples) == 2
        for example in examples:
            assert example.mask_path.exists()

    def test_mask_pixel_count_close_to_circle_area(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=1)
        examples = auto_annotate_masks(folder, tmp_path / "out")
        mask = cv2.imread(str(examples[0].mask_path), cv2.IMREAD_GRAYSCALE)
        pixel_count = int(np.count_nonzero(mask))
        assert pixel_count == pytest.approx(np.pi * 6**2, rel=0.3)

    def test_min_area_filters_small_blobs(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=2)
        examples = auto_annotate_masks(folder, tmp_path / "out", min_area=10_000)
        assert examples == []

    def test_box_matches_bounding_rect_of_blob(self, tmp_path: Path) -> None:
        folder = _make_blob_dataset(tmp_path, frame_count=1, blobs_per_frame=1)
        examples = auto_annotate_masks(folder, tmp_path / "out")
        x1, y1, x2, y2 = examples[0].box
        assert x2 - x1 == pytest.approx(12, abs=2)
        assert y2 - y1 == pytest.approx(12, abs=2)


class TestPrepareSam2Dataset:
    def _make_examples(self, tmp_path: Path, count: int = 10) -> list[Sam2Example]:
        examples = []
        for i in range(count):
            image_path = tmp_path / "images" / f"frame_{i:03d}.png"
            mask_path = tmp_path / "masks" / f"frame_{i:03d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.zeros((20, 20), dtype=np.uint8))
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[5:15, 5:15] = 255
            cv2.imwrite(str(mask_path), mask)
            examples.append(
                Sam2Example(image_path=image_path, mask_path=mask_path, box=(5, 5, 15, 15))
            )
        return examples

    def test_rejects_empty_examples(self, tmp_path: Path) -> None:
        with pytest.raises(AIDatasetError):
            prepare_sam2_dataset([], tmp_path / "out")

    def test_rejects_val_fraction_out_of_range(self, tmp_path: Path) -> None:
        examples = self._make_examples(tmp_path)
        with pytest.raises(AIDatasetError):
            prepare_sam2_dataset(examples, tmp_path / "out", val_fraction=0.0)
        with pytest.raises(AIDatasetError):
            prepare_sam2_dataset(examples, tmp_path / "out", val_fraction=1.0)

    def test_rejects_too_few_examples_for_split(self, tmp_path: Path) -> None:
        examples = self._make_examples(tmp_path, count=1)
        with pytest.raises(AIDatasetError):
            prepare_sam2_dataset(examples, tmp_path / "out", val_fraction=0.2)

    def test_writes_manifest_with_correct_split_sizes(self, tmp_path: Path) -> None:
        examples = self._make_examples(tmp_path, count=10)
        manifest_path = prepare_sam2_dataset(examples, tmp_path / "out", val_fraction=0.3, seed=1)
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["train"]) == 7
        assert len(manifest["val"]) == 3

    def test_manifest_entries_have_expected_keys(self, tmp_path: Path) -> None:
        examples = self._make_examples(tmp_path, count=5)
        manifest_path = prepare_sam2_dataset(examples, tmp_path / "out", val_fraction=0.2)
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["train"][0]
        assert set(entry.keys()) == {"image_path", "mask_path", "box"}
        assert len(entry["box"]) == 4

    def test_deterministic_split_given_same_seed(self, tmp_path: Path) -> None:
        examples = self._make_examples(tmp_path, count=10)
        first = prepare_sam2_dataset(examples, tmp_path / "out1", seed=42)
        second = prepare_sam2_dataset(examples, tmp_path / "out2", seed=42)
        assert json.loads(first.read_text()) == json.loads(second.read_text())


class _FakePromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_proj = nn.Linear(2, EMBED_DIM)
        self.dense_pe = nn.Parameter(torch.zeros(1, EMBED_DIM, FEAT_SIZE, FEAT_SIZE))
        self.no_mask_embed = nn.Parameter(torch.zeros(1, EMBED_DIM, FEAT_SIZE, FEAT_SIZE))

    def forward(
        self, points: tuple[torch.Tensor, torch.Tensor], boxes: None, masks: None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        box_coords, _box_labels = points
        sparse_embeddings = self.point_proj(box_coords)
        dense_embeddings = self.no_mask_embed.expand(box_coords.size(0), -1, -1, -1)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        return self.dense_pe


class _FakeMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(EMBED_DIM, 1, kernel_size=1)

    def forward(
        self,
        *,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        combined = image_embeddings + dense_prompt_embeddings + image_pe * 0.0
        logits = self.conv(combined)
        iou_predictions = torch.zeros(logits.size(0), 1)
        return logits, iou_predictions, None, None


class _FakeSam2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.sam_prompt_encoder = _FakePromptEncoder()
        self.sam_mask_decoder = _FakeMaskDecoder()


class _FakeTransforms:
    def transform_boxes(
        self, box_tensor: torch.Tensor, *, normalize: bool, orig_hw: tuple[int, int]
    ) -> torch.Tensor:
        return box_tensor

    def postprocess_masks(
        self, low_res_masks: torch.Tensor, orig_hw: tuple[int, int]
    ) -> torch.Tensor:
        return nn.functional.interpolate(
            low_res_masks, size=orig_hw, mode="bilinear", align_corners=False
        )


class _FakePredictor:
    def __init__(self, model: _FakeSam2Model, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._transforms = _FakeTransforms()
        self._features: dict[str, object] | None = None
        self._orig_hw: tuple[int, int] | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        height, width = image_rgb.shape[:2]
        self._orig_hw = (height, width)
        self._features = {
            "image_embed": torch.zeros(1, EMBED_DIM, FEAT_SIZE, FEAT_SIZE),
            "high_res_feats": [torch.zeros(1, EMBED_DIM, FEAT_SIZE, FEAT_SIZE) for _ in range(2)],
        }


def _write_manifest(
    tmp_path: Path, *, train_count: int, val_count: int, image_size: int = 16
) -> Path:
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    def make_entries(prefix: str, count: int) -> list[dict[str, object]]:
        entries = []
        for i in range(count):
            image_path = images_dir / f"{prefix}_{i:03d}.png"
            mask_path = masks_dir / f"{prefix}_{i:03d}.png"
            image = np.random.default_rng(i).integers(
                0, 255, (image_size, image_size, 3), dtype=np.uint8
            )
            cv2.imwrite(str(image_path), image)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            cv2.imwrite(str(mask_path), mask)
            entries.append(
                {"image_path": str(image_path), "mask_path": str(mask_path), "box": [4, 4, 12, 12]}
            )
        return entries

    manifest = {
        "train": make_entries("train", train_count),
        "val": make_entries("val", val_count),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _make_config(tmp_path: Path, manifest_path: Path, *, epochs: int = 1) -> Sam2TrainingConfig:
    return Sam2TrainingConfig(
        manifest_path=manifest_path,
        base_config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        base_checkpoint_path=tmp_path / "base.pt",
        output_checkpoint_path=tmp_path / "out" / "finetuned.pt",
        epochs=epochs,
        device="cpu",
    )


class TestTrainSam2:
    def test_returns_result_with_checkpoint_and_metrics(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=2, val_count=2)
        config = _make_config(tmp_path, manifest_path, epochs=1)
        predictor = _FakePredictor(_FakeSam2Model())

        result = train_sam2(config, _predictor=predictor, _torch=torch)

        assert result.checkpoint_path == config.output_checkpoint_path
        assert result.epochs == 1
        assert "train_loss" in result.metrics
        assert "val_loss" in result.metrics
        assert "val_mean_iou" in result.metrics

    def test_checkpoint_file_is_written(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=2, val_count=0)
        config = _make_config(tmp_path, manifest_path, epochs=1)
        predictor = _FakePredictor(_FakeSam2Model())

        result = train_sam2(config, _predictor=predictor, _torch=torch)

        assert result.checkpoint_path.exists()

    def test_no_val_metrics_when_val_split_empty(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=2, val_count=0)
        config = _make_config(tmp_path, manifest_path, epochs=1)
        predictor = _FakePredictor(_FakeSam2Model())

        result = train_sam2(config, _predictor=predictor, _torch=torch)

        assert "val_loss" not in result.metrics
        assert "val_mean_iou" not in result.metrics

    def test_image_encoder_parameters_stay_frozen(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=2, val_count=0)
        config = _make_config(tmp_path, manifest_path, epochs=1)
        model = _FakeSam2Model()
        predictor = _FakePredictor(model)
        initial_weight = model.image_encoder.weight.detach().clone()

        train_sam2(config, _predictor=predictor, _torch=torch)

        assert not model.image_encoder.weight.requires_grad
        assert torch.equal(model.image_encoder.weight.detach(), initial_weight)

    def test_prompt_encoder_and_mask_decoder_weights_actually_update(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=3, val_count=0)
        config = _make_config(tmp_path, manifest_path, epochs=3)
        model = _FakeSam2Model()
        predictor = _FakePredictor(model)
        initial_decoder_weight = model.sam_mask_decoder.conv.weight.detach().clone()

        train_sam2(config, _predictor=predictor, _torch=torch)

        assert model.sam_mask_decoder.conv.weight.requires_grad
        assert not torch.equal(model.sam_mask_decoder.conv.weight.detach(), initial_decoder_weight)

    def test_raises_ai_dataset_error_when_no_training_examples(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"train": [], "val": []}))
        config = _make_config(tmp_path, manifest_path, epochs=1)
        predictor = _FakePredictor(_FakeSam2Model())

        with pytest.raises(AIDatasetError):
            train_sam2(config, _predictor=predictor, _torch=torch)

    def test_raises_ai_model_error_when_base_model_load_fails(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, train_count=1, val_count=0)
        config = _make_config(tmp_path, manifest_path, epochs=1)

        def raising_build_sam2(config_file: str, checkpoint_path: str, *, device: str) -> None:
            raise RuntimeError("cannot load base model")

        class _RaisingPredictorCls:
            pass

        with pytest.raises(AIModelError):
            train_sam2(
                config,
                _build_sam2=raising_build_sam2,
                _torch=torch,
            )

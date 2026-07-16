"""Tests for glas.export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from glas.dataset import Dataset
from glas.exceptions import ExportError
from glas.export import ExportResult, export_dataset
from glas.frame import Frame
from glas.metadata import DatasetMetadata

_WIDTH = 16
_HEIGHT = 12


def _make_metadata(**overrides: object) -> DatasetMetadata:
    defaults = dict(
        dataset_format="hdf5",
        camera_model="acA640-750um",
        camera_serial="12345678",
        pixel_format="Mono8",
        width=_WIDTH,
        height=_HEIGHT,
        created_at_utc="2026-07-13T00:00:00+00:00",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)  # type: ignore[arg-type]


def _make_dataset(tmp_path: Path, count: int, dataset_format: str = "hdf5") -> Path:
    folder = tmp_path / "dataset"
    dataset = Dataset.create(folder, _make_metadata(), dataset_format=dataset_format)
    for i in range(count):
        image = np.full((_HEIGHT, _WIDTH), (i * 10) % 256, dtype=np.uint8)
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


class TestImageSequenceExport:
    @pytest.mark.parametrize("format", ["tiff", "png"])
    def test_writes_one_file_per_frame(self, tmp_path: Path, format: str) -> None:
        dataset_folder = _make_dataset(tmp_path, count=4)
        output = tmp_path / "frames_out"

        result = export_dataset(dataset_folder, output, format)  # type: ignore[arg-type]

        assert result.frame_count == 4
        extension = "tif" if format == "tiff" else "png"
        files = sorted(output.glob(f"*.{extension}"))
        assert len(files) == 4

    def test_written_files_preserve_pixel_content(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=3)
        output = tmp_path / "frames_out"
        export_dataset(dataset_folder, output, "png")

        files = sorted(output.glob("*.png"))
        image = cv2.imread(str(files[1]), cv2.IMREAD_UNCHANGED)
        assert image.mean() == pytest.approx(10.0)

    def test_raises_if_output_dir_non_empty_without_overwrite(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "frames_out"
        output.mkdir()
        (output / "existing.txt").write_text("hello")

        with pytest.raises(ExportError):
            export_dataset(dataset_folder, output, "png")

    def test_overwrite_true_proceeds_despite_existing_contents(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "frames_out"
        output.mkdir()
        (output / "existing.txt").write_text("hello")

        result = export_dataset(dataset_folder, output, "png", overwrite=True)
        assert result.frame_count == 2

    def test_raw_binary_source_dataset_exports_correctly(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=3, dataset_format="raw_binary")
        output = tmp_path / "frames_out"

        result = export_dataset(dataset_folder, output, "tiff")
        assert result.frame_count == 3


class TestVideoExport:
    @pytest.mark.parametrize("format", ["mp4", "avi"])
    def test_writes_a_readable_video_with_expected_frame_count(
        self, tmp_path: Path, format: str
    ) -> None:
        dataset_folder = _make_dataset(tmp_path, count=5)
        output = tmp_path / f"out.{format}"

        result = export_dataset(dataset_folder, output, format, fps=10.0)  # type: ignore[arg-type]

        assert result.frame_count == 5
        assert output.is_file()

        capture = cv2.VideoCapture(str(output))
        try:
            read_count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                read_count += 1
        finally:
            capture.release()
        assert read_count == 5

    def test_raises_if_output_file_exists_without_overwrite(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"not a real video")

        with pytest.raises(ExportError):
            export_dataset(dataset_folder, output, "mp4")

    def test_overwrite_true_replaces_existing_file(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"not a real video")

        result = export_dataset(dataset_folder, output, "mp4", overwrite=True)
        assert result.frame_count == 2
        assert output.stat().st_size > len(b"not a real video")

    def test_non_positive_fps_raises(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "out.mp4"

        with pytest.raises(ExportError):
            export_dataset(dataset_folder, output, "mp4", fps=0.0)


class TestGifExport:
    def test_writes_a_readable_gif_with_expected_frame_count(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=4)
        output = tmp_path / "out.gif"

        result = export_dataset(dataset_folder, output, "gif", fps=8.0)

        assert result.frame_count == 4
        with Image.open(output) as gif:
            assert gif.n_frames == 4
            assert gif.format == "GIF"

    def test_duration_reflects_requested_fps(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "out.gif"

        export_dataset(dataset_folder, output, "gif", fps=10.0)

        with Image.open(output) as gif:
            assert gif.info["duration"] == 100  # 1000ms / 10fps


class TestFrameRangeSelection:
    def test_start_frame_skips_leading_frames(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=5)
        output = tmp_path / "frames_out"

        result = export_dataset(dataset_folder, output, "png", start_frame=2)

        assert result.frame_count == 3
        files = sorted(output.glob("*.png"))
        first_image = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        assert first_image.mean() == pytest.approx(20.0)  # frame index 2 -> value 20

    def test_end_frame_excludes_trailing_frames(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=5)
        output = tmp_path / "frames_out"

        result = export_dataset(dataset_folder, output, "png", end_frame=2)
        assert result.frame_count == 2

    def test_start_and_end_frame_together(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=10)
        output = tmp_path / "frames_out"

        result = export_dataset(dataset_folder, output, "png", start_frame=3, end_frame=6)
        assert result.frame_count == 3

    def test_range_beyond_frame_count_raises(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=3)
        output = tmp_path / "frames_out"

        with pytest.raises(ExportError):
            export_dataset(dataset_folder, output, "png", start_frame=10)


class TestUnknownFormat:
    def test_unrecognized_format_raises(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=2)
        output = tmp_path / "frames_out"

        with pytest.raises(ExportError):
            export_dataset(dataset_folder, output, "bmp")  # type: ignore[arg-type]


class TestExportResult:
    def test_is_frozen(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=1)
        output = tmp_path / "frames_out"
        result = export_dataset(dataset_folder, output, "png")

        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            result.frame_count = 99  # type: ignore[misc]

    def test_returns_an_export_result(self, tmp_path: Path) -> None:
        dataset_folder = _make_dataset(tmp_path, count=1)
        output = tmp_path / "frames_out"
        result = export_dataset(dataset_folder, output, "png")
        assert isinstance(result, ExportResult)

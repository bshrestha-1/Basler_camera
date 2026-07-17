"""Tests for glas.calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glas.calibration import (
    SpatialCalibration,
    calibrate_from_checkerboard,
    calibrate_from_known_distance,
    load_calibration,
    save_calibration,
)
from glas.exceptions import CalibrationError


def _make_checkerboard(
    squares: tuple[int, int] = (8, 8), square_size_px: int = 40, margin_px: int = 40
) -> np.ndarray:
    cols, rows = squares
    board_w = cols * square_size_px
    board_h = rows * square_size_px
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[
                    r * square_size_px : (r + 1) * square_size_px,
                    c * square_size_px : (c + 1) * square_size_px,
                ] = 255
    image = np.full((board_h + 2 * margin_px, board_w + 2 * margin_px), 255, dtype=np.uint8)
    image[margin_px : margin_px + board_h, margin_px : margin_px + board_w] = board
    return image


class TestSpatialCalibration:
    def test_rejects_non_positive_mm_per_pixel(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            SpatialCalibration(mm_per_pixel=0, method="two_point", created_at_utc="2026-01-01")

    def test_px_to_mm(self) -> None:
        calibration = SpatialCalibration(
            mm_per_pixel=0.5, method="two_point", created_at_utc="2026-01-01"
        )
        assert calibration.px_to_mm(10) == pytest.approx(5.0)

    def test_mm_to_px(self) -> None:
        calibration = SpatialCalibration(
            mm_per_pixel=0.5, method="two_point", created_at_utc="2026-01-01"
        )
        assert calibration.mm_to_px(5.0) == pytest.approx(10.0)

    def test_px_to_mm_area(self) -> None:
        calibration = SpatialCalibration(
            mm_per_pixel=2.0, method="two_point", created_at_utc="2026-01-01"
        )
        assert calibration.px_to_mm_area(10) == pytest.approx(40.0)

    def test_roundtrip_px_mm_px(self) -> None:
        calibration = SpatialCalibration(
            mm_per_pixel=0.37, method="two_point", created_at_utc="2026-01-01"
        )
        assert calibration.mm_to_px(calibration.px_to_mm(123.0)) == pytest.approx(123.0)


class TestCalibrateFromKnownDistance:
    def test_computes_expected_scale(self) -> None:
        calibration = calibrate_from_known_distance((0, 0), (100, 0), distance_mm=50.0)
        assert calibration.mm_per_pixel == pytest.approx(0.5)
        assert calibration.method == "two_point"

    def test_diagonal_distance(self) -> None:
        calibration = calibrate_from_known_distance((0, 0), (3, 4), distance_mm=10.0)
        assert calibration.mm_per_pixel == pytest.approx(2.0)

    def test_rejects_non_positive_distance(self) -> None:
        with pytest.raises(CalibrationError, match="distance_mm"):
            calibrate_from_known_distance((0, 0), (10, 0), distance_mm=0.0)
        with pytest.raises(CalibrationError, match="distance_mm"):
            calibrate_from_known_distance((0, 0), (10, 0), distance_mm=-1.0)

    def test_rejects_coincident_points(self) -> None:
        with pytest.raises(CalibrationError, match="coincide"):
            calibrate_from_known_distance((5, 5), (5, 5), distance_mm=10.0)

    def test_sets_created_at_utc(self) -> None:
        calibration = calibrate_from_known_distance((0, 0), (10, 0), distance_mm=5.0)
        assert calibration.created_at_utc


class TestCalibrateFromCheckerboard:
    def test_computes_expected_scale(self) -> None:
        image = _make_checkerboard(square_size_px=40)
        calibration = calibrate_from_checkerboard(image, (7, 7), square_size_mm=25.0)
        assert calibration.mm_per_pixel == pytest.approx(25.0 / 40.0, rel=1e-3)
        assert calibration.method == "checkerboard"

    def test_different_square_size(self) -> None:
        image = _make_checkerboard(square_size_px=50)
        calibration = calibrate_from_checkerboard(image, (7, 7), square_size_mm=10.0)
        assert calibration.mm_per_pixel == pytest.approx(10.0 / 50.0, rel=1e-3)

    def test_accepts_color_image(self) -> None:
        mono = _make_checkerboard(square_size_px=40)
        color = np.stack([mono, mono, mono], axis=-1)
        calibration = calibrate_from_checkerboard(color, (7, 7), square_size_mm=25.0)
        assert calibration.mm_per_pixel == pytest.approx(25.0 / 40.0, rel=1e-3)

    def test_rejects_non_positive_square_size(self) -> None:
        image = _make_checkerboard()
        with pytest.raises(CalibrationError, match="square_size_mm"):
            calibrate_from_checkerboard(image, (7, 7), square_size_mm=0.0)

    def test_rejects_image_without_pattern(self) -> None:
        blank = np.full((200, 200), 128, dtype=np.uint8)
        with pytest.raises(CalibrationError, match="Could not find"):
            calibrate_from_checkerboard(blank, (7, 7), square_size_mm=10.0)

    def test_rejects_wrong_pattern_size(self) -> None:
        image = _make_checkerboard(squares=(8, 8), square_size_px=40)
        with pytest.raises(CalibrationError, match="Could not find"):
            calibrate_from_checkerboard(image, (20, 20), square_size_mm=10.0)


class TestSaveLoadCalibration:
    def test_roundtrip(self, tmp_path: Path) -> None:
        calibration = calibrate_from_known_distance((0, 0), (100, 0), distance_mm=50.0)
        path = tmp_path / "calibration.json"
        save_calibration(calibration, path)
        loaded = load_calibration(path)
        assert loaded == calibration

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        calibration = calibrate_from_known_distance((0, 0), (100, 0), distance_mm=50.0)
        path = tmp_path / "nested" / "dir" / "calibration.json"
        save_calibration(calibration, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="not found"):
            load_calibration(tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        with pytest.raises(CalibrationError, match="Could not read"):
            load_calibration(path)

    def test_load_wrong_schema_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_schema.json"
        path.write_text('{"foo": "bar"}')
        with pytest.raises(CalibrationError):
            load_calibration(path)

    def test_saved_file_is_pretty_printed_json(self, tmp_path: Path) -> None:
        calibration = calibrate_from_known_distance((0, 0), (100, 0), distance_mm=50.0)
        path = tmp_path / "calibration.json"
        save_calibration(calibration, path)
        text = path.read_text()
        assert "\n" in text
        assert text.endswith("\n")

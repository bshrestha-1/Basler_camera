"""Tests for glas.analysis.tracking_utils."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from glas.analysis.tracking_utils import Detection, detect_particles, link_nearest


def _make_detection(
    x: float, y: float, radius: float = 5.0, area: float | None = None
) -> Detection:
    return Detection(x=x, y=y, radius=radius, area=area if area is not None else radius**2)


class TestDetection:
    def test_is_frozen(self) -> None:
        detection = _make_detection(1.0, 2.0)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            detection.x = 5.0  # type: ignore[misc]

    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(ValueError):
            Detection(x=0.0, y=0.0, radius=-1.0, area=1.0)

    def test_rejects_negative_area(self) -> None:
        with pytest.raises(ValueError):
            Detection(x=0.0, y=0.0, radius=1.0, area=-1.0)


class TestDetectParticles:
    def test_empty_image_yields_no_detections(self) -> None:
        image = np.zeros((50, 50), dtype=np.uint8)
        assert detect_particles(image) == []

    def test_single_bright_circle_on_dark_background(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (40, 60), 6, 255, -1)

        detections = detect_particles(image)
        assert len(detections) == 1
        assert detections[0].x == pytest.approx(40, abs=1.0)
        assert detections[0].y == pytest.approx(60, abs=1.0)
        assert detections[0].radius == pytest.approx(6, abs=1.0)

    def test_single_dark_circle_on_bright_background_needs_invert(self) -> None:
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.circle(image, (40, 60), 6, 0, -1)

        detections = detect_particles(image, invert=True)
        assert len(detections) == 1
        assert detections[0].x == pytest.approx(40, abs=1.0)
        assert detections[0].y == pytest.approx(60, abs=1.0)

        # Without invert=True, the *bright background* is what's above
        # threshold, not the dark circle -- it's detected as one big blob
        # covering the frame, centered nowhere near the actual particle.
        without_invert = detect_particles(image, invert=False)
        assert len(without_invert) == 1
        assert without_invert[0].x != pytest.approx(40, abs=1.0)

    def test_detects_multiple_particles(self) -> None:
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (30, 30), 5, 255, -1)
        cv2.circle(image, (150, 150), 8, 255, -1)
        cv2.circle(image, (30, 150), 10, 255, -1)

        detections = detect_particles(image)
        assert len(detections) == 3

    def test_min_area_filters_small_blobs(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (20, 20), 2, 255, -1)  # tiny
        cv2.circle(image, (70, 70), 10, 255, -1)  # large

        detections = detect_particles(image, min_area=50.0)
        assert len(detections) == 1
        assert detections[0].x == pytest.approx(70, abs=1.0)

    def test_max_area_filters_large_blobs(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (20, 20), 2, 255, -1)  # tiny
        cv2.circle(image, (70, 70), 10, 255, -1)  # large

        detections = detect_particles(image, min_area=1.0, max_area=50.0)
        assert len(detections) == 1
        assert detections[0].x == pytest.approx(20, abs=1.0)

    def test_explicit_threshold_overrides_otsu(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 6, 128, -1)  # mid-brightness particle

        assert detect_particles(image, threshold=200) == []  # too bright a threshold
        detections = detect_particles(image, threshold=100)
        assert len(detections) == 1

    def test_16_bit_image_is_scaled_before_thresholding(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint16)
        cv2.circle(image, (40, 40), 6, 60000, -1)

        detections = detect_particles(image)
        assert len(detections) == 1
        assert detections[0].x == pytest.approx(40, abs=1.0)

    def test_returned_area_is_reported(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 10, 255, -1)

        detections = detect_particles(image)
        assert detections[0].area > 0


class TestLinkNearest:
    def test_matches_close_detections(self) -> None:
        previous = [_make_detection(10, 10)]
        current = [_make_detection(11, 11)]

        pairs = link_nearest(previous, current, max_distance=5.0)
        assert pairs == [(0, 0)]

    def test_no_match_beyond_max_distance(self) -> None:
        previous = [_make_detection(0, 0)]
        current = [_make_detection(100, 100)]

        pairs = link_nearest(previous, current, max_distance=5.0)
        assert pairs == []

    def test_empty_previous_yields_no_pairs(self) -> None:
        current = [_make_detection(0, 0)]
        assert link_nearest([], current, max_distance=5.0) == []

    def test_empty_current_yields_no_pairs(self) -> None:
        previous = [_make_detection(0, 0)]
        assert link_nearest(previous, [], max_distance=5.0) == []

    def test_each_index_matched_at_most_once(self) -> None:
        previous = [_make_detection(0, 0), _make_detection(100, 100)]
        current = [_make_detection(1, 1)]

        pairs = link_nearest(previous, current, max_distance=200.0)
        assert len(pairs) == 1
        assert pairs[0][0] == 0  # nearer previous detection wins

    def test_greedy_matching_prefers_globally_closest_pairs_first(self) -> None:
        # previous[0] is closest to current[1]; previous[1] is closest to current[0].
        # A correct greedy-by-distance match should pick both nearest pairs,
        # not fall back to index order.
        previous = [_make_detection(0, 0), _make_detection(10, 10)]
        current = [_make_detection(9, 9), _make_detection(1, 1)]

        pairs = link_nearest(previous, current, max_distance=50.0)
        assert set(pairs) == {(0, 1), (1, 0)}

    def test_result_is_sorted_by_previous_index(self) -> None:
        previous = [_make_detection(0, 0), _make_detection(50, 50)]
        current = [_make_detection(51, 51), _make_detection(1, 1)]

        pairs = link_nearest(previous, current, max_distance=10.0)
        assert pairs == sorted(pairs)

"""Tests for glas.frame."""

from __future__ import annotations

import numpy as np
import pytest

from glas.frame import Frame, pixel_format_dtype


def _make_frame(frame_id: int = 0, width: int = 64, height: int = 48) -> Frame:
    image = np.zeros((height, width), dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        image=image,
        pixel_format="Mono8",
        host_timestamp_ns=123,
        device_timestamp_ticks=456,
    )


def test_width_and_height_reflect_image_shape() -> None:
    frame = _make_frame(width=64, height=48)
    assert frame.width == 64
    assert frame.height == 48


def test_nbytes_matches_image_nbytes() -> None:
    frame = _make_frame(width=64, height=48)
    assert frame.nbytes == 64 * 48


def test_color_image_shape_is_preserved() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    frame = Frame(
        frame_id=0,
        image=image,
        pixel_format="RGB8Packed",
        host_timestamp_ns=0,
        device_timestamp_ticks=0,
    )
    assert frame.height == 10
    assert frame.width == 20


def test_frame_fields_are_accessible() -> None:
    frame = _make_frame(frame_id=7)
    assert frame.frame_id == 7
    assert frame.pixel_format == "Mono8"
    assert frame.host_timestamp_ns == 123
    assert frame.device_timestamp_ticks == 456


def test_frame_equality_is_identity_based_not_content_based() -> None:
    # A frozen dataclass's auto-generated __eq__ would try `image == image`,
    # which returns an element-wise array rather than a bool -- eq=False
    # avoids that trap entirely.
    frame_a = _make_frame()
    frame_b = _make_frame()
    assert frame_a != frame_b
    assert frame_a == frame_a


def test_frame_image_is_independent_of_source_array() -> None:
    source = np.ones((4, 4), dtype=np.uint8)
    frame = Frame(
        frame_id=0,
        image=source.copy(),
        pixel_format="Mono8",
        host_timestamp_ns=0,
        device_timestamp_ticks=0,
    )
    source[:] = 0
    assert frame.image.sum() == 16


@pytest.mark.parametrize(
    ("pixel_format", "expected_dtype"),
    [
        ("Mono8", np.uint8),
        ("Mono10", np.uint16),
        ("Mono12", np.uint16),
        ("Mono16", np.uint16),
    ],
)
def test_pixel_format_dtype_resolves_known_mono_formats(
    pixel_format: str, expected_dtype: type
) -> None:
    assert pixel_format_dtype(pixel_format) == np.dtype(expected_dtype)


def test_pixel_format_dtype_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match="RGB8Packed"):
        pixel_format_dtype("RGB8Packed")

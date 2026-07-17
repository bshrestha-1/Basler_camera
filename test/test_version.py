"""Tests for glas.version."""

from __future__ import annotations

from glas.version import VERSION_INFO, __version__


def test_version_is_semver_string() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_version_info_matches_version_string() -> None:
    expected = tuple(int(part) for part in __version__.split("."))
    assert expected == VERSION_INFO


def test_version_info_is_three_ints() -> None:
    assert len(VERSION_INFO) == 3
    assert all(isinstance(part, int) for part in VERSION_INFO)

"""Tests for the ``glas trigger`` CLI commands.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

pypylon = pytest.importorskip("pypylon")

from glas.camera_info import detect_cameras  # noqa: E402
from glas.cli import app  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )

runner = CliRunner()


def test_status_reports_disabled_by_default() -> None:
    result = runner.invoke(app, ["trigger", "status"])
    assert result.exit_code == 0
    assert "disabled" in result.output


def test_enable_then_status_reports_enabled() -> None:
    enable_result = runner.invoke(
        app, ["trigger", "enable", "--source", "Line1", "--activation", "RisingEdge"]
    )
    assert enable_result.exit_code == 0
    assert "Hardware trigger enabled" in enable_result.output
    assert "source=Line1" in enable_result.output


def test_enable_then_disable() -> None:
    runner.invoke(app, ["trigger", "enable"])
    disable_result = runner.invoke(app, ["trigger", "disable"])
    assert disable_result.exit_code == 0
    assert "Hardware trigger disabled" in disable_result.output


def test_enable_unsupported_source_fails_cleanly() -> None:
    result = runner.invoke(app, ["trigger", "enable", "--source", "NotARealLine"])
    assert result.exit_code == 1
    assert "Could not enable hardware trigger" in result.output


def test_unknown_serial_fails_cleanly() -> None:
    result = runner.invoke(app, ["trigger", "status", "--serial", "not-a-real-serial"])
    assert result.exit_code == 1
    assert "Could not read trigger status" in result.output

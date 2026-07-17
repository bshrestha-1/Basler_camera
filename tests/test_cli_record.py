"""Tests for the ``glas record`` CLI command.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

pypylon = pytest.importorskip("pypylon")

from glas.camera_info import detect_cameras  # noqa: E402
from glas.cli import app  # noqa: E402
from glas.experiment import ExperimentManager  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )

runner = CliRunner()


def test_record_for_a_fixed_duration_creates_a_finalized_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["record", str(tmp_path), "--duration", "0.3"])

    assert result.exit_code == 0
    assert "Connected to" in result.output
    assert "Recorded" in result.output
    assert (tmp_path / "Run0001" / "metadata.json").is_file()


def test_record_carries_name_and_tags_into_metadata(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "record",
            str(tmp_path),
            "--duration",
            "0.2",
            "--name",
            "cli smoke test",
            "--tag",
            "foo",
            "--tag",
            "bar",
        ],
    )
    assert result.exit_code == 0

    manager = ExperimentManager(tmp_path)
    summary = manager.get_experiment("Run0001")
    assert summary.name == "cli smoke test"
    assert summary.tags == ["foo", "bar"]


def test_record_carries_notes(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["record", str(tmp_path), "--duration", "0.2", "--notes", "shaker at 60 Hz"]
    )
    assert result.exit_code == 0

    manager = ExperimentManager(tmp_path)
    summary = manager.get_experiment("Run0001")
    assert summary.notes == "shaker at 60 Hz"


def test_record_with_unknown_serial_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["record", str(tmp_path), "--duration", "0.2", "--serial", "no-such-serial"]
    )
    assert result.exit_code == 1
    assert "Could not connect" in result.output


def test_sequential_recordings_get_separate_run_folders(tmp_path: Path) -> None:
    first = runner.invoke(app, ["record", str(tmp_path), "--duration", "0.2"])
    second = runner.invoke(app, ["record", str(tmp_path), "--duration", "0.2"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / "Run0001").is_dir()
    assert (tmp_path / "Run0002").is_dir()

"""Tests for glas.cli."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from glas.cli import app
from glas.version import __version__

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_init_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    result = runner.invoke(app, ["config", "init", "--path", str(target)])
    assert result.exit_code == 0
    assert target.exists()


def test_config_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("existing: true")
    result = runner.invoke(app, ["config", "init", "--path", str(target)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_config_init_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("existing: true")
    result = runner.invoke(app, ["config", "init", "--path", str(target), "--force"])
    assert result.exit_code == 0
    assert "existing" not in target.read_text()


def test_config_validate_accepts_generated_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    runner.invoke(app, ["config", "init", "--path", str(target)])
    result = runner.invoke(app, ["config", "validate", str(target)])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_config_validate_rejects_bad_file(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text("logging:\n  level: NOT_A_LEVEL\n")
    result = runner.invoke(app, ["config", "validate", str(target)])
    assert result.exit_code == 1


def test_config_show_prints_settings(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    runner.invoke(app, ["config", "init", "--path", str(target)])
    result = runner.invoke(app, ["config", "show", "--path", str(target)])
    assert result.exit_code == 0
    assert "log_level" in result.output

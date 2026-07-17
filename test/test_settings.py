"""Tests for glas.settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from glas.exceptions import JSONValidationError, SettingsError
from glas.settings import DEFAULT_CONFIG, Settings


def test_settings_load_uses_defaults_without_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GLAS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings.load()
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path(DEFAULT_CONFIG["paths"]["data_dir"]).expanduser()


def test_settings_load_reads_explicit_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "paths": {"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path / "logs")},
                "logging": {
                    "level": "DEBUG",
                    "file": "glas.log",
                    "max_bytes": 2048,
                    "backup_count": 2,
                    "console": False,
                },
            }
        )
    )
    settings = Settings.load(config_path=config_file)
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == tmp_path / "data"
    assert settings.log_console is False


def test_settings_load_rejects_invalid_level(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"logging": {"level": "VERBOSE"}}))
    with pytest.raises(JSONValidationError):
        Settings.load(config_path=config_file)


def test_settings_from_dict_missing_key_raises() -> None:
    with pytest.raises(JSONValidationError):
        Settings.from_dict({"paths": {"data_dir": "x"}})


def test_settings_from_dict_rejects_empty_string_fields() -> None:
    data = {
        "paths": {"data_dir": "", "log_dir": "y"},
        "logging": {
            "level": "INFO",
            "file": "glas.log",
            "max_bytes": 2048,
            "backup_count": 1,
            "console": True,
        },
    }
    with pytest.raises(JSONValidationError) as exc_info:
        Settings.from_dict(data)
    assert exc_info.value.errors


def test_settings_ensure_directories_creates_paths(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        log_level="INFO",
        log_file="glas.log",
        log_max_bytes=1024,
        log_backup_count=1,
        log_console=True,
    )
    settings.ensure_directories()
    assert settings.data_dir.is_dir()
    assert settings.log_dir.is_dir()


def test_settings_ensure_directories_wraps_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        log_level="INFO",
        log_file="glas.log",
        log_max_bytes=1024,
        log_backup_count=1,
        log_console=True,
    )

    def _raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "mkdir", _raise_os_error)
    with pytest.raises(SettingsError):
        settings.ensure_directories()


def test_default_config_is_valid() -> None:
    settings = Settings.from_dict(DEFAULT_CONFIG)
    assert settings.log_level == "INFO"

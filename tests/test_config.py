"""Tests for glas.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from glas.config import (
    deep_merge,
    find_config_file,
    load_config,
    read_yaml_file,
)
from glas.exceptions import ConfigurationError


def test_read_yaml_file_parses_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("a: 1\nb:\n  c: 2\n")
    assert read_yaml_file(config_file) == {"a": 1, "b": {"c": 2}}


def test_read_yaml_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        read_yaml_file(tmp_path / "missing.yaml")


def test_read_yaml_file_empty_returns_empty_dict(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    assert read_yaml_file(config_file) == {}


def test_read_yaml_file_invalid_yaml_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("a: [1, 2\n")
    with pytest.raises(ConfigurationError):
        read_yaml_file(config_file)


def test_read_yaml_file_non_mapping_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigurationError):
        read_yaml_file(config_file)


def test_deep_merge_overrides_scalars() -> None:
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_deep_merge_merges_nested_dicts() -> None:
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    override = {"a": {"x": 2}}
    deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"x": 2}}


def test_find_config_file_prefers_explicit_path(tmp_path: Path) -> None:
    config_file = tmp_path / "explicit.yaml"
    config_file.write_text("a: 1")
    result = find_config_file(explicit_path=config_file)
    assert result == config_file


def test_find_config_file_explicit_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        find_config_file(explicit_path=tmp_path / "missing.yaml")


def test_find_config_file_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "from_env.yaml"
    config_file.write_text("a: 1")
    monkeypatch.setenv("GLAS_CONFIG", str(config_file))
    assert find_config_file() == config_file


def test_find_config_file_falls_back_to_search_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "search" / "config.yaml"
    config_file.parent.mkdir()
    config_file.write_text("a: 1")
    result = find_config_file(search_paths=[tmp_path / "nope.yaml", config_file])
    assert result == config_file


def test_find_config_file_returns_none_when_nothing_found(tmp_path: Path) -> None:
    assert find_config_file(search_paths=[tmp_path / "nope.yaml"]) is None


def test_load_config_returns_defaults_when_no_file_found() -> None:
    defaults = {"a": 1}
    assert load_config(defaults=defaults) == defaults


def test_load_config_merges_file_over_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("a: 2\nb: 3\n")
    result = load_config(defaults={"a": 1}, explicit_path=config_file)
    assert result == {"a": 2, "b": 3}

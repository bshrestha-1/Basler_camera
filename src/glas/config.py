"""Generic configuration file loading, merging, and JSON Schema validation.

This module is domain-agnostic: it knows how to locate, read, merge, and
validate YAML configuration files, but has no knowledge of what GLAS's
specific settings should look like. See :mod:`glas.settings` for the
GLAS-specific schema and the typed :class:`~glas.settings.Settings` object
built on top of this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from glas.exceptions import ConfigurationError, JSONValidationError


def read_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file into a dictionary.

    Parameters
    ----------
    path : pathlib.Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content. An empty file yields an empty dict.

    Raises
    ------
    ConfigurationError
        If the file does not exist, cannot be read, is not valid YAML, or
        does not parse to a mapping at the top level.
    """
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file {path} must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}."
        )
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings, with ``override`` taking precedence.

    Parameters
    ----------
    base : Mapping
        The base mapping (e.g. defaults).
    override : Mapping
        Values that take precedence over ``base``. Nested dicts are merged
        recursively rather than replaced wholesale.

    Returns
    -------
    dict
        A new dictionary; neither input is mutated.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_json(instance: Any, schema: Mapping[str, Any]) -> None:
    """Validate ``instance`` against a JSON Schema.

    Parameters
    ----------
    instance : Any
        The data to validate (typically a dict loaded from YAML or JSON).
    schema : Mapping
        A JSON Schema (draft 2020-12) describing the expected structure.

    Raises
    ------
    JSONValidationError
        If ``instance`` violates the schema. All violations are collected
        into :attr:`~glas.exceptions.JSONValidationError.errors`.
    """
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: [str(p) for p in e.path])
    if errors:
        messages = [
            f"{'.'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
            for error in errors
        ]
        raise JSONValidationError(
            f"Configuration failed schema validation with {len(messages)} error(s).",
            errors=messages,
        )


def find_config_file(
    explicit_path: Path | None = None,
    env_var: str = "GLAS_CONFIG",
    search_paths: Sequence[Path] = (),
) -> Path | None:
    """Locate a configuration file using an explicit path, env var, or search list.

    Resolution order: ``explicit_path``, then the path in the ``env_var``
    environment variable (if set), then each path in ``search_paths`` in
    order. The first path that exists wins.

    Parameters
    ----------
    explicit_path : pathlib.Path, optional
        A path supplied directly by the caller (e.g. a ``--config`` CLI flag).
    env_var : str, default "GLAS_CONFIG"
        Name of the environment variable that may hold a config file path.
    search_paths : Sequence[pathlib.Path], default ()
        Additional candidate paths to check, in priority order.

    Returns
    -------
    pathlib.Path or None
        The first existing path found, or ``None`` if none exist.

    Raises
    ------
    ConfigurationError
        If ``explicit_path`` is given but does not exist.
    """
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {explicit_path}")
        return explicit_path

    env_value = os.environ.get(env_var)
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_file():
            raise ConfigurationError(f"Configuration file from ${env_var} not found: {env_path}")
        return env_path

    for candidate in search_paths:
        if candidate.is_file():
            return candidate

    return None


def load_config(
    defaults: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    explicit_path: Path | None = None,
    env_var: str = "GLAS_CONFIG",
    search_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Load configuration, merging file contents over ``defaults``.

    If no configuration file is found via :func:`find_config_file`, the
    defaults are returned unchanged. If a file is found, its contents are
    deep-merged over ``defaults`` and the merged result is validated
    against ``schema`` (if provided).

    Parameters
    ----------
    defaults : Mapping
        Default configuration values.
    schema : Mapping, optional
        JSON Schema to validate the merged configuration against.
    explicit_path : pathlib.Path, optional
        Explicit configuration file path, e.g. from a CLI flag.
    env_var : str, default "GLAS_CONFIG"
        Environment variable that may hold a configuration file path.
    search_paths : Sequence[pathlib.Path], default ()
        Fallback search locations, in priority order.

    Returns
    -------
    dict
        The merged (and validated) configuration.

    Raises
    ------
    ConfigurationError
        If a configuration file is specified or found but cannot be read.
    JSONValidationError
        If the merged configuration violates ``schema``.
    """
    config_path = find_config_file(
        explicit_path=explicit_path, env_var=env_var, search_paths=search_paths
    )

    merged = dict(defaults)
    if config_path is not None:
        file_data = read_yaml_file(config_path)
        merged = deep_merge(defaults, file_data)

    if schema is not None:
        validate_json(merged, schema)

    return merged

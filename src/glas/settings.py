"""GLAS-specific application settings.

Defines the default configuration, its JSON Schema, and a typed
:class:`Settings` object that the rest of GLAS depends on instead of
passing raw dictionaries around.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glas.config import load_config
from glas.exceptions import SettingsError

DEFAULT_DATA_DIR = Path.home() / "glas_data"

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_dir": str(DEFAULT_DATA_DIR),
        "log_dir": str(DEFAULT_DATA_DIR / "logs"),
    },
    "logging": {
        "level": "INFO",
        "file": "glas.log",
        "max_bytes": 10 * 1024 * 1024,
        "backup_count": 5,
        "console": True,
    },
}

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GLAS configuration",
    "type": "object",
    "required": ["paths", "logging"],
    "additionalProperties": True,
    "properties": {
        "paths": {
            "type": "object",
            "required": ["data_dir", "log_dir"],
            "properties": {
                "data_dir": {"type": "string", "minLength": 1},
                "log_dir": {"type": "string", "minLength": 1},
            },
        },
        "logging": {
            "type": "object",
            "required": ["level", "file", "max_bytes", "backup_count", "console"],
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                "file": {"type": "string", "minLength": 1},
                "max_bytes": {"type": "integer", "minimum": 1024},
                "backup_count": {"type": "integer", "minimum": 0},
                "console": {"type": "boolean"},
            },
        },
    },
}

DEFAULT_SEARCH_PATHS: tuple[Path, ...] = (
    Path.cwd() / "glas.yaml",
    Path.home() / ".config" / "glas" / "config.yaml",
)


@dataclass(frozen=True)
class Settings:
    """Typed, validated GLAS application settings.

    Attributes
    ----------
    data_dir : pathlib.Path
        Root directory for experiment data output.
    log_dir : pathlib.Path
        Directory where log files are written.
    log_level : str
        Root logging level, e.g. ``"INFO"``.
    log_file : str
        Log file name within ``log_dir``.
    log_max_bytes : int
        Maximum log file size in bytes before rotation.
    log_backup_count : int
        Number of rotated log files retained.
    log_console : bool
        Whether logging also writes to standard error.
    """

    data_dir: Path
    log_dir: Path
    log_level: str
    log_file: str
    log_max_bytes: int
    log_backup_count: int
    log_console: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Settings:
        """Build a :class:`Settings` instance from a validated config dict.

        Parameters
        ----------
        data : Mapping
            Configuration data matching :data:`CONFIG_SCHEMA`.

        Returns
        -------
        Settings

        Raises
        ------
        SettingsError
            If expected keys are missing from ``data``.
        """
        try:
            paths = data["paths"]
            logging_cfg = data["logging"]
            return cls(
                data_dir=Path(paths["data_dir"]).expanduser(),
                log_dir=Path(paths["log_dir"]).expanduser(),
                log_level=logging_cfg["level"],
                log_file=logging_cfg["file"],
                log_max_bytes=int(logging_cfg["max_bytes"]),
                log_backup_count=int(logging_cfg["backup_count"]),
                log_console=bool(logging_cfg["console"]),
            )
        except KeyError as exc:
            raise SettingsError(f"Missing required configuration key: {exc}") from exc

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """Load settings from a config file, falling back to defaults.

        Parameters
        ----------
        config_path : pathlib.Path, optional
            Explicit configuration file path. If ``None``, resolution
            follows :func:`glas.config.find_config_file` using the
            ``GLAS_CONFIG`` environment variable and
            :data:`DEFAULT_SEARCH_PATHS`.

        Returns
        -------
        Settings
        """
        merged = load_config(
            defaults=DEFAULT_CONFIG,
            schema=CONFIG_SCHEMA,
            explicit_path=config_path,
            search_paths=DEFAULT_SEARCH_PATHS,
        )
        return cls.from_dict(merged)

    def ensure_directories(self) -> None:
        """Create :attr:`data_dir` and :attr:`log_dir` if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

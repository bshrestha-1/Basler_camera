"""GLAS-specific application settings.

Defines the default configuration and a typed, Pydantic-validated
:class:`Settings` object that the rest of GLAS depends on instead of
passing raw dictionaries around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from glas.config import load_config
from glas.exceptions import JSONValidationError, SettingsError

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

DEFAULT_SEARCH_PATHS: tuple[Path, ...] = (
    Path.cwd() / "glas.yaml",
    Path.home() / ".config" / "glas" / "config.yaml",
)


class _PathsConfig(BaseModel):
    """The ``paths`` section of a GLAS configuration file."""

    model_config = ConfigDict(frozen=True)

    data_dir: str = Field(min_length=1)
    log_dir: str = Field(min_length=1)


class _LoggingConfig(BaseModel):
    """The ``logging`` section of a GLAS configuration file."""

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    file: str = Field(min_length=1)
    max_bytes: int = Field(ge=1024)
    backup_count: int = Field(ge=0)
    console: bool


class _RawConfig(BaseModel):
    """The full, nested shape of a GLAS configuration file, as validated on load."""

    model_config = ConfigDict(frozen=True, extra="allow")

    paths: _PathsConfig
    logging: _LoggingConfig


class Settings(BaseModel):
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

    model_config = ConfigDict(frozen=True)

    data_dir: Path
    log_dir: Path
    log_level: str
    log_file: str
    log_max_bytes: int
    log_backup_count: int
    log_console: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        """Build validated :class:`Settings` from a raw (nested) config dict.

        Parameters
        ----------
        data : dict
            Configuration data with ``paths`` and ``logging`` sections,
            as produced by merging :data:`DEFAULT_CONFIG` with a loaded
            configuration file.

        Returns
        -------
        Settings

        Raises
        ------
        JSONValidationError
            If ``data`` does not match the expected structure.
        """
        try:
            raw = _RawConfig.model_validate(data)
        except ValidationError as exc:
            raise JSONValidationError.from_pydantic(exc, context="Settings") from exc

        return cls(
            data_dir=Path(raw.paths.data_dir).expanduser(),
            log_dir=Path(raw.paths.log_dir).expanduser(),
            log_level=raw.logging.level,
            log_file=raw.logging.file,
            log_max_bytes=raw.logging.max_bytes,
            log_backup_count=raw.logging.backup_count,
            log_console=raw.logging.console,
        )

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
            explicit_path=config_path,
            search_paths=DEFAULT_SEARCH_PATHS,
        )
        return cls.from_dict(merged)

    def ensure_directories(self) -> None:
        """Create :attr:`data_dir` and :attr:`log_dir` if they do not exist.

        Raises
        ------
        SettingsError
            If either directory cannot be created (e.g. a permissions
            error).
        """
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SettingsError(f"Could not create settings directories: {exc}") from exc

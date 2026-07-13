"""GLAS: Granular Lab Acquisition System.

A production-quality acquisition and analysis platform built around a
Basler ace acA640-750um camera, for granular-material physics experiments.
"""

from glas.exceptions import (
    ConfigurationError,
    GLASError,
    JSONValidationError,
    LoggingError,
    SettingsError,
)
from glas.logger import configure_logging, get_logger
from glas.settings import Settings
from glas.version import VERSION_INFO, __version__

__all__ = [
    "__version__",
    "VERSION_INFO",
    "GLASError",
    "ConfigurationError",
    "JSONValidationError",
    "LoggingError",
    "SettingsError",
    "configure_logging",
    "get_logger",
    "Settings",
]

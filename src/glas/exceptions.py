"""Exception hierarchy shared across GLAS.

All exceptions raised by GLAS code inherit from :class:`GLASError`, so
calling code can catch every project-specific failure with a single
``except GLASError:`` clause while still being able to catch more specific
errors when needed.
"""

from __future__ import annotations


class GLASError(Exception):
    """Base class for all exceptions raised by GLAS."""


class ConfigurationError(GLASError):
    """Raised when a configuration file cannot be found, read, or parsed."""


class JSONValidationError(GLASError):
    """Raised when data fails JSON Schema validation.

    Parameters
    ----------
    message : str
        Human-readable summary of the validation failure.
    errors : list of str, optional
        Individual validation error messages, one per schema violation.

    Attributes
    ----------
    errors : list of str
        Individual validation error messages collected during validation.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors if errors is not None else []


class SettingsError(GLASError):
    """Raised when application settings cannot be constructed or are invalid."""


class LoggingError(GLASError):
    """Raised when the logging subsystem cannot be configured."""

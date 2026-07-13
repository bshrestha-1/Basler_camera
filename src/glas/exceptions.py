"""Exception hierarchy shared across GLAS.

All exceptions raised by GLAS code inherit from :class:`GLASError`, so
calling code can catch every project-specific failure with a single
``except GLASError:`` clause while still being able to catch more specific
errors when needed.
"""

from __future__ import annotations

from pydantic import ValidationError as _PydanticValidationError


class GLASError(Exception):
    """Base class for all exceptions raised by GLAS."""


class ConfigurationError(GLASError):
    """Raised when a configuration file cannot be found, read, or parsed."""


class JSONValidationError(GLASError):
    """Raised when structured data (typically loaded from YAML or JSON) fails validation.

    Parameters
    ----------
    message : str
        Human-readable summary of the validation failure.
    errors : list of str, optional
        Individual validation error messages, one per violation.

    Attributes
    ----------
    errors : list of str
        Individual validation error messages collected during validation.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors if errors is not None else []

    @classmethod
    def from_pydantic(cls, exc: _PydanticValidationError, context: str) -> JSONValidationError:
        """Build a :class:`JSONValidationError` from a Pydantic validation failure.

        Parameters
        ----------
        exc : pydantic.ValidationError
            The validation failure raised by a Pydantic model's
            ``model_validate()``.
        context : str
            What was being validated, e.g. ``"Settings"`` or
            ``"Dataset metadata"``, used in the summary message.

        Returns
        -------
        JSONValidationError
            With one entry in :attr:`errors` per Pydantic error, in
            ``field.path: message`` form.
        """
        errors = [
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        ]
        return cls(f"{context} failed validation with {len(errors)} error(s).", errors=errors)


class SettingsError(GLASError):
    """Raised when application settings cannot be constructed or are invalid."""


class LoggingError(GLASError):
    """Raised when the logging subsystem cannot be configured."""


class CameraError(GLASError):
    """Base class for all camera-related exceptions."""


class CameraDriverError(CameraError):
    """Raised when the pypylon / Basler Pylon driver is unavailable or fails.

    Covers both a missing ``pypylon`` installation and low-level GenICam
    transport-layer failures (e.g. the device enumeration call itself
    raising) that are not specific to a single camera.
    """


class CameraNotFoundError(CameraError):
    """Raised when no camera, or no camera matching a given serial number, is found."""


class CameraConnectionError(CameraError):
    """Raised when a camera cannot be opened, closed, or is used in the wrong state."""


class CameraConfigurationError(CameraError):
    """Raised when a proposed camera parameter value is invalid.

    Parameters
    ----------
    message : str
        Human-readable summary of the validation failure.
    errors : list of str, optional
        Individual violation messages, one per invalid field.

    Attributes
    ----------
    errors : list of str
        Individual violation messages collected during validation.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors if errors is not None else []


class CameraFeatureUnavailableError(CameraError):
    """Raised when a requested feature is not exposed by the connected device."""


class AcquisitionError(CameraError):
    """Raised when a frame grab fails outright, as opposed to an ordinary timeout."""


class DatasetError(GLASError):
    """Base class for dataset storage, metadata, and validation exceptions."""


class DatasetFormatError(DatasetError):
    """Raised when a requested dataset storage format is invalid or unavailable."""


class DatasetIOError(DatasetError):
    """Raised when writing or reading dataset files on disk fails."""


class WriterError(DatasetError):
    """Raised when the background dataset writer is used in an invalid state."""


class RecorderError(GLASError):
    """Raised when a recording session or its controller is used in an invalid state."""

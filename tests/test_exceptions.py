"""Tests for glas.exceptions."""

from __future__ import annotations

import pytest

from glas.exceptions import (
    AccelerometerError,
    BrazilNutError,
    CalibrationError,
    ConfigurationError,
    ConvectionError,
    DisplayError,
    ExperimentNotFoundError,
    ExportError,
    GLASError,
    HardwareError,
    InstrumentCommandError,
    InstrumentConnectionError,
    JSONValidationError,
    LoggingError,
    PackingError,
    ReportError,
    SegregationError,
    SettingsError,
)


@pytest.mark.parametrize(
    "exc_type",
    [
        ConfigurationError,
        JSONValidationError,
        SettingsError,
        LoggingError,
        DisplayError,
        ExportError,
        ExperimentNotFoundError,
        BrazilNutError,
        ConvectionError,
        PackingError,
        SegregationError,
        AccelerometerError,
        HardwareError,
        InstrumentConnectionError,
        InstrumentCommandError,
        CalibrationError,
        ReportError,
    ],
)
def test_all_exceptions_inherit_from_glaserror(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, GLASError)


def test_glaserror_is_an_exception() -> None:
    assert issubclass(GLASError, Exception)


def test_json_validation_error_stores_errors() -> None:
    err = JSONValidationError("bad config", errors=["a.b: too short"])
    assert err.errors == ["a.b: too short"]
    assert str(err) == "bad config"


def test_json_validation_error_defaults_to_empty_errors() -> None:
    err = JSONValidationError("bad config")
    assert err.errors == []


def test_exceptions_are_catchable_as_glaserror() -> None:
    with pytest.raises(GLASError):
        raise ConfigurationError("missing file")


def test_hardware_subexceptions_inherit_from_hardware_error() -> None:
    assert issubclass(InstrumentConnectionError, HardwareError)
    assert issubclass(InstrumentCommandError, HardwareError)

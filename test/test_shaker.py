"""Tests for glas.hardware.shaker."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from glas.hardware.shaker import ShakerCalibration, ShakerController
from glas.hardware.waveform_generator import SiglentSDG1032X


class FakeSCPITransport:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def write(self, command: str) -> None:
        self.sent.append(command)

    def query(self, command: str) -> str:
        self.sent.append(command)
        return "0"

    def close(self) -> None:
        pass


def _controller(volts_per_g: float = 0.5, frequency_hz: float = 60.0) -> ShakerController:
    generator = SiglentSDG1032X(FakeSCPITransport())
    calibration = ShakerCalibration(volts_per_g=volts_per_g, frequency_hz=frequency_hz)
    return ShakerController(generator, calibration)


class TestShakerCalibration:
    def test_is_frozen(self) -> None:
        calibration = ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError subtype
            calibration.volts_per_g = 1.0  # type: ignore[misc,unused-ignore]

    def test_rejects_non_positive_volts_per_g(self) -> None:
        with pytest.raises(ValidationError):
            ShakerCalibration(volts_per_g=0.0, frequency_hz=60.0)
        with pytest.raises(ValidationError):
            ShakerCalibration(volts_per_g=-1.0, frequency_hz=60.0)

    def test_rejects_non_positive_frequency(self) -> None:
        with pytest.raises(ValidationError):
            ShakerCalibration(volts_per_g=0.5, frequency_hz=0.0)


class TestRequiredDriveVoltage:
    def test_computes_exact_voltage(self) -> None:
        controller = _controller(volts_per_g=0.5)
        assert controller.required_drive_voltage(2.0) == pytest.approx(1.0)

    def test_scales_linearly_with_gamma(self) -> None:
        controller = _controller(volts_per_g=0.25)
        assert controller.required_drive_voltage(4.0) == pytest.approx(1.0)
        assert controller.required_drive_voltage(8.0) == pytest.approx(2.0)

    def test_rejects_non_positive_gamma(self) -> None:
        controller = _controller()
        with pytest.raises(ValueError):
            controller.required_drive_voltage(0.0)
        with pytest.raises(ValueError):
            controller.required_drive_voltage(-1.0)


class TestSetTargetGamma:
    def test_sends_calibration_frequency_by_default(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        calibration = ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        controller = ShakerController(generator, calibration)

        voltage = controller.set_target_gamma(2.0)

        assert voltage == pytest.approx(1.0)
        assert transport.sent == ["C1:BSWV WVTP,SINE,FRQ,60.0HZ,AMP,1.0V,OFST,0.0V,PHSE,0.0"]

    def test_explicit_frequency_overrides_calibration_frequency(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        calibration = ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        controller = ShakerController(generator, calibration)

        controller.set_target_gamma(1.0, frequency_hz=100.0)

        assert transport.sent == ["C1:BSWV WVTP,SINE,FRQ,100.0HZ,AMP,0.5V,OFST,0.0V,PHSE,0.0"]

    def test_uses_configured_channel(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        calibration = ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        controller = ShakerController(generator, calibration, channel=2)

        controller.set_target_gamma(2.0)

        assert transport.sent == ["C2:BSWV WVTP,SINE,FRQ,60.0HZ,AMP,1.0V,OFST,0.0V,PHSE,0.0"]

    def test_rejects_non_positive_gamma(self) -> None:
        controller = _controller()
        with pytest.raises(ValueError):
            controller.set_target_gamma(0.0)

    def test_rejects_non_positive_explicit_frequency(self) -> None:
        controller = _controller()
        with pytest.raises(ValueError):
            controller.set_target_gamma(1.0, frequency_hz=0.0)


class TestStartStop:
    def test_start_enables_output(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        controller = ShakerController(
            generator, ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        )
        controller.start()
        assert transport.sent == ["C1:OUTP ON"]

    def test_stop_disables_output(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        controller = ShakerController(
            generator, ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
        )
        controller.stop()
        assert transport.sent == ["C1:OUTP OFF"]

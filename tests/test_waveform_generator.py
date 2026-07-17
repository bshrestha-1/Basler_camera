"""Tests for glas.hardware.waveform_generator."""

from __future__ import annotations

import pytest

from glas.hardware.waveform_generator import SiglentSDG1032X


class FakeSCPITransport:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.sent: list[str] = []
        self._responses = list(responses) if responses is not None else []
        self.closed = False

    def write(self, command: str) -> None:
        self.sent.append(command)

    def query(self, command: str) -> str:
        self.sent.append(command)
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


class TestSetSineWave:
    def test_sends_expected_command(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)

        generator.set_sine_wave(1, frequency_hz=100.0, amplitude_vpp=2.0)

        assert transport.sent == ["C1:BSWV WVTP,SINE,FRQ,100.0HZ,AMP,2.0V,OFST,0.0V,PHSE,0.0"]

    def test_includes_offset_and_phase(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)

        generator.set_sine_wave(
            2, frequency_hz=50.0, amplitude_vpp=1.5, offset_v=0.25, phase_deg=90.0
        )

        assert transport.sent == ["C2:BSWV WVTP,SINE,FRQ,50.0HZ,AMP,1.5V,OFST,0.25V,PHSE,90.0"]

    def test_rejects_invalid_channel(self) -> None:
        generator = SiglentSDG1032X(FakeSCPITransport())
        with pytest.raises(ValueError, match="channel"):
            generator.set_sine_wave(3, frequency_hz=100.0, amplitude_vpp=2.0)
        with pytest.raises(ValueError, match="channel"):
            generator.set_sine_wave(0, frequency_hz=100.0, amplitude_vpp=2.0)

    def test_rejects_non_positive_frequency(self) -> None:
        generator = SiglentSDG1032X(FakeSCPITransport())
        with pytest.raises(ValueError):
            generator.set_sine_wave(1, frequency_hz=0.0, amplitude_vpp=2.0)
        with pytest.raises(ValueError):
            generator.set_sine_wave(1, frequency_hz=-10.0, amplitude_vpp=2.0)

    def test_rejects_non_positive_amplitude(self) -> None:
        generator = SiglentSDG1032X(FakeSCPITransport())
        with pytest.raises(ValueError):
            generator.set_sine_wave(1, frequency_hz=100.0, amplitude_vpp=0.0)
        with pytest.raises(ValueError):
            generator.set_sine_wave(1, frequency_hz=100.0, amplitude_vpp=-1.0)


class TestSetFrequency:
    def test_sends_expected_command(self) -> None:
        transport = FakeSCPITransport()
        SiglentSDG1032X(transport).set_frequency(1, 250.0)
        assert transport.sent == ["C1:BSWV FRQ,250.0HZ"]

    def test_rejects_invalid_channel(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).set_frequency(5, 100.0)

    def test_rejects_non_positive_frequency(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).set_frequency(1, 0.0)


class TestSetAmplitude:
    def test_sends_expected_command(self) -> None:
        transport = FakeSCPITransport()
        SiglentSDG1032X(transport).set_amplitude(2, 4.0)
        assert transport.sent == ["C2:BSWV AMP,4.0V"]

    def test_rejects_invalid_channel(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).set_amplitude(5, 1.0)

    def test_rejects_non_positive_amplitude(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).set_amplitude(1, -1.0)


class TestOutputControl:
    def test_enable_output_sends_expected_command(self) -> None:
        transport = FakeSCPITransport()
        SiglentSDG1032X(transport).enable_output(1)
        assert transport.sent == ["C1:OUTP ON"]

    def test_disable_output_sends_expected_command(self) -> None:
        transport = FakeSCPITransport()
        SiglentSDG1032X(transport).disable_output(2)
        assert transport.sent == ["C2:OUTP OFF"]

    def test_enable_output_rejects_invalid_channel(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).enable_output(0)

    def test_disable_output_rejects_invalid_channel(self) -> None:
        with pytest.raises(ValueError):
            SiglentSDG1032X(FakeSCPITransport()).disable_output(0)


class TestInheritedScpiBehavior:
    def test_identify_works_through_subclass(self) -> None:
        transport = FakeSCPITransport(["Siglent Technologies,SDG1032X,SN,1.0"])
        generator = SiglentSDG1032X(transport)
        assert generator.identify() == "Siglent Technologies,SDG1032X,SN,1.0"

    def test_close_closes_transport(self) -> None:
        transport = FakeSCPITransport()
        generator = SiglentSDG1032X(transport)
        generator.close()
        assert transport.closed is True

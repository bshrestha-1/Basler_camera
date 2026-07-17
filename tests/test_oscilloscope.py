"""Tests for glas.hardware.oscilloscope."""

from __future__ import annotations

import pytest

from glas.exceptions import InstrumentCommandError
from glas.hardware.oscilloscope import SCPIOscilloscope


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


class TestQueryFloat:
    def test_parses_numeric_response(self) -> None:
        transport = FakeSCPITransport(["3.14159"])
        scope = SCPIOscilloscope(transport)

        result = scope.query_float("C1:PAVA? PKPK")

        assert result == pytest.approx(3.14159)
        assert transport.sent == ["C1:PAVA? PKPK"]

    def test_parses_scientific_notation(self) -> None:
        transport = FakeSCPITransport(["1.5E-03"])
        result = SCPIOscilloscope(transport).query_float("C1:PAVA? MEAN")
        assert result == pytest.approx(1.5e-3)

    def test_parses_negative_values(self) -> None:
        transport = FakeSCPITransport(["-2.5"])
        result = SCPIOscilloscope(transport).query_float("C1:PAVA? MEAN")
        assert result == pytest.approx(-2.5)

    def test_non_numeric_response_raises(self) -> None:
        transport = FakeSCPITransport(["not-a-number"])
        with pytest.raises(InstrumentCommandError):
            SCPIOscilloscope(transport).query_float("C1:PAVA? PKPK")


class TestInheritedScpiBehavior:
    def test_identify_works_through_subclass(self) -> None:
        transport = FakeSCPITransport(["Generic Instruments,SCOPE1,SN,1.0"])
        scope = SCPIOscilloscope(transport)
        assert scope.identify() == "Generic Instruments,SCOPE1,SN,1.0"

    def test_raw_write_and_query_available(self) -> None:
        transport = FakeSCPITransport(["1"])
        scope = SCPIOscilloscope(transport)
        scope.write(":TRIG:MODE AUTO")
        response = scope.query(":TRIG:STATE?")
        assert transport.sent == [":TRIG:MODE AUTO", ":TRIG:STATE?"]
        assert response == "1"

    def test_close_closes_transport(self) -> None:
        transport = FakeSCPITransport()
        scope = SCPIOscilloscope(transport)
        scope.close()
        assert transport.closed is True

"""Tests for glas.hardware.scpi."""

from __future__ import annotations

import socket

import pytest

from glas.exceptions import InstrumentConnectionError
from glas.hardware.scpi import SCPIInstrument, SocketSCPITransport


class FakeSCPITransport:
    """An in-memory :class:`~glas.hardware.scpi.SCPITransport` for testing.

    Records every command sent and returns pre-programmed responses for
    :meth:`query`, in order -- no socket, no physical instrument.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self.sent: list[str] = []
        self._responses = list(responses) if responses is not None else []
        self.closed = False

    def write(self, command: str) -> None:
        self.sent.append(command)

    def query(self, command: str) -> str:
        self.sent.append(command)
        if not self._responses:
            raise AssertionError(f"No response programmed for query {command!r}.")
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


class TestSCPIInstrument:
    def test_identify_sends_idn_query(self) -> None:
        transport = FakeSCPITransport(["Siglent,SDG1032X,SN12345,1.01.01.33R3"])
        instrument = SCPIInstrument(transport)

        result = instrument.identify()

        assert transport.sent == ["*IDN?"]
        assert result == "Siglent,SDG1032X,SN12345,1.01.01.33R3"

    def test_reset_sends_rst(self) -> None:
        transport = FakeSCPITransport()
        SCPIInstrument(transport).reset()
        assert transport.sent == ["*RST"]

    def test_clear_status_sends_cls(self) -> None:
        transport = FakeSCPITransport()
        SCPIInstrument(transport).clear_status()
        assert transport.sent == ["*CLS"]

    def test_self_test_passed_true_on_zero(self) -> None:
        transport = FakeSCPITransport(["0"])
        assert SCPIInstrument(transport).self_test_passed() is True
        assert transport.sent == ["*TST?"]

    def test_self_test_passed_false_on_nonzero(self) -> None:
        transport = FakeSCPITransport(["1"])
        assert SCPIInstrument(transport).self_test_passed() is False

    def test_operation_complete_true_on_one(self) -> None:
        transport = FakeSCPITransport(["1"])
        assert SCPIInstrument(transport).operation_complete() is True
        assert transport.sent == ["*OPC?"]

    def test_operation_complete_false_on_zero(self) -> None:
        transport = FakeSCPITransport(["0"])
        assert SCPIInstrument(transport).operation_complete() is False

    def test_write_and_query_pass_through(self) -> None:
        transport = FakeSCPITransport(["42"])
        instrument = SCPIInstrument(transport)

        instrument.write("C1:BSWV WVTP,SINE")
        response = instrument.query("C1:BSWV?")

        assert transport.sent == ["C1:BSWV WVTP,SINE", "C1:BSWV?"]
        assert response == "42"

    def test_close_closes_transport(self) -> None:
        transport = FakeSCPITransport()
        instrument = SCPIInstrument(transport)
        instrument.close()
        assert transport.closed is True

    def test_context_manager_closes_on_exit(self) -> None:
        transport = FakeSCPITransport()
        with SCPIInstrument(transport) as instrument:
            assert isinstance(instrument, SCPIInstrument)
        assert transport.closed is True


class TestSocketSCPITransport:
    def test_raises_on_connection_refused(self) -> None:
        # Port 1 is a privileged port almost never listening; connecting
        # to localhost on it should be refused immediately.
        with pytest.raises(InstrumentConnectionError):
            SocketSCPITransport("127.0.0.1", port=1, timeout_s=0.5)

    def test_round_trips_against_a_local_echo_server(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        transport = SocketSCPITransport(host, port=port, timeout_s=2.0)
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(2.0)

                transport.write("*RST")
                assert connection.recv(1024) == b"*RST\n"

                # Queue the "instrument"'s reply in the socket's send
                # buffer before querying, so the client's blocking read
                # can be satisfied without a second thread.
                connection.sendall(b"FAKE,INSTRUMENT,1\n")
                result = transport.query("*IDN?")
                assert connection.recv(1024) == b"*IDN?\n"

                assert result == "FAKE,INSTRUMENT,1"
        finally:
            transport.close()
            server.close()

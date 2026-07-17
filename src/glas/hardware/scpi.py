"""Generic SCPI transport and base instrument.

Provides a minimal, vendor-agnostic foundation for talking to any
SCPI-compliant (Standard Commands for Programmable Instruments) lab
instrument -- function generators, oscilloscopes, and similar bench
equipment -- over a raw TCP socket: the "SCPI-raw" convention most
LAN-connected instruments (including the Siglent SDG1032X) support on
port 5025, with no VISA runtime required.

The actual socket I/O sits behind :class:`SCPITransport`, a small
protocol with exactly the operations a SCPI command needs (write a
command, query a response, close the connection). Every device-specific
class in :mod:`glas.hardware` is built on :class:`SCPIInstrument`, which
takes a transport rather than opening its own socket -- so command
building and response parsing for the waveform generator, the shaker
controller, and the generic oscilloscope wrapper are all unit-testable
against a fake transport, with no physical instrument or network access
required.
"""

from __future__ import annotations

import socket
from typing import Protocol

from glas.exceptions import InstrumentConnectionError

DEFAULT_SCPI_PORT = 5025
"""The de facto standard "SCPI-raw" TCP port used by most LAN-connected
bench instruments. Consult the specific instrument's manual if it uses a
different port."""

DEFAULT_TIMEOUT_S = 5.0


class SCPITransport(Protocol):
    """The minimal operations a SCPI command layer needs from a transport."""

    def write(self, command: str) -> None:
        """Send a command with no response expected."""
        ...

    def query(self, command: str) -> str:
        """Send a command and return its response, with trailing whitespace stripped."""
        ...

    def close(self) -> None:
        """Release the underlying connection. Safe to call more than once."""
        ...


class SocketSCPITransport:
    """A SCPI transport over a raw TCP socket ("SCPI-raw", port 5025 by default).

    Parameters
    ----------
    host : str
        Instrument's IP address or hostname.
    port : int, default 5025
        TCP port (see :data:`DEFAULT_SCPI_PORT`).
    timeout_s : float, default 5.0
        Socket timeout, in seconds, applied to connecting and to each
        subsequent read/write.

    Raises
    ------
    InstrumentConnectionError
        If the connection cannot be established within ``timeout_s``.
    """

    def __init__(
        self, host: str, *, port: int = DEFAULT_SCPI_PORT, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        try:
            self._socket = socket.create_connection((host, port), timeout=timeout_s)
        except OSError as exc:
            raise InstrumentConnectionError(
                f"Could not connect to SCPI instrument at {host}:{port}: {exc}"
            ) from exc
        self._buffer = b""

    def write(self, command: str) -> None:
        """Send ``command`` with a trailing newline; raises no response is read."""
        self._send(command)

    def query(self, command: str) -> str:
        """Send ``command`` and return the next newline-terminated response line."""
        self._send(command)
        return self._read_line()

    def close(self) -> None:
        """Close the underlying socket. Safe to call more than once."""
        self._socket.close()

    def _send(self, command: str) -> None:
        try:
            self._socket.sendall(command.encode("ascii") + b"\n")
        except OSError as exc:
            raise InstrumentConnectionError(f"Could not send SCPI command: {exc}") from exc

    def _read_line(self) -> str:
        while b"\n" not in self._buffer:
            try:
                chunk = self._socket.recv(4096)
            except OSError as exc:
                raise InstrumentConnectionError(f"Could not read SCPI response: {exc}") from exc
            if not chunk:
                raise InstrumentConnectionError(
                    "SCPI instrument closed the connection before a full response arrived."
                )
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("ascii", errors="replace").strip()


class SCPIInstrument:
    """Base class for a SCPI-compliant instrument, built on an injected transport.

    Every device-specific subclass in :mod:`glas.hardware` composes this
    rather than talking to a socket directly, so its own command-building
    logic stays testable against a fake :class:`SCPITransport`.

    Parameters
    ----------
    transport : SCPITransport
        An already-connected transport (e.g. :class:`SocketSCPITransport`,
        or a fake for testing).
    """

    def __init__(self, transport: SCPITransport) -> None:
        self._transport = transport

    def write(self, command: str) -> None:
        """Send a raw SCPI command with no response expected."""
        self._transport.write(command)

    def query(self, command: str) -> str:
        """Send a raw SCPI command and return its response."""
        return self._transport.query(command)

    def identify(self) -> str:
        """Query the instrument's identity string (the standard ``*IDN?`` command)."""
        return self.query("*IDN?")

    def reset(self) -> None:
        """Reset the instrument to its power-on default state (``*RST``)."""
        self.write("*RST")

    def clear_status(self) -> None:
        """Clear the instrument's status and error queues (``*CLS``)."""
        self.write("*CLS")

    def self_test_passed(self) -> bool:
        """Run the instrument's self-test and report whether it passed (``*TST?``).

        Per IEEE 488.2, ``*TST?`` returns ``0`` on success and a nonzero,
        instrument-defined error code on failure.
        """
        return self.query("*TST?").strip() == "0"

    def operation_complete(self) -> bool:
        """Block until all pending operations finish, per IEEE 488.2 (``*OPC?``).

        Useful before issuing a command that depends on a previous one
        having taken effect (e.g. waiting for a waveform change to
        settle before triggering a measurement).
        """
        return self.query("*OPC?").strip() == "1"

    def close(self) -> None:
        """Close the underlying transport. Safe to call more than once."""
        self._transport.close()

    def __enter__(self) -> SCPIInstrument:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

"""Tests for glas.hardware.daq."""

from __future__ import annotations

import pytest

from glas.exceptions import InstrumentConnectionError
from glas.hardware.daq import LabJackDAQ, NiDAQ


class FakeLjm:
    """A fake ``labjack.ljm`` module, for testing without the real SDK/driver."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.opened_with: tuple[str, str, str] | None = None
        self.closed_handle: int | None = None
        self._values = values if values is not None else {}
        self._next_handle = 1

    def openS(self, device_type: str, connection_type: str, identifier: str) -> int:  # noqa: N802
        self.opened_with = (device_type, connection_type, identifier)
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def eReadName(self, handle: int, name: str) -> float:  # noqa: N802
        return self._values[name]

    def close(self, handle: int) -> None:
        self.closed_handle = handle


class TestLabJackDAQ:
    def test_connect_passes_through_parameters(self) -> None:
        fake = FakeLjm()
        daq = LabJackDAQ(device_type="T7", connection_type="USB", identifier="470012345", _ljm=fake)
        daq.connect()
        assert fake.opened_with == ("T7", "USB", "470012345")

    def test_read_channel_maps_to_ain_name(self) -> None:
        fake = FakeLjm({"AIN0": 1.5, "AIN3": -0.25})
        daq = LabJackDAQ(_ljm=fake)
        daq.connect()

        assert daq.read_channel(0) == pytest.approx(1.5)
        assert daq.read_channel(3) == pytest.approx(-0.25)

    def test_read_channels_reads_each_channel(self) -> None:
        fake = FakeLjm({"AIN0": 1.0, "AIN1": 2.0})
        daq = LabJackDAQ(_ljm=fake)
        daq.connect()

        assert daq.read_channels([0, 1]) == {0: 1.0, 1: 2.0}

    def test_close_calls_ljm_close_with_handle(self) -> None:
        fake = FakeLjm()
        daq = LabJackDAQ(_ljm=fake)
        daq.connect()
        daq.close()
        assert fake.closed_handle is not None

    def test_read_before_connect_raises(self) -> None:
        with pytest.raises(InstrumentConnectionError):
            LabJackDAQ(_ljm=FakeLjm()).read_channel(0)

    def test_missing_sdk_raises_on_connect(self) -> None:
        with pytest.raises(InstrumentConnectionError):
            LabJackDAQ().connect()

    def test_context_manager_connects_and_closes(self) -> None:
        fake = FakeLjm({"AIN0": 5.0})
        with LabJackDAQ(_ljm=fake) as daq:
            assert daq.read_channel(0) == pytest.approx(5.0)
        assert fake.closed_handle is not None


class _FakeAIChannels:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add_ai_voltage_chan(self, name: str) -> None:
        self.added.append(name)


class _FakeTask:
    def __init__(self, value: float) -> None:
        self.ai_channels = _FakeAIChannels()
        self._value = value

    def read(self) -> float:
        return self._value

    def __enter__(self) -> _FakeTask:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class FakeNidaqmx:
    """A fake ``nidaqmx`` module, for testing without the real SDK/driver."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.tasks: list[_FakeTask] = []

    def Task(self) -> _FakeTask:  # noqa: N802
        task = _FakeTask(self.value)
        self.tasks.append(task)
        return task


class TestNiDAQ:
    def test_read_channel_configures_and_reads(self) -> None:
        fake = FakeNidaqmx(value=3.3)
        daq = NiDAQ("Dev1", _nidaqmx=fake)
        daq.connect()

        result = daq.read_channel(0)

        assert result == pytest.approx(3.3)
        assert fake.tasks[0].ai_channels.added == ["Dev1/ai0"]

    def test_read_channels_reads_each_channel(self) -> None:
        fake = FakeNidaqmx(value=1.0)
        daq = NiDAQ("Dev2", _nidaqmx=fake)
        daq.connect()

        result = daq.read_channels([0, 1])

        assert result == {0: 1.0, 1: 1.0}
        assert fake.tasks[0].ai_channels.added == ["Dev2/ai0"]
        assert fake.tasks[1].ai_channels.added == ["Dev2/ai1"]

    def test_read_before_connect_raises(self) -> None:
        with pytest.raises(InstrumentConnectionError):
            NiDAQ("Dev1", _nidaqmx=FakeNidaqmx()).read_channel(0)

    def test_missing_sdk_raises_on_connect(self) -> None:
        with pytest.raises(InstrumentConnectionError):
            NiDAQ("Dev1").connect()

    def test_context_manager_connects_and_closes(self) -> None:
        fake = FakeNidaqmx(value=2.0)
        with NiDAQ("Dev1", _nidaqmx=fake) as daq:
            assert daq.read_channel(0) == pytest.approx(2.0)

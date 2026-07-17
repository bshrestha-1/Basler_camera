# Hardware Integration

Phase 17 adds support for the lab equipment used to drive and monitor a
vibrated granular-material experiment: a Siglent SDG1032X function
generator, a Modal Shop 2025E shaker (via that generator), LabJack and
National Instruments DAQ devices, a generic SCPI oscilloscope, and
Basler camera hardware triggering.

```
glas.camera.Camera.enable_hardware_trigger()   -- extends the existing Camera class

glas.hardware.scpi.SCPITransport --> SCPIInstrument --> SiglentSDG1032X --> ShakerController
                                                      \-> SCPIOscilloscope

glas.hardware.daq.AnalogInputDAQ --> LabJackDAQ
                                  \-> NiDAQ
```

## Design principle: hardware you can test without hardware

Every class in this phase is built so its own logic -- command building,
argument validation, error wrapping -- is unit-testable without a
physical instrument attached:

- The SCPI-based classes (`SiglentSDG1032X`, `SCPIOscilloscope`,
  `ShakerController`) take a transport object rather than opening their
  own socket. Tests inject a fake transport; real use injects
  `SocketSCPITransport` (or nothing, and it connects over TCP itself).
- The DAQ classes (`LabJackDAQ`, `NiDAQ`) defer importing their vendor
  SDK (`labjack-ljm`, `nidaqmx`) until `connect()`, and accept the SDK
  module itself as an injectable constructor parameter. Neither SDK is a
  hard GLAS dependency.
- Camera hardware triggering extends `glas.camera.Camera` directly (not
  a separate module), and is tested against pypylon's built-in camera
  emulator the same way the rest of `glas.camera` is.

See each module's own docstring for the specifics.

## Camera hardware triggering

```python
from glas.camera import Camera

camera = Camera()
camera.connect()
camera.enable_hardware_trigger(source="Line1", activation="RisingEdge")
# ... record ...
camera.disable_hardware_trigger()
```

or from the command line:

```bash
glas trigger enable --source Line1 --activation RisingEdge
glas trigger status
glas trigger disable
```

`enable_hardware_trigger()` configures the four related GenICam features
Basler cameras expose for triggering (`TriggerSelector`, `TriggerSource`,
`TriggerActivation`, `TriggerMode`) in the order GenICam expects.
Invalid `source`/`activation` values raise `CameraConfigurationError`
listing what the connected device actually supports; a device with no
triggering support at all raises `CameraFeatureUnavailableError`.

## Siglent SDG1032X waveform generator

```python
from glas.hardware.scpi import SocketSCPITransport
from glas.hardware.waveform_generator import SiglentSDG1032X

generator = SiglentSDG1032X(SocketSCPITransport("192.168.1.50"))
generator.set_sine_wave(1, frequency_hz=60.0, amplitude_vpp=2.0)
generator.enable_output(1)
```

or from the command line:

```bash
glas waveform-gen sine 192.168.1.50 --frequency-hz 60 --amplitude-vpp 2.0 --enable
```

Uses the `BSWV` ("basic wave") SCPI command family documented in
Siglent's SDG-X series programming guide -- the common subset needed to
drive a shaker at a target sinusoidal frequency and amplitude. Connects
over the "SCPI-raw" TCP convention (port 5025) most LAN-connected bench
instruments support, no VISA runtime required.

## Modal Shop 2025E shaker

The 2025E is a power amplifier -- an analog gain stage with no digital
control interface, so there is no protocol to talk to it over directly.
What's controllable digitally is the waveform generator feeding its
input: frequency and drive voltage. `ShakerController` computes the
drive voltage needed to reach a target Gamma (see
[`accelerometer.md`](accelerometer.md) for what Gamma means) from a
`ShakerCalibration` measured once for the specific shaker + amplifier +
fixture + mounted-mass combination in use:

```python
from glas.hardware.waveform_generator import SiglentSDG1032X
from glas.hardware.shaker import ShakerCalibration, ShakerController

generator = SiglentSDG1032X(SocketSCPITransport("192.168.1.50"))
calibration = ShakerCalibration(volts_per_g=0.5, frequency_hz=60.0)
shaker = ShakerController(generator, calibration)

shaker.set_target_gamma(2.0)  # -> 1.0 Vpp at 60 Hz, per this calibration
shaker.start()
```

or from the command line:

```bash
glas shaker set-gamma 192.168.1.50 2.0 --volts-per-g 0.5 --calibration-frequency-hz 60 --start
```

A shaker's gain (g per volt) depends on the amplifier, the fixture, and
the mounted mass together, and on frequency -- there is no universal
constant. Measure `volts_per_g` at the frequency intended for use, and
re-measure if the fixture or mounted mass changes.

## Oscilloscope

No specific oscilloscope model was given, and SCPI dialects vary
significantly between vendors (Siglent SDS, Keysight InfiniiVision,
Rigol, Tektronix, ...) even for basic operations. `SCPIOscilloscope`
only adds the universal IEEE 488.2 commands (identify, reset, self-test,
operation-complete) plus `query_float()`, a small helper for the common
pattern of a measurement query returning a bare number:

```python
from glas.hardware.oscilloscope import SCPIOscilloscope

scope = SCPIOscilloscope(SocketSCPITransport("192.168.1.60"))
peak_to_peak = scope.query_float("C1:PAVA? PKPK")  # syntax varies by vendor
```

or from the command line, for any raw SCPI query:

```bash
glas oscilloscope query 192.168.1.60 "*IDN?"
```

Use `write()`/`query()` directly with the exact command syntax from the
connected oscilloscope's own programming guide for anything
model-specific (channel scaling, trigger setup, waveform capture).

## LabJack and National Instruments DAQ

```python
from glas.hardware.daq import LabJackDAQ, NiDAQ

with LabJackDAQ(device_type="T7") as daq:
    voltage = daq.read_channel(0)

with NiDAQ("Dev1") as daq:
    voltages = daq.read_channels([0, 1, 2])
```

or from the command line:

```bash
glas daq read labjack --channel 0
glas daq read ni --device-name Dev1 --channel 0
```

Both require their vendor's proprietary driver runtime installed
(LabJack's LJM driver, National Instruments' NI-DAQmx runtime) --
neither is a hard GLAS dependency, since most installations won't have
this specific hardware. `connect()` raises `InstrumentConnectionError`
with a clear message (naming the missing package) if the corresponding
Python SDK isn't installed.

## Testing

The SCPI-based classes are tested against a fake transport recording
every command sent, with exact expected-command assertions (e.g.
`compute_gamma`-derived drive voltages, `BSWV` command strings) -- no
physical instrument or network access required. `SocketSCPITransport`
itself is tested against a real local TCP server (both a connection-
refused case and a full write/read round trip). The DAQ classes are
tested against fake `labjack.ljm`/`nidaqmx` modules standing in for the
real vendor SDKs, covering both the happy path and the "SDK not
installed" error path. Camera hardware triggering is tested against
pypylon's built-in camera emulator, exercising the real GenICam node
access, not a mock. CLI commands for the SCPI-based instruments are
tested against real local TCP servers (mirroring how
`tests/test_scpi.py` tests the transport itself); the DAQ CLI command is
tested against the real "SDK not installed" failure, since neither
vendor SDK is installed in this environment.

# Accelerometer Synchronization

Phase 16 imports a PCB Piezotronics 352C22 accelerometer recording
(exported as a CSV file by whatever DAQ software captured it), computes
the standard vibration diagnostics used in granular-material physics --
frequency, displacement amplitude, and the dimensionless acceleration
Gamma -- and aligns the recording's timeline with a recorded camera
dataset's frames.

```
import_accelerometer_csv() -> AccelerometerRecording -> compute_vibration_frequency()
                                        |                          |
                                        |                 compute_vibration_amplitude()
                                        |                          |
                                        |                    compute_gamma()
                                        |
                              synchronize_with_frames()
```

Hardware-triggered synchronization -- so the accelerometer and camera
clocks share an exact common zero point -- is Phase 17's concern
(function generators, shakers, and DAQ integration). Here,
`synchronize_with_frames()` aligns the two recordings' *relative*
timelines, assuming they started at the same real-world moment unless an
explicit offset is given.

## Quickstart

```python
from glas.accelerometer import analyze_vibration

metrics = analyze_vibration(Path("shaker_run.csv"), plot_path=Path("signal.png"))
print(f"frequency: {metrics.frequency_hz:.1f} Hz")
print(f"amplitude: {metrics.amplitude_m * 1000:.3f} mm")
print(f"Gamma: {metrics.gamma:.2f}")
```

or from the command line:

```bash
glas accelerometer analyze shaker_run.csv --plot signal.png
```

`analyze_vibration()` runs the whole frequency/amplitude/Gamma pipeline
over a CSV file in one call, the same role
`glas.analysis.analyze_packing()` plays for its own phase.

## Importing a recording

`import_accelerometer_csv()` reads a CSV file with a header row and at
least two columns: a timestamp column (`time_s` by default) and a
measured-value column (`voltage_v` by default). The value column can
hold either:

- **Raw sensor voltage** (`value_units="volts"`, the default): converted
  to acceleration in g via the accelerometer's sensitivity,
  `acceleration_g = voltage_v * 1000 / sensitivity_mv_per_g`. Use the
  value from the specific unit's calibration certificate --
  `DEFAULT_SENSITIVITY_MV_PER_G` (10.0 mV/g) is only a typical nominal
  starting point for a quick check, not a substitute for calibration
  data.
- **Already-converted acceleration** (`value_units="g"`): passed through
  unchanged, for DAQ software that does the voltage-to-g conversion
  itself before export.

Column names are configurable via `time_column`/`value_column` for DAQ
software with different export conventions. Timestamps must be strictly
increasing; malformed rows, missing columns, or a missing file all raise
`AccelerometerError` with a message identifying exactly what went wrong.

## Frequency, amplitude, and Gamma

- `compute_vibration_frequency()` finds the dominant frequency via a
  real FFT of the (mean-removed) acceleration signal, assuming uniform
  sampling at the recording's nominal sample rate. Needs at least 4
  samples.
- `compute_gamma()` returns the dimensionless vibration intensity
  `Gamma = peak acceleration / g` -- the standard control parameter for
  a sinusoidally vibrated granular bed. Since the accelerometer measures
  acceleration directly (not displacement), this is simply the peak
  measured acceleration expressed in units of g: the textbook definition
  `Gamma = A * omega^2 / g` has a `g` in both the numerator (via
  `A * omega^2`, the acceleration amplitude) and denominator, and they
  cancel when acceleration is already measured in g.
- `compute_vibration_amplitude()` inverts that relationship --
  `A = peak_acceleration / omega^2` -- to recover displacement amplitude
  in meters, assuming sinusoidal motion.

## Synchronizing with camera frames

`synchronize_with_frames()` takes an `AccelerometerRecording` and a
sequence of `Frame`s (e.g. from `glas.dataset.iter_frames()`), and
returns one acceleration value per frame -- the nearest accelerometer
sample in time, found via `numpy.searchsorted` (no interpolation, no new
dependency).

Both recordings' timelines are shifted to start at zero (the first
accelerometer sample, the first frame) before matching, which assumes
the two recordings started at the same real-world moment. This is
approximately true if both were started manually within the same second
or two; pass `offset_s` to correct for a known difference (positive if
the accelerometer recording started before the camera recording).
Getting this precisely right -- so the two clocks share an *exact*
common zero point via a hardware trigger pulse -- is Phase 17's job;
Phase 16's synchronization is a best-effort software alignment for setups
without that hardware yet.

From the command line, `glas accelerometer sync` writes the synchronized
result straight to a CSV (`frame_id,host_timestamp_ns,acceleration_g`):

```bash
glas accelerometer sync shaker_run.csv ~/glas_data/Run0001 --output synced.csv
```

## Testing

`import_accelerometer_csv()` is tested against real CSV files covering
both unit conventions, custom column names, and every error path
(missing file, missing columns, non-numeric values, too few rows,
non-increasing timestamps). `compute_vibration_frequency()`,
`compute_gamma()`, and `compute_vibration_amplitude()` are tested
against a hand-constructed exact sinusoidal signal with a known
frequency and displacement amplitude, recovering both to within the
FFT's frequency resolution. `synchronize_with_frames()` is tested
against hand-picked sample/frame timestamps with a known nearest-sample
answer, including the `offset_s` parameter and the empty-frames error
path. `plot_vibration_signal()`'s output is verified as a real, readable
PNG, and `analyze_vibration()` is tested end to end against a real CSV
file.

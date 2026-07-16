# Performance Monitor

Phase 7 answers the operational question a long-running lab acquisition
needs answered continuously: is the pipeline keeping up, and is the host
machine about to become the bottleneck?

```
RingBuffer -.
             v
           PerformanceMonitor -> PerformanceSnapshot (fps, queue usage,
                                                        dropped frames,
                                                        CPU, RAM, disk)
```

Like `glas.preview.Preview`, `PerformanceMonitor` only ever reads a
`RingBuffer`'s lock-free, point-in-time `stats()` snapshot -- never
`pop()` -- so attaching a monitor to a buffer that's also being recorded
from or previewed can never affect either of them.

## Quickstart

```python
import time
from glas.monitor import PerformanceMonitor

monitor = PerformanceMonitor(recorder.buffer, data_dir=recorder.dataset.folder)

while recording:
    snapshot = monitor.sample()
    print(
        f"{snapshot.fps:.1f} fps | "
        f"buffer {snapshot.buffer_occupancy_percent:.0f}% | "
        f"dropped {snapshot.dropped_frame_count} | "
        f"cpu {snapshot.cpu_percent:.0f}% | "
        f"ram {snapshot.memory_used_mb:.0f} MB | "
        f"disk free {snapshot.disk_free_gb:.1f} GB"
    )
    time.sleep(1.0)
```

`data_dir` should point at the filesystem you care about running out of
space on -- typically the experiment folder currently being recorded to,
or its parent data directory. It must already exist; `PerformanceMonitor`
raises `FileNotFoundError` at construction otherwise, rather than failing
silently or on the first `sample()` call.

## What each field means

- **`fps`** -- frames per second pushed onto the ring buffer, averaged
  over the most recent `fps_window` (default 30) calls to `sample()`.
  Derived from `RingBufferStats.pushed`, a counter incremented by every
  producer push regardless of what any consumer does with the frames
  afterward -- so this stays accurate even if `sample()` itself is called
  irregularly, and even if nothing is draining the buffer at all.
- **`buffer_size` / `buffer_capacity` / `buffer_occupancy_percent`** --
  how full the ring buffer is right now. Sustained values near 100% mean
  downstream consumers (a `DatasetWriter`, a `Preview`) aren't draining
  frames as fast as they arrive -- the next thing to check is
  `dropped_frame_count` climbing.
- **`dropped_frame_count`** -- total frames overwritten because the
  buffer was full, for the buffer's whole lifetime (not just since the
  last sample).
- **`cpu_percent` / `memory_used_mb` / `memory_percent`** -- this
  process's own resource footprint (via `psutil.Process`), not
  system-wide usage. Reporting the GLAS process specifically, rather than
  the whole machine, is deliberately what a lab operator watching *this*
  software actually wants to know -- "is GLAS itself struggling" -- and
  what stays meaningful regardless of what else happens to be running on
  the machine.
- **`disk_free_gb` / `disk_used_percent`** -- free space and usage on the
  filesystem holding `data_dir`, via `shutil.disk_usage`. A long
  recording session (uncompressed HDF5 frames add up fast) can fill a
  disk gradually enough that catching it early matters.

## Testing

All of `PerformanceMonitor`'s buffer-derived fields are tested against a
real (in-memory) `RingBuffer`, and disk fields against a real `tmp_path`
filesystem. `cpu_percent`/`memory_used_mb`/`memory_percent` are exercised
against the real, running test process via `psutil` (not mocked) --
assertions are necessarily loose (non-negative, or positive for memory,
which any live process always has), since exact CPU/RAM usage isn't
deterministic. FPS is tested deterministically by monkeypatching
`glas.monitor.time.monotonic` to fixed values rather than relying on real
wall-clock timing, avoiding the kind of flakiness a live-timing-based test
would have.

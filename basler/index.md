# GLAS Documentation

GLAS (Granular Lab Acquisition System) is a production-quality acquisition
and analysis platform built around a Basler ace acA640-750um camera, for
granular-material physics experiments (Brazil nut effect, convection,
packing fraction, segregation, and related studies).

The project is developed in phases; each phase ships as a fully tested,
documented, production-ready release before the next one starts.

## Current status: Phase 1 — Core Infrastructure (v0.1.0)

Phase 1 lays the foundation that every later phase builds on. No camera or
acquisition code exists yet — this phase only provides:

- **Configuration management** (`glas.config`, `glas.settings`)
- **Logging** (`glas.logger`)
- **Error handling** (`glas.exceptions`)
- **JSON Schema validation** (`glas.config.validate_json`)
- **Command-line interface** (`glas.cli`)
- **Unit tests** (`tests/`)

See [`configuration.md`](configuration.md) for how configuration works and
[`development.md`](development.md) for the developer workflow.

## Roadmap

| Phase | Version | Goal |
|-------|---------|------|
| 1 | v0.1 | Core infrastructure (this release) |
| 2 | v0.2 | Camera layer — detect, connect, configure the Basler camera |
| 3 | v0.3 | Image acquisition — producer thread, ring buffer, no disk writes |
| 4 | v0.4 | Dataset writer — HDF5 storage, metadata, checksums |
| 5 | v0.5 | Recorder — start/stop/pause/resume (first usable release) |
| 6 | v0.6 | Live preview — histogram, FPS, zoom, crosshair |
| 7 | v0.7 | Performance monitor — FPS, queue usage, CPU/RAM/disk |
| 8 | v0.8 | Export engine — TIFF, PNG, MP4, AVI, GIF |
| 9 | v0.9 | Experiment manager — automatic run folders, metadata |
| 10 | v1.0 | Production release — the version used to collect scientific data |
| 11 | v1.1 | Analysis engine — particle tracking |
| 12 | v1.2 | Brazil nut analysis |
| 13 | v1.3 | Convection analysis — optical flow, velocity fields |
| 14 | v1.4 | Packing analysis |
| 15 | v1.5 | Segregation analysis |
| 16 | v1.6 | Accelerometer synchronization (PCB 352C22, Γ, frequency/amplitude) |
| 17 | v1.7 | Hardware integration — function generators, shakers, DAQ |
| 18 | v2.0 | GUI |
| 19 | v2.5 | AI analysis — YOLO / SAM2 |
| 20 | v3.0 | Full research platform |

Each phase's files, features, and acceptance criteria are tracked in the
project's development roadmap.

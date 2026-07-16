# Developer Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs GLAS in editable mode plus the development tools: `pytest`,
`pytest-cov`, `ruff`, and `mypy`.

## Running the test suite

```bash
pytest
pytest --cov=glas --cov-report=term-missing   # with coverage
```

Every module under `src/glas/` has a corresponding `tests/test_*.py` file.
Camera- and dataset-touching tests run against pypylon's built-in emulated
camera transport layer (`PYLON_CAMEMU`, set in `tests/conftest.py`), so the
full suite runs without physical hardware attached. Tests must pass before
a phase is considered complete.

## Linting and formatting

```bash
ruff check src tests      # lint
ruff format src tests     # format
```

## Type checking

```bash
mypy
```

The project is fully type-annotated and checked in `strict` mode
(see `[tool.mypy]` in `pyproject.toml`).

## Continuous integration

`.github/workflows/ci.yml` runs `pytest`, `ruff check`, `ruff format --check`,
and `mypy` on every push and pull request against `main`, across Python
3.10/3.11/3.12. A PR isn't mergeable until it's green; run the same four
commands locally before pushing to catch failures early.

## Coding standards

- **PEP 8** compliant, enforced by `ruff`.
- **Type-annotated** throughout; checked with `mypy --strict`.
- **NumPy-style docstrings** on every public module, class, and function.
- **Pydantic v2 for data modeling and validation.** Every data-carrying
  type that is loaded from a file, crosses a boundary, or is otherwise
  worth validating is a Pydantic `BaseModel` (`model_config =
  ConfigDict(frozen=True)` for immutable value types), not a plain
  `dataclass`. `glas.frame.Frame` is the one deliberate exception --
  see its module docstring for why (hot path, never a validation
  boundary, holds a numpy array Pydantic can't meaningfully validate
  anyway).
- **No placeholder code.** Every module merged into `main` is fully
  functional for the phase it belongs to — no `TODO` stubs or
  not-implemented branches.
- **Semantic Versioning.** The version lives in one place
  (`glas/version.py`) and is reflected in `pyproject.toml` and
  `CHANGELOG.md` together on every release.

## Project layout

```
src/glas/            Package source (importable as `import glas`)
  __init__.py         Public API surface
  version.py           Semantic version (single source of truth)
  exceptions.py        Exception hierarchy (GLASError and subclasses)
  logger.py            Logging: Rich-formatted console + plain rotating file
  config.py            Generic YAML loading and merging (no validation)
  settings.py          GLAS-specific defaults and Pydantic-validated Settings
  cli.py               Typer command-line interface
  camera_info.py        Camera discovery, USB diagnostics
  camera_validator.py   Pure validation logic for camera parameters
  camera.py             Camera connection and control
  frame.py               Frame data structure (the Pydantic exception)
  ringbuffer.py          Drop-oldest ring buffer (push/pop and peek)
  acquisition.py         Producer thread: Camera -> RingBuffer
  metadata.py             Dataset metadata (Pydantic-validated)
  timestamps.py           Per-frame timestamp bookkeeping, gap detection
  dataset.py               On-disk dataset storage (HDF5 / raw binary), iter_frames()
  writer.py                 Background writer thread: RingBuffer -> Dataset
  recorder.py                One recording session: start/stop/pause/resume
  controller.py                Top-level orchestration, graceful shutdown
  preview.py                   Live-frame tracking, zoom/crosshair/ROI, FPS, histogram
  display.py                    OpenCV rendering and windowing (headless-safe)
  monitor.py                     Pipeline/system performance sampling
  export.py                      TIFF/PNG/MP4/AVI/GIF dataset export
  experiment.py                   Search/list recordings by name, tag, camera model
  analysis/                        Particle detection and tracking
    __init__.py
    tracking_utils.py               Pure detection/linking functions
    particle_tracking.py            ParticleTracker, track_dataset()
tests/               pytest test suite, one file per src module
docs/                Project documentation
.github/workflows/    CI (ci.yml)
```

## Phase workflow

GLAS is built one roadmap phase at a time (see `docs/index.md`). A phase is
not considered done until:

1. All features listed for the phase are implemented with no stubs.
2. Every new module has a passing `pytest` test file.
3. `ruff check`, `ruff format --check`, and `mypy` are clean.
4. `CHANGELOG.md` has an entry for the release.
5. Documentation in `docs/` reflects the new functionality.

Only after that does work start on the next phase.

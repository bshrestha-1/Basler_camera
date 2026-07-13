# GLAS — Granular Lab Acquisition System

GLAS is a production-quality acquisition and analysis platform for a
**Basler ace acA640-750um** camera, built for granular-material physics
research (Brazil nut effect, convection, packing fraction, segregation,
and related studies). It is developed in phases, each shipped as a fully
tested, documented, production-ready release.

**Current release: Phase 1 — Core Infrastructure (v0.1.0).** No camera or
acquisition code exists yet; this phase establishes configuration,
logging, error handling, and the CLI that every later phase depends on.

## Installation

```bash
git clone https://github.com/bshrestha-1/basler_camera.git
cd basler_camera
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development (tests, linting, type checking):

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
glas --version

# Write a default configuration file
glas config init --path ~/.config/glas/config.yaml

# Validate it
glas config validate ~/.config/glas/config.yaml

# See the resolved settings (defaults merged with your file)
glas config show
```

```python
from glas import Settings, configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

settings = Settings.load()
settings.ensure_directories()
logger.info("GLAS ready, data_dir=%s", settings.data_dir)
```

See [`docs/configuration.md`](docs/configuration.md) for the full
configuration schema and file-resolution order.

## Project layout

```
src/glas/          Package source (import glas)
tests/              pytest unit tests (one file per module)
docs/               Documentation
pyproject.toml      Packaging, dependencies, tool configuration
requirements.txt    Runtime dependencies
CHANGELOG.md         Release history (Keep a Changelog format)
```

## Development

```bash
pytest                        # run tests
pytest --cov=glas             # with coverage
ruff check src tests          # lint
ruff format src tests         # format
mypy                          # type check
```

See [`docs/development.md`](docs/development.md) for the full developer
workflow and coding standards, and [`docs/index.md`](docs/index.md) for
the complete 20-phase roadmap (camera layer, acquisition, dataset storage,
recording, live preview, analysis, hardware synchronization, GUI, and
beyond).

## License

MIT — see [`LICENSE`](LICENSE).

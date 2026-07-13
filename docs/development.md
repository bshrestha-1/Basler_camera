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
Tests must pass before a phase is considered complete.

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

## Coding standards

- **PEP 8** compliant, enforced by `ruff`.
- **Type-annotated** throughout; checked with `mypy --strict`.
- **NumPy-style docstrings** on every public module, class, and function.
- **No placeholder code.** Every module merged into `main` is fully
  functional for the phase it belongs to — no `TODO` stubs or
  not-implemented branches.
- **Semantic Versioning.** The version lives in one place
  (`glas/version.py`) and is reflected in `pyproject.toml` and
  `CHANGELOG.md` together on every release.

## Project layout

```
src/glas/          Package source (importable as `import glas`)
  __init__.py      Public API surface
  version.py       Semantic version (single source of truth)
  exceptions.py    Exception hierarchy (GLASError and subclasses)
  logger.py        Logging configuration (console + rotating file)
  config.py        Generic YAML loading, merging, JSON Schema validation
  settings.py      GLAS-specific defaults, schema, and typed Settings
  cli.py           Typer command-line interface
tests/             pytest test suite, one file per src module
docs/              Project documentation
```

## Phase workflow

GLAS is built one roadmap phase at a time (see `docs/index.md`). A phase is
not considered done until:

1. All features listed for the phase are implemented with no stubs.
2. Every new module has a passing `pytest` test file.
3. `ruff check` and `mypy` are clean.
4. `CHANGELOG.md` has an entry for the release.
5. Documentation in `docs/` reflects the new functionality.

Only after that does work start on the next phase.

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-13

Phase 1 — Core Infrastructure. Establishes the project foundation; no camera
or acquisition code yet.

### Added

- Project scaffolding: `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE` (MIT).
- `glas.config`: YAML configuration loading, deep-merging, and JSON Schema validation.
- `glas.settings`: typed, validated `Settings` dataclass with default configuration and schema.
- `glas.logger`: rotating-file and console logging via `configure_logging` / `get_logger`.
- `glas.exceptions`: project-wide exception hierarchy rooted at `GLASError`.
- `glas.version`: single-source semantic version (`__version__`, `VERSION_INFO`).
- `glas.cli`: Typer-based command-line interface (`glas --version`, `glas config init|show|validate`).
- `pytest` unit test suite covering all Phase 1 modules.
- Developer and configuration documentation in `docs/`.

[Unreleased]: https://github.com/bshrestha-1/basler_camera/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bshrestha-1/basler_camera/releases/tag/v0.1.0

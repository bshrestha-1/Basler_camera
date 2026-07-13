"""GLAS command-line interface.

Provides subcommands for inspecting the installed version and managing
configuration files. Camera and recording commands are added in later
development phases.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from glas.config import deep_merge, read_yaml_file
from glas.exceptions import ConfigurationError, JSONValidationError
from glas.logger import configure_logging, get_logger
from glas.settings import DEFAULT_CONFIG, Settings
from glas.version import __version__

app = typer.Typer(
    name="glas",
    help="GLAS: Granular Lab Acquisition System command-line interface.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage GLAS configuration files.")
app.add_typer(config_app, name="config")

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "glas" / "config.yaml"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"glas {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the GLAS version and exit.",
    ),
) -> None:
    """GLAS: Granular Lab Acquisition System."""


@config_app.command("init")
def config_init(
    path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--path", "-p", help="Where to write the new configuration file."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite the file if it already exists."
    ),
) -> None:
    """Write a default configuration file to PATH."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    typer.echo(f"Wrote default configuration to {path}")


@config_app.command("validate")
def config_validate(
    path: Path = typer.Argument(..., help="Configuration file to validate."),
) -> None:
    """Validate a configuration file against the GLAS schema."""
    try:
        file_data = read_yaml_file(path)
        merged = deep_merge(DEFAULT_CONFIG, file_data)
        Settings.from_dict(merged)
    except (ConfigurationError, JSONValidationError) as exc:
        typer.echo(f"Invalid configuration: {exc}", err=True)
        if isinstance(exc, JSONValidationError):
            for error in exc.errors:
                typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{path} is valid.")


@config_app.command("show")
def config_show(
    path: Path | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Configuration file to load instead of the default search path.",
    ),
) -> None:
    """Load configuration (defaults merged with a file, if found) and print it."""
    try:
        settings = Settings.load(config_path=path)
    except (ConfigurationError, JSONValidationError) as exc:
        typer.echo(f"Could not load configuration: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for field_name, value in settings.__dict__.items():
        typer.echo(f"{field_name}: {value}")


def run() -> None:
    """Entry point used by the ``glas`` console script."""
    configure_logging()
    app()


if __name__ == "__main__":
    run()

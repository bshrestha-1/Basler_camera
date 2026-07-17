"""Tests for the ``glas gui`` CLI command.

Does not actually launch a Qt event loop -- :func:`glas.gui.app.main` is
monkeypatched so these tests exercise only :mod:`glas.cli`'s own
lazy-import and exit-code plumbing, which is fully independent of whether
PySide6 is installed or a display is available.
"""

from __future__ import annotations

import builtins
from pathlib import Path

from typer.testing import CliRunner

from glas.cli import app

runner = CliRunner()


class TestGuiCommand:
    def test_launches_gui_and_forwards_exit_code(self, tmp_path: Path, monkeypatch) -> None:
        calls = []

        def fake_main(base_data_dir: Path) -> int:
            calls.append(base_data_dir)
            return 0

        monkeypatch.setattr("glas.gui.app.main", fake_main)

        result = runner.invoke(app, ["gui", str(tmp_path)])

        assert result.exit_code == 0
        assert calls == [tmp_path]

    def test_forwards_nonzero_exit_code(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("glas.gui.app.main", lambda base_data_dir: 1)

        result = runner.invoke(app, ["gui", str(tmp_path)])

        assert result.exit_code == 1

    def test_missing_pyside6_shows_install_hint(self, tmp_path: Path, monkeypatch) -> None:
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if name == "glas.gui.app" or name.startswith("PySide6"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        result = runner.invoke(app, ["gui", str(tmp_path)])

        assert result.exit_code == 1
        assert "pip install glas[gui]" in result.output

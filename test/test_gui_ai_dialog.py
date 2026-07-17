"""Tests for glas.gui.ai_dialog."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.gui.ai_dialog import show_missing_ai_dependencies_dialog


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_shows_a_warning_box_naming_every_missing_package(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, str, str]] = []
    monkeypatch.setattr(
        "glas.gui.ai_dialog.QMessageBox.warning",
        lambda parent, title, text: calls.append((parent, title, text)),
    )

    show_missing_ai_dependencies_dialog(None, ["torch", "ultralytics"])

    assert len(calls) == 1
    _, title, text = calls[0]
    assert title == "Missing AI Dependencies"
    assert "torch" in text
    assert "ultralytics" in text


def test_works_with_a_parent_widget(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QWidget

    calls: list[object] = []
    monkeypatch.setattr(
        "glas.gui.ai_dialog.QMessageBox.warning",
        lambda parent, title, text: calls.append(parent),
    )

    parent = QWidget()
    show_missing_ai_dependencies_dialog(parent, ["sam2"])

    assert calls == [parent]

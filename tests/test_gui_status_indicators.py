"""Tests for glas.gui.status_indicators."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QProgressBar

from glas.gui.status_indicators import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    resource_bar_color,
    status_dot_html,
    update_resource_bar,
)


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


class TestStatusDotHtml:
    def test_includes_the_color_and_text(self) -> None:
        html = status_dot_html(COLOR_GREEN, "Connected")
        assert COLOR_GREEN in html
        assert "Connected" in html
        assert "●" in html


class TestResourceBarColor:
    def test_low_usage_is_green(self) -> None:
        assert resource_bar_color(10.0) == COLOR_GREEN

    def test_just_below_warning_threshold_is_green(self) -> None:
        assert resource_bar_color(69.9) == COLOR_GREEN

    def test_at_warning_threshold_is_yellow(self) -> None:
        assert resource_bar_color(70.0) == COLOR_YELLOW

    def test_just_below_critical_threshold_is_yellow(self) -> None:
        assert resource_bar_color(89.9) == COLOR_YELLOW

    def test_at_critical_threshold_is_red(self) -> None:
        assert resource_bar_color(90.0) == COLOR_RED

    def test_over_100_percent_is_still_red(self) -> None:
        assert resource_bar_color(150.0) == COLOR_RED


class TestUpdateResourceBar:
    def test_sets_value_and_format_text(self, qapp: QApplication) -> None:
        bar = QProgressBar()
        bar.setRange(0, 100)
        update_resource_bar(bar, 42.0, "14/256 (42%)")
        assert bar.value() == 42
        assert bar.format() == "14/256 (42%)"

    def test_clamps_value_over_100_to_100(self, qapp: QApplication) -> None:
        bar = QProgressBar()
        bar.setRange(0, 100)
        update_resource_bar(bar, 150.0, "150%")
        assert bar.value() == 100

    def test_applies_green_stylesheet_below_warning_threshold(self, qapp: QApplication) -> None:
        bar = QProgressBar()
        bar.setRange(0, 100)
        update_resource_bar(bar, 10.0, "10%")
        assert COLOR_GREEN in bar.styleSheet()

    def test_applies_red_stylesheet_at_critical_threshold(self, qapp: QApplication) -> None:
        bar = QProgressBar()
        bar.setRange(0, 100)
        update_resource_bar(bar, 95.0, "95%")
        assert COLOR_RED in bar.styleSheet()

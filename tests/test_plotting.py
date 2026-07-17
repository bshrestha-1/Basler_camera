"""Tests for glas.plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from glas.plotting import (
    PUBLICATION_DPI,
    PUBLICATION_PALETTE,
    apply_publication_style,
    savefig_publication,
    style_axes,
)


class TestPublicationPalette:
    def test_has_eight_colors(self) -> None:
        assert len(PUBLICATION_PALETTE) == 8

    def test_every_entry_is_a_hex_color(self) -> None:
        for color in PUBLICATION_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7

    def test_every_color_is_unique(self) -> None:
        assert len(set(PUBLICATION_PALETTE)) == len(PUBLICATION_PALETTE)


class TestApplyPublicationStyle:
    def test_sets_savefig_dpi(self) -> None:
        apply_publication_style()
        assert plt.rcParams["savefig.dpi"] == PUBLICATION_DPI

    def test_sets_color_cycle_to_publication_palette(self) -> None:
        apply_publication_style()
        cycle_colors = [entry["color"] for entry in plt.rcParams["axes.prop_cycle"]]
        assert cycle_colors == list(PUBLICATION_PALETTE)

    def test_is_idempotent(self) -> None:
        apply_publication_style()
        first = dict(plt.rcParams)
        apply_publication_style()
        second = dict(plt.rcParams)
        assert first["savefig.dpi"] == second["savefig.dpi"]
        assert first["font.size"] == second["font.size"]

    def test_enables_grid(self) -> None:
        apply_publication_style()
        assert plt.rcParams["axes.grid"] is True


class TestStyleAxes:
    def test_removes_top_and_right_spines(self) -> None:
        fig, ax = plt.subplots()
        try:
            style_axes(ax)
            assert ax.spines["top"].get_visible() is False
            assert ax.spines["right"].get_visible() is False
        finally:
            plt.close(fig)

    def test_leaves_left_and_bottom_spines_visible(self) -> None:
        fig, ax = plt.subplots()
        try:
            style_axes(ax)
            assert ax.spines["left"].get_visible() is True
            assert ax.spines["bottom"].get_visible() is True
        finally:
            plt.close(fig)


class TestSavefigPublication:
    def test_writes_a_file(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        output_path = tmp_path / "figure.png"
        savefig_publication(fig, output_path)
        assert output_path.exists()

    def test_returns_the_output_path(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        output_path = tmp_path / "figure.png"
        result = savefig_publication(fig, output_path)
        assert result == output_path

    def test_closes_the_figure_by_default(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        fig_number = fig.number
        savefig_publication(fig, tmp_path / "figure.png")
        assert not plt.fignum_exists(fig_number)

    def test_close_false_keeps_the_figure_open(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        fig_number = fig.number
        try:
            savefig_publication(fig, tmp_path / "figure.png", close=False)
            assert plt.fignum_exists(fig_number)
        finally:
            plt.close(fig)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        output_path = tmp_path / "nested" / "dir" / "figure.png"
        savefig_publication(fig, output_path)
        assert output_path.exists()

    def test_supports_vector_formats(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        output_path = tmp_path / "figure.pdf"
        savefig_publication(fig, output_path)
        assert output_path.exists()
        assert output_path.read_bytes().startswith(b"%PDF")

    def test_raster_output_is_saved_at_publication_dpi(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([1, 2, 3], [1, 4, 9])
        output_path = tmp_path / "figure.png"
        savefig_publication(fig, output_path)

        with Image.open(output_path) as image:
            dpi_x, _ = image.info.get("dpi", (0, 0))
        assert round(dpi_x) == PUBLICATION_DPI

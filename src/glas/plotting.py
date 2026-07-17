"""Shared publication-quality styling for every ``plot_*`` function across GLAS.

Every analysis module (:mod:`glas.analysis.brazil_nut`,
:mod:`glas.analysis.convection`, :mod:`glas.analysis.packing`,
:mod:`glas.analysis.segregation`, :mod:`glas.accelerometer`) draws its own
plot, but all of them should look like they belong to the same figure set
when dropped into a paper or a report -- same fonts, same colorblind-safe
palette, same resolution -- rather than each picking up matplotlib's
mismatched defaults independently. :func:`apply_publication_style` is the
one-line hook every ``plot_*`` function calls before drawing anything;
:func:`savefig_publication` is the one-line hook it calls to save.

Nothing here changes what data gets plotted -- only how it looks and at
what resolution/format it's written to disk. A figure written through
:func:`savefig_publication` at 300 DPI to a ``.png`` is print-quality; the
same call with a ``.pdf``/``.svg`` path produces a vector figure
matplotlib's own format inference already handles, with the palette and
fonts set consistently either way.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: Okabe-Ito palette: eight colors, chosen to remain distinguishable under
#: the three common forms of color vision deficiency as well as in
#: grayscale printing -- the standard colorblind-safe qualitative palette
#: for scientific figures.
PUBLICATION_PALETTE: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
)

#: DPI every raster figure (or rasterized element within a vector figure)
#: is saved at -- print-quality, well above typical journal requirements
#: (usually 300 DPI minimum for line art/photos).
PUBLICATION_DPI = 300


def apply_publication_style() -> None:
    """Set matplotlib rcParams for consistent, publication-quality figures.

    Idempotent and cheap -- safe to call at the top of every ``plot_*``
    function rather than once globally, so each plotting call is
    self-contained and doesn't depend on import order.

    Sets the color cycle to :data:`PUBLICATION_PALETTE`, a readable
    sans-serif font stack at sizes appropriate for a print figure, a
    light background grid, and ``savefig.dpi``/``figure.dpi`` to
    :data:`PUBLICATION_DPI`.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.prop_cycle": cycler(color=PUBLICATION_PALETTE),
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.dpi": 100,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax: Axes) -> None:
    """Remove the top and right spines from ``ax``.

    A conventional publication-figure touch-up that ``rcParams`` alone
    can't express (spine visibility is per-axes, not global) -- call
    after creating axes, for figures where the extra polish is wanted.
    Purely cosmetic; safe to skip for figures with many stacked/shared
    axes where the box outline is more readable left intact.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig_publication(fig: Figure, output_path: Path, *, close: bool = True) -> Path:
    """Save ``fig`` to ``output_path`` at publication quality, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    output_path : pathlib.Path
        Destination file. Format is inferred from the extension, exactly
        as ``Figure.savefig`` does (``.png`` for raster, ``.pdf``/``.svg``
        for vector). Parent directories are created if missing.
    close : bool, default True
        Close ``fig`` after saving, freeing its memory -- matplotlib
        does not do this automatically, and every ``plot_*`` function in
        GLAS creates a fresh figure per call, so leaving this ``True``
        (the default) avoids accumulating open figures across many
        calls. Pass ``False`` if the caller still needs ``fig`` (e.g. to
        embed it elsewhere) after this call returns.

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PUBLICATION_DPI)
    if close:
        plt.close(fig)
    return output_path

"""Shared colored-dot status text and color-coded resource bars for the GUI.

Every widget that shows a connection/recording state (camera connect
status, recorder state, per-device status) or a system resource gauge
(USB bandwidth, buffer occupancy, memory, CPU, disk) goes through this
module, so the colors and thresholds stay consistent across the whole
application rather than each widget inventing its own.
"""

from __future__ import annotations

from PySide6.QtWidgets import QProgressBar

COLOR_GREEN = "#2ecc71"
"""Healthy / connected / active-and-fine."""

COLOR_RED = "#e74c3c"
"""Disconnected, error, or a resource gauge in its critical range."""

COLOR_YELLOW = "#f1c40f"
"""Transitional (connecting) or cautionary (paused, resource gauge elevated)."""

COLOR_GRAY = "#888888"
"""Idle / inactive, nothing wrong."""

_BAR_WARNING_THRESHOLD = 70.0
_BAR_CRITICAL_THRESHOLD = 90.0


def status_dot_html(color: str, text: str) -> str:
    """Build rich text for a colored status dot followed by a label.

    Parameters
    ----------
    color : str
        A CSS color, typically one of this module's ``COLOR_*`` constants.
    text : str
        The status text shown after the dot.

    Returns
    -------
    str
        HTML suitable for ``QLabel.setText()`` (Qt auto-detects HTML
        content and renders it as rich text).
    """
    return f'<span style="color: {color};">●</span> {text}'


def resource_bar_color(percent: float) -> str:
    """Pick a color for a resource gauge from how full it is.

    Parameters
    ----------
    percent : float
        A usage percentage. Values are not required to be clamped to
        ``[0, 100]`` (e.g. multi-core CPU usage can exceed 100); anything
        at or above the critical threshold reads as critical.

    Returns
    -------
    str
        :data:`COLOR_GREEN` below the warning threshold,
        :data:`COLOR_YELLOW` between the warning and critical thresholds,
        :data:`COLOR_RED` at or above the critical threshold.
    """
    if percent >= _BAR_CRITICAL_THRESHOLD:
        return COLOR_RED
    if percent >= _BAR_WARNING_THRESHOLD:
        return COLOR_YELLOW
    return COLOR_GREEN


def update_resource_bar(bar: QProgressBar, percent: float, display_text: str) -> None:
    """Set a resource gauge's fill level, color, and displayed text in one call.

    Parameters
    ----------
    bar : QProgressBar
        Expected to already have range ``(0, 100)``.
    percent : float
        Usage percentage driving both the fill level (clamped to
        ``[0, 100]`` for display -- a bar cannot render past full) and
        the color (via :func:`resource_bar_color`, using the unclamped
        value).
    display_text : str
        Text shown on the bar instead of Qt's default ``"NN%"``, e.g.
        ``"14/256 (5%)"`` or ``"3.2 GB (42%)"``.
    """
    bar.setValue(max(0, min(100, round(percent))))
    bar.setFormat(display_text)
    color = resource_bar_color(percent)
    bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

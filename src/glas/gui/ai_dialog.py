"""The missing-AI-dependency dialog shown when a YOLO/SAM2 feature is used without ``glas[ai]``.

One function, used by every widget that can trigger an AI feature
(currently :class:`~glas.gui.widgets.analysis_panel_widget.AnalysisPanelWidget`'s
Detection and Segmentation tabs) -- a single place to keep the dialog's
wording and behavior consistent, per the project requirement that a
missing AI dependency always surface as "a clear dialog explaining which
packages are missing and how to install them," not a silent failure or a
generic error string.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from glas.ai.dependencies import describe_missing_ai_packages


def show_missing_ai_dependencies_dialog(parent: QWidget | None, missing: list[str]) -> None:
    """Show a modal dialog naming every missing AI package and the install command that fixes it.

    Parameters
    ----------
    parent : QWidget, optional
        Dialog owner, for correct modality/positioning.
    missing : list of str
        Missing package names, e.g. from
        :func:`~glas.ai.dependencies.missing_ai_packages`.
    """
    QMessageBox.warning(parent, "Missing AI Dependencies", describe_missing_ai_packages(missing))

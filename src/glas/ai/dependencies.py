"""Lazy imports for GLAS's optional AI stack: ``torch``, ``ultralytics``, and ``sam2``.

None of these are hard dependencies of GLAS -- ``import glas``, the CLI,
and the GUI all work with none of them installed, the same optional-extra
pattern already used for ``PySide6`` (:mod:`glas.gui`), ``labjack-ljm``,
and ``nidaqmx`` (:mod:`glas.hardware.daq`). Every function here defers its
import to call time and raises :class:`~glas.exceptions.AIDependencyError`
with the exact ``pip install`` command that fixes it if the package is
missing, so both the CLI and the GUI can surface one consistent message.

Each function also accepts an ``_module`` override (mirroring
:class:`glas.hardware.daq.LabJackDAQ`'s ``_ljm`` parameter) so calling code
-- and tests -- can inject a stand-in module instead of the real,
heavyweight package.
"""

from __future__ import annotations

from typing import Any

from glas.exceptions import AIDependencyError

AI_EXTRA_INSTALL_HINT = 'pip install "glas[ai]"'
"""The single install command that satisfies every AI dependency at once."""

_PACKAGE_INSTALL_HINTS: dict[str, str] = {
    "torch": "pip install torch",
    "ultralytics": "pip install ultralytics",
    "sam2": "pip install sam2",
}


def _missing_dependency_message(package: str, feature: str) -> str:
    hint = _PACKAGE_INSTALL_HINTS.get(package, f"pip install {package}")
    return (
        f"{feature} requires the '{package}' package, which is not installed. "
        f"Install it with `{hint}`, or install every AI dependency at once with "
        f"`{AI_EXTRA_INSTALL_HINT}`."
    )


def import_torch(_module: Any | None = None) -> Any:
    """Import and return the ``torch`` module.

    Parameters
    ----------
    _module : optional
        Pre-imported module to return instead of importing ``torch`` --
        for tests, so this can be exercised without the real package
        installed.

    Returns
    -------
    module
        The ``torch`` module (or ``_module`` if given).

    Raises
    ------
    AIDependencyError
        If ``torch`` is not installed.
    """
    if _module is not None:
        return _module
    try:
        import torch
    except ImportError as exc:
        raise AIDependencyError(_missing_dependency_message("torch", "GLAS AI features")) from exc
    return torch


def import_ultralytics(_module: Any | None = None) -> Any:
    """Import and return the ``ultralytics`` module (YOLO detection/training).

    Parameters
    ----------
    _module : optional
        Pre-imported module to return instead of importing ``ultralytics``
        -- for tests.

    Returns
    -------
    module
        The ``ultralytics`` module (or ``_module`` if given).

    Raises
    ------
    AIDependencyError
        If ``ultralytics`` is not installed.
    """
    if _module is not None:
        return _module
    try:
        import ultralytics
    except ImportError as exc:
        raise AIDependencyError(
            _missing_dependency_message("ultralytics", "YOLO particle detection")
        ) from exc
    return ultralytics


def import_build_sam2(_func: Any | None = None) -> Any:
    """Import and return ``sam2.build_sam.build_sam2``, the SAM2 model builder.

    Parameters
    ----------
    _func : optional
        Pre-imported callable to return instead of importing ``sam2`` --
        for tests.

    Returns
    -------
    callable
        ``sam2.build_sam.build_sam2`` (or ``_func`` if given).

    Raises
    ------
    AIDependencyError
        If ``sam2`` is not installed.
    """
    if _func is not None:
        return _func
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise AIDependencyError(
            _missing_dependency_message("sam2", "SAM2 particle segmentation")
        ) from exc
    return build_sam2


def import_sam2_image_predictor(_cls: Any | None = None) -> Any:
    """Import and return ``sam2.sam2_image_predictor.SAM2ImagePredictor``.

    Parameters
    ----------
    _cls : optional
        Pre-imported class to return instead of importing ``sam2`` -- for
        tests.

    Returns
    -------
    type
        ``SAM2ImagePredictor`` (or ``_cls`` if given).

    Raises
    ------
    AIDependencyError
        If ``sam2`` is not installed.
    """
    if _cls is not None:
        return _cls
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise AIDependencyError(
            _missing_dependency_message("sam2", "SAM2 particle segmentation")
        ) from exc
    return SAM2ImagePredictor


def missing_ai_packages() -> list[str]:
    """Report which AI packages are not importable, without raising.

    Checks ``torch``, ``ultralytics``, and ``sam2`` in that order (the
    order the GUI's missing-dependency dialog lists them in, since
    ``ultralytics`` and ``sam2`` both depend on ``torch``).

    Returns
    -------
    list of str
        Package names that failed to import. Empty if every AI package is
        available.
    """
    missing: list[str] = []
    for package in ("torch", "ultralytics", "sam2"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def describe_missing_ai_packages(missing: list[str]) -> str:
    """Build a human-readable explanation of missing AI packages and how to fix it.

    Parameters
    ----------
    missing : list of str
        Package names, e.g. the result of :func:`missing_ai_packages`.

    Returns
    -------
    str
        A multi-line message naming each missing package and the install
        command that resolves all of them -- used verbatim by both the
        CLI's error output and the GUI's missing-dependency dialog.
    """
    lines = ["The following AI packages are not installed:"]
    lines.extend(f"  - {package}" for package in missing)
    lines.append("")
    lines.append(f"Install all of them with:\n    {AI_EXTRA_INSTALL_HINT}")
    return "\n".join(lines)

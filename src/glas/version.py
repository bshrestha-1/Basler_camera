"""Single source of truth for the GLAS package version.

The version follows Semantic Versioning (https://semver.org). Every other
part of the project -- ``pyproject.toml``, the CLI ``--version`` flag, and
generated dataset metadata -- reads from :data:`__version__` so the number
only has to change in one place.
"""

from __future__ import annotations

__all__ = ["__version__", "VERSION_INFO"]

__version__: str = "0.6.0"
"""Current GLAS release version (semantic versioning: MAJOR.MINOR.PATCH)."""

VERSION_INFO: tuple[int, int, int] = (0, 6, 0)
"""``__version__`` as a ``(major, minor, patch)`` tuple for programmatic use."""

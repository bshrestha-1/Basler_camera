"""Bridges Python's :mod:`logging` into a Qt signal, for the log console widget.

GLAS's backend logs through the standard library (:func:`glas.logger.get_logger`)
and knows nothing about Qt. :class:`QtLogHandler` is a plain
``logging.Handler`` that re-emits every record it receives as a Qt
signal, so :class:`~glas.gui.widgets.log_console.LogConsoleWidget` can
display camera, recording, and export events live without the backend
depending on PySide6 at all.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, SignalInstance

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _LogSignalEmitter(QObject):
    """Holds the actual Qt signal.

    Kept separate from :class:`QtLogHandler` rather than making the
    handler itself a ``QObject`` subclass: both ``logging.Handler`` and
    ``QObject`` define their own, incompatible ``emit(...)`` method, so
    inheriting from both at once creates an unresolvable name conflict.
    Composition sidesteps it entirely.
    """

    message_logged = Signal(str, str)


class QtLogHandler(logging.Handler):
    """A ``logging.Handler`` that emits :attr:`message_logged` for every record.

    Attach to the ``"glas"`` logger (the root logger every
    :func:`glas.logger.get_logger` call attaches under) to receive every
    log message the backend produces, GUI or CLI code path alike.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        self._emitter = _LogSignalEmitter()

    @property
    def message_logged(self) -> SignalInstance:
        """``Signal(level_name: str, message: str)``, emitted once per log record."""
        return self._emitter.message_logged

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001 - a formatting bug must not crash logging itself
            message = record.getMessage()
        self._emitter.message_logged.emit(record.levelname, message)

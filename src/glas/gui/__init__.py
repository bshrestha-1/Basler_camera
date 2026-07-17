"""GLAS desktop GUI (PySide6/Qt6).

A thin presentation layer over the existing GLAS backend
(:mod:`glas.camera`, :mod:`glas.controller`, :mod:`glas.experiment`,
:mod:`glas.monitor`, :mod:`glas.analysis`, :mod:`glas.accelerometer`,
:mod:`glas.hardware`) -- no acquisition, recording, analysis, or export
logic lives here; every widget delegates to the same backend classes the
CLI (:mod:`glas.cli`) already uses, so the two never drift apart::

    glas.gui.widgets    (View)       -- QWidget subclasses, no business logic
    glas.gui.viewmodels (ViewModel)  -- QObject subclasses wrapping backend
                                         classes, translating their plain
                                         Python API into Qt signals/slots
    glas.<other modules> (Model)     -- the existing, Qt-free GLAS backend

PySide6 is an optional dependency (the ``gui`` extra,
``pip install glas[gui]``): nothing outside this subpackage imports it,
so the rest of GLAS (including the CLI) works without it installed.
Import :func:`glas.gui.app.main` (not this package's own namespace) to
launch the application -- keeping PySide6 out of this file means
``import glas`` itself never requires it.
"""

from __future__ import annotations

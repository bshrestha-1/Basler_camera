"""ViewModel driving the live preview widget from a :class:`~glas.preview.Preview`.

Polls :meth:`~glas.preview.Preview.update` on a
:class:`~PySide6.QtCore.QTimer` and re-emits each new frame as a Qt
signal. :class:`~glas.preview.Preview` itself reads a
:class:`~glas.ringbuffer.RingBuffer` non-destructively (``peek()``, never
``pop()``), so attaching this to a buffer a
:class:`~glas.recorder.Recorder` is simultaneously writing from can never
slow the recording down or steal a frame the dataset writer needs -- the
same guarantee the CLI's own preview already relies on.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from glas.preview import Preview
from glas.ringbuffer import RingBuffer

DEFAULT_POLL_INTERVAL_MS = 33  # ~30 Hz UI refresh cap


class LiveFeedViewModel(QObject):
    """Polls a :class:`~glas.preview.Preview` and emits each new frame.

    Signals
    -------
    frame_ready(Frame)
        Emitted whenever a new (not-previously-seen) frame is available.
    """

    frame_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.preview: Preview | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)

    def attach(self, buffer: RingBuffer) -> None:
        """Start polling ``buffer`` for new frames via a fresh :class:`Preview`."""
        self.preview = Preview(buffer)
        self._timer.start()

    def detach(self) -> None:
        """Stop polling and release the current :class:`Preview`."""
        self._timer.stop()
        self.preview = None

    @property
    def is_attached(self) -> bool:
        """``True`` if currently polling a buffer."""
        return self.preview is not None

    def _poll(self) -> None:
        print("POLL CALLED")

        if self.preview is None:
            print("preview is None")
            return

        frame = self.preview.update()
        print("frame =", frame)

        if frame is not None:
            print("EMITTING", frame.frame_id)
            self.frame_ready.emit(frame)
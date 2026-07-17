"""High-level recording orchestration.

:class:`RecorderController` is the single entry point a CLI, GUI, or
script would use: it owns the camera connection, builds the experiment
folder and metadata for each new recording, and creates and tracks a
:class:`~glas.recorder.Recorder` for it. :meth:`RecorderController.graceful_shutdown`
adds opt-in SIGINT/SIGTERM handling so an interrupted process still
finalizes its recording safely rather than losing data.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

from glas.camera import Camera
from glas.camera_info import CameraInfo
from glas.dataset import Dataset, create_experiment_folder, resolve_dataset_format
from glas.exceptions import CameraConnectionError, RecorderError
from glas.experiment import build_experiment_extra
from glas.logger import get_logger
from glas.metadata import DatasetMetadata
from glas.recorder import Recorder, RecorderProgress, RecorderState

logger = get_logger(__name__)

DEFAULT_BUFFER_CAPACITY = 256
_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class RecorderController:
    """Owns a camera connection and orchestrates recordings on it.

    Parameters
    ----------
    base_data_dir : pathlib.Path
        Directory new experiment folders are created under (see
        :func:`glas.dataset.create_experiment_folder`).
    camera : Camera, optional
        Camera to use. A new, not-yet-connected
        :class:`~glas.camera.Camera` is created if omitted.

    Examples
    --------
    >>> controller = RecorderController(Path("~/glas_data").expanduser())  # doctest: +SKIP
    >>> controller.connect()  # doctest: +SKIP
    >>> controller.start_recording(notes="shaker at 60 Hz")  # doctest: +SKIP
    >>> time.sleep(10.0)  # doctest: +SKIP
    >>> controller.stop_recording()  # doctest: +SKIP
    >>> controller.disconnect()  # doctest: +SKIP
    """

    def __init__(self, base_data_dir: Path, camera: Camera | None = None) -> None:
        self._base_data_dir = base_data_dir
        self._camera = camera if camera is not None else Camera()
        self._current_recorder: Recorder | None = None

    @property
    def camera(self) -> Camera:
        """The camera this controller manages."""
        return self._camera

    @property
    def base_data_dir(self) -> Path:
        """Directory new experiment folders are created under."""
        return self._base_data_dir

    @base_data_dir.setter
    def base_data_dir(self, value: Path) -> None:
        self._base_data_dir = value

    def connect(self, serial_number: str | None = None) -> CameraInfo:
        """Connect the camera. See :meth:`glas.camera.Camera.connect`."""
        return self._camera.connect(serial_number=serial_number)

    def disconnect(self) -> None:
        """Disconnect the camera.

        Raises
        ------
        RecorderError
            If a recording is currently active; stop it first.
        """
        if self._is_recording_active():
            raise RecorderError("Cannot disconnect the camera while a recording is active.")
        self._camera.disconnect()

    def start_recording(
        self,
        notes: str = "",
        name: str = "",
        tags: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
        dataset_format: str = "auto",
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
    ) -> Recorder:
        """Create a new experiment folder and dataset, and start recording into it.

        Parameters
        ----------
        notes : str, default ""
            Free-text operator notes, stored in the dataset metadata.
        name : str, default ""
            Human-readable experiment name. Stored in the dataset
            metadata's ``extra`` field under a reserved key so
            :class:`~glas.experiment.ExperimentManager` can find this
            recording by name later; omitted entirely if empty.
        tags : sequence of str, optional
            Tags to attach, for the same reason as ``name``.
        extra : dict, optional
            Additional metadata fields (e.g. operator), merged in
            alongside ``name``/``tags``.
        dataset_format : {"hdf5", "raw_binary", "auto"}, default "auto"
            Storage backend; see :meth:`glas.dataset.Dataset.create`.
        buffer_capacity : int, default 256
            Ring buffer capacity for the recording's
            :class:`~glas.acquisition.Acquisition`.

        Returns
        -------
        Recorder
            The now-recording session.

        Raises
        ------
        RecorderError
            If a recording is already in progress.
        CameraConnectionError
            If the camera is not connected.
        """
        if self._is_recording_active():
            raise RecorderError("A recording is already in progress; stop it first.")
        if not self._camera.is_connected:
            raise CameraConnectionError("Camera must be connected before starting a recording.")

        info = self._camera.get_info()
        folder = create_experiment_folder(self._base_data_dir)
        resolved_format = resolve_dataset_format(dataset_format)
        metadata = DatasetMetadata(
            dataset_format=resolved_format,
            camera_model=info.model_name,
            camera_serial=info.serial_number,
            pixel_format=self._camera.pixel_format,
            width=self._camera.roi.width,
            height=self._camera.roi.height,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            exposure_time_us=self._camera.exposure_time_us,
            gain_db=self._camera.gain_db,
            notes=notes,
            extra=build_experiment_extra(name=name, tags=tags, extra=extra),
        )
        dataset = Dataset.create(folder, metadata, dataset_format=resolved_format)

        recorder = Recorder(self._camera, dataset, buffer_capacity=buffer_capacity)
        recorder.start()
        self._current_recorder = recorder
        return recorder

    def stop_recording(self) -> DatasetMetadata:
        """Stop the current recording and finalize its dataset.

        Raises
        ------
        RecorderError
            If no recording is in progress.
        """
        recorder = self._require_current_recorder()
        metadata = recorder.stop()
        self._current_recorder = None
        return metadata

    def pause_recording(self) -> None:
        """Pause the current recording. See :meth:`glas.recorder.Recorder.pause`."""
        self._require_current_recorder().pause()

    def resume_recording(self) -> None:
        """Resume the current recording. See :meth:`glas.recorder.Recorder.resume`."""
        self._require_current_recorder().resume()

    def progress(self) -> RecorderProgress | None:
        """Progress of the current recording, or ``None`` if none is active."""
        if self._current_recorder is None:
            return None
        return self._current_recorder.progress()

    def _is_recording_active(self) -> bool:
        return self._current_recorder is not None and self._current_recorder.state in (
            RecorderState.RECORDING,
            RecorderState.PAUSED,
        )

    def _require_current_recorder(self) -> Recorder:
        if self._current_recorder is None:
            raise RecorderError("No recording is in progress.")
        return self._current_recorder

    @contextmanager
    def graceful_shutdown(self) -> Iterator[threading.Event]:
        """Stop any active recording safely on SIGINT/SIGTERM, or on exit.

        Installs handlers for the duration of this ``with`` block that
        set the yielded :class:`threading.Event` rather than acting
        directly -- signal handlers should do as little as possible;
        check the event in your own loop to react to the request.
        Whether the block exits normally, via an exception, or because a
        signal was received, any recording still active when it exits is
        stopped and its dataset finalized before the previous signal
        handlers are restored.

        Yields
        ------
        threading.Event
            Set when SIGINT or SIGTERM is received inside the block.

        Examples
        --------
        >>> with controller.graceful_shutdown() as shutdown:  # doctest: +SKIP
        ...     controller.start_recording()
        ...     while not shutdown.is_set():
        ...         time.sleep(0.1)
        """
        shutdown_requested = threading.Event()

        def _handle_signal(signum: int, frame: FrameType | None) -> None:
            logger.warning("Received signal %d; requesting graceful shutdown.", signum)
            shutdown_requested.set()

        previous_handlers = {sig: signal.signal(sig, _handle_signal) for sig in _SHUTDOWN_SIGNALS}
        try:
            yield shutdown_requested
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if self._is_recording_active():
                logger.info("Finalizing active recording during graceful shutdown.")
                self.stop_recording()

"""ViewModel wrapping :class:`glas.experiment.ExperimentManager` for the dataset browser widget.

Search, delete, duplicate, and export all delegate straight to existing
backend functions (:class:`~glas.experiment.ExperimentManager`,
:func:`~glas.export.export_dataset`) -- the same ones
``glas experiment list``/``glas export`` use, so the GUI and CLI can
never disagree about what a "finalized experiment" is or how exporting
works.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from glas.exceptions import DatasetError, ExperimentNotFoundError, ExportError
from glas.experiment import ExperimentManager, ExperimentSummary
from glas.export import ExportFormat, export_dataset


class DatasetViewModel(QObject):
    """Lists, searches, deletes, duplicates, and exports recorded experiments.

    Signals
    -------
    experiments_changed(list)
        Emitted after :meth:`refresh` with the current
        ``list[ExperimentSummary]``.
    export_finished(object)
        Emitted after a successful :meth:`export`, with the resulting
        :class:`~glas.export.ExportResult`.
    error_occurred(str)
        Emitted instead of the corresponding signal above when an
        operation fails.
    """

    experiments_changed = Signal(list)
    export_finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, base_data_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._manager = ExperimentManager(base_data_dir)

    @property
    def manager(self) -> ExperimentManager:
        """The underlying manager, for widgets that need direct read access."""
        return self._manager

    def refresh(
        self,
        *,
        name_contains: str | None = None,
        tag: str | None = None,
        camera_model: str | None = None,
    ) -> None:
        """Re-list experiments (optionally filtered), emitting :attr:`experiments_changed`."""
        experiments = self._manager.search_experiments(
            name_contains=name_contains, tag=tag, camera_model=camera_model
        )
        self.experiments_changed.emit(experiments)

    def delete(self, run_id: str) -> None:
        """Delete an experiment, then re-emit the refreshed list."""
        try:
            self._manager.delete_experiment(run_id)
        except ExperimentNotFoundError as exc:
            self.error_occurred.emit(str(exc))
            return
        self.refresh()

    def duplicate(self, run_id: str, *, new_name: str = "") -> ExperimentSummary | None:
        """Duplicate an experiment, then re-emit the refreshed list.

        Returns
        -------
        ExperimentSummary or None
            The new copy's summary, or ``None`` if the operation failed
            (in which case :attr:`error_occurred` was emitted instead).
        """
        try:
            copy = self._manager.duplicate_experiment(run_id, new_name=new_name)
        except ExperimentNotFoundError as exc:
            self.error_occurred.emit(str(exc))
            return None
        self.refresh()
        return copy

    def export(
        self,
        run_id: str,
        output_path: Path,
        export_format: ExportFormat,
        *,
        fps: float = 30.0,
        start_frame: int | None = None,
        end_frame: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Export an experiment, emitting :attr:`export_finished` or :attr:`error_occurred`."""
        try:
            summary = self._manager.get_experiment(run_id)
            result = export_dataset(
                summary.folder,
                output_path,
                export_format,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                overwrite=overwrite,
            )
        except (ExperimentNotFoundError, ExportError, DatasetError) as exc:
            self.error_occurred.emit(str(exc))
            return
        self.export_finished.emit(result)

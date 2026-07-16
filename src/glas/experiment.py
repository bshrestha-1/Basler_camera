"""Experiment-level organization across many recordings.

Earlier phases already cover automatic per-recording folder numbering
(:func:`glas.dataset.create_experiment_folder`) and a canonical metadata
record for each one (:class:`glas.metadata.DatasetMetadata`).
:class:`ExperimentManager` sits above both: it builds a searchable index
across every recording under a base data directory, and defines a
convention -- reserved keys inside :attr:`~glas.metadata.DatasetMetadata.extra`
-- for attaching a human-readable name and tags to a recording at the
point it's created.

This deliberately does not change :class:`~glas.metadata.DatasetMetadata`'s
schema: ``extra`` has carried this exact forward-compatibility promise
since Phase 4 (see its docstring), so introducing a second metadata file,
or new top-level fields that every already-recorded dataset would be
missing, isn't necessary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from glas.dataset import create_experiment_folder
from glas.exceptions import DatasetError, ExperimentNotFoundError, JSONValidationError
from glas.logger import get_logger
from glas.metadata import DatasetMetadata, load_metadata_json

logger = get_logger(__name__)

#: Reserved glas.metadata.DatasetMetadata.extra keys ExperimentManager reads
#: and writes. Any other code populating extra is free to add its own keys
#: alongside these -- extra is otherwise unstructured.
NAME_KEY = "experiment_name"
TAGS_KEY = "experiment_tags"

_METADATA_FILENAME = "metadata.json"


class ExperimentSummary(BaseModel):
    """A lightweight, searchable summary of one recorded experiment.

    Attributes
    ----------
    folder : pathlib.Path
        The experiment's folder.
    run_id : str
        Folder name, e.g. ``"Run0001"``.
    name : str
        Human-readable experiment name, from
        ``DatasetMetadata.extra["experiment_name"]``. Empty if never set.
    tags : list of str
        Tags attached to the experiment, from
        ``DatasetMetadata.extra["experiment_tags"]``. Empty if never set.
    notes : str
        Free-text operator notes (``DatasetMetadata.notes``).
    created_at_utc : str
        ISO 8601 UTC timestamp the recording started.
    frame_count : int
        Number of frames in the recording.
    camera_model : str
        Camera model that captured the recording.
    metadata : DatasetMetadata
        The full underlying metadata, for anything not surfaced above.
    """

    model_config = ConfigDict(frozen=True)

    folder: Path
    run_id: str
    name: str
    tags: list[str]
    notes: str
    created_at_utc: str
    frame_count: int
    camera_model: str
    metadata: DatasetMetadata


def build_experiment_extra(
    name: str = "",
    tags: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``DatasetMetadata.extra`` dict carrying experiment name/tags.

    Pass the result to ``DatasetMetadata(extra=...)`` (or
    ``RecorderController.start_recording(extra=...)``) so
    :meth:`ExperimentManager.list_experiments` and
    :meth:`ExperimentManager.search_experiments` can find the recording by
    name or tag later.

    Parameters
    ----------
    name : str, default ""
        Human-readable experiment name. Omitted from the result if empty.
    tags : sequence of str, optional
        Tags to attach. Omitted from the result if empty or ``None``.
    extra : dict, optional
        Additional caller-supplied fields to merge in alongside the
        reserved name/tags keys.

    Returns
    -------
    dict
        ``extra``, with :data:`NAME_KEY`/:data:`TAGS_KEY` added if
        ``name``/``tags`` were given.
    """
    merged = dict(extra or {})
    if name:
        merged[NAME_KEY] = name
    if tags:
        merged[TAGS_KEY] = list(tags)
    return merged


def _summarize(folder: Path, metadata: DatasetMetadata) -> ExperimentSummary:
    tags = metadata.extra.get(TAGS_KEY, [])
    return ExperimentSummary(
        folder=folder,
        run_id=folder.name,
        name=str(metadata.extra.get(NAME_KEY, "")),
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
        notes=metadata.notes,
        created_at_utc=metadata.created_at_utc,
        frame_count=metadata.frame_count,
        camera_model=metadata.camera_model,
        metadata=metadata,
    )


def _load_summary(folder: Path) -> ExperimentSummary | None:
    metadata_path = folder / _METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        metadata = load_metadata_json(metadata_path)
    except (DatasetError, JSONValidationError):
        logger.warning("Skipping %s: %s could not be read.", folder, _METADATA_FILENAME)
        return None
    return _summarize(folder, metadata)


class ExperimentManager:
    """Creates, organizes, and searches recordings under a base data directory.

    Parameters
    ----------
    base_data_dir : pathlib.Path
        Directory experiment folders are created under and scanned from.
        Does not need to exist yet -- :meth:`new_folder` creates it.

    Examples
    --------
    >>> manager = ExperimentManager(Path("~/glas_data").expanduser())  # doctest: +SKIP
    >>> extra = build_experiment_extra(name="shaker sweep", tags=["brazil-nut", "60hz"])
    >>> controller.start_recording(extra=extra)  # doctest: +SKIP
    >>> [s.name for s in manager.search_experiments(tag="brazil-nut")]  # doctest: +SKIP
    ['shaker sweep']
    """

    def __init__(self, base_data_dir: Path) -> None:
        self._base_data_dir = base_data_dir

    @property
    def base_data_dir(self) -> Path:
        """Directory experiment folders are created under and scanned from."""
        return self._base_data_dir

    def new_folder(self, prefix: str = "Run", width: int = 4) -> Path:
        """Create the next automatically numbered experiment folder.

        Thin wrapper around :func:`glas.dataset.create_experiment_folder`,
        exposed here too so callers working through ``ExperimentManager``
        don't need a second import for it.
        """
        return create_experiment_folder(self._base_data_dir, prefix=prefix, width=width)

    def list_experiments(self) -> list[ExperimentSummary]:
        """List every finalized experiment under :attr:`base_data_dir`.

        Sorted by folder name (which sorts chronologically for the
        default zero-padded ``RunNNNN`` naming). Folders with no readable
        ``metadata.json`` -- still recording, or abandoned before any
        frames were finalized -- are skipped rather than raising.

        Returns
        -------
        list of ExperimentSummary
        """
        if not self._base_data_dir.is_dir():
            return []

        summaries = []
        for folder in sorted(self._base_data_dir.iterdir()):
            if not folder.is_dir():
                continue
            summary = _load_summary(folder)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def search_experiments(
        self,
        *,
        name_contains: str | None = None,
        tag: str | None = None,
        camera_model: str | None = None,
    ) -> list[ExperimentSummary]:
        """Filter :meth:`list_experiments` by name, tag, and/or camera model.

        Parameters
        ----------
        name_contains : str, optional
            Case-insensitive substring match against
            :attr:`ExperimentSummary.name`.
        tag : str, optional
            Only experiments with this exact tag in
            :attr:`ExperimentSummary.tags`.
        camera_model : str, optional
            Only experiments recorded with this camera model.

        Returns
        -------
        list of ExperimentSummary
            Every filter given must match (a logical AND); omitting all
            three returns the same as :meth:`list_experiments`.
        """
        results = self.list_experiments()
        if name_contains is not None:
            needle = name_contains.lower()
            results = [s for s in results if needle in s.name.lower()]
        if tag is not None:
            results = [s for s in results if tag in s.tags]
        if camera_model is not None:
            results = [s for s in results if s.camera_model == camera_model]
        return results

    def get_experiment(self, run_id: str) -> ExperimentSummary:
        """Look up one experiment by its folder name.

        Parameters
        ----------
        run_id : str
            Folder name, e.g. ``"Run0001"``.

        Returns
        -------
        ExperimentSummary

        Raises
        ------
        ExperimentNotFoundError
            If ``run_id`` doesn't exist under :attr:`base_data_dir`, or
            has no readable ``metadata.json`` (not yet finalized).
        """
        summary = _load_summary(self._base_data_dir / run_id)
        if summary is None:
            raise ExperimentNotFoundError(
                f"No finalized experiment {run_id!r} found under {self._base_data_dir}."
            )
        return summary

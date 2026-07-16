# Experiment Manager

Phase 9 sits above `Dataset`/`RecorderController`: it builds a searchable
index across every recording under a base data directory, and defines a
convention for attaching a human-readable name and tags to a recording at
the point it's created.

```
RecorderController.start_recording(name=..., tags=[...])
                        |
                        v
        DatasetMetadata.extra["experiment_name"/"experiment_tags"]
                        |
                        v
        ExperimentManager.list_experiments() / search_experiments() / get_experiment()
```

This deliberately does not change `DatasetMetadata`'s schema.
`extra` has carried this exact forward-compatibility promise since Phase 4
(see its docstring in `glas.metadata`) -- introducing a second metadata
file, or new top-level fields every already-recorded dataset would be
missing, isn't necessary. `glas.experiment` just defines two reserved keys
inside `extra` (`experiment_name`, `experiment_tags`) and a manager that
knows to look for them.

## Quickstart

```python
from pathlib import Path
from glas.controller import RecorderController
from glas.experiment import ExperimentManager

controller = RecorderController(Path("~/glas_data").expanduser())
controller.connect()
controller.start_recording(name="shaker sweep", tags=["brazil-nut", "60hz"])
# ... record, then stop_recording() ...
controller.disconnect()

manager = ExperimentManager(Path("~/glas_data").expanduser())
for summary in manager.search_experiments(tag="brazil-nut"):
    print(summary.run_id, summary.name, summary.frame_count)
```

`RecorderController.start_recording()` gained `name`/`tags` parameters
that build the reserved `extra` keys automatically -- callers don't need
to import `glas.experiment` themselves just to record something
discoverable later. Use `build_experiment_extra()` directly if you're
constructing `DatasetMetadata` yourself, without going through
`RecorderController`.

## `ExperimentManager`

- `new_folder(prefix="Run", width=4)` -- thin wrapper around
  `glas.dataset.create_experiment_folder()`, so `ExperimentManager` alone
  covers both halves of this phase's job (folders and metadata) without a
  second import.
- `list_experiments()` -- every finalized experiment under
  `base_data_dir`, in folder order. A folder with no readable
  `metadata.json` -- still recording, or abandoned before any frames were
  finalized -- is skipped, not an error; a folder whose `metadata.json`
  fails to parse is logged and skipped the same way, rather than making
  one bad dataset break the whole listing.
- `search_experiments(name_contains=..., tag=..., camera_model=...)` --
  filters `list_experiments()`. Every filter given must match (a logical
  AND); passing none returns the same as `list_experiments()`.
- `get_experiment(run_id)` -- look up one experiment by folder name (e.g.
  `"Run0001"`). Raises `ExperimentNotFoundError` if it doesn't exist or
  isn't finalized yet.

Each result is an `ExperimentSummary`: `folder`, `run_id`, `name`, `tags`,
`notes`, `created_at_utc`, `frame_count`, `camera_model`, and the full
underlying `metadata` for anything not surfaced directly.

## Testing

All of `glas.experiment` is tested against real `tmp_path` datasets (no
camera needed) -- folder creation, listing (including skipping
in-progress and corrupt-metadata folders), every search filter alone and
combined, and both success and not-found paths for `get_experiment()`.
`RecorderController.start_recording(name=..., tags=...)` is tested against
pypylon's emulated camera to confirm the reserved `extra` keys actually
land in the finalized dataset's metadata, and that omitting them leaves
`extra` exactly as it was before this phase (no surprise keys added to
existing recordings that don't use names/tags).

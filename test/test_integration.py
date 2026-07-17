"""End-to-end integration test spanning every phase together.

Each phase's own test module already exercises that phase in isolation
against a real (emulated) camera or a real dataset on disk. This module
proves the whole system works *together*, in the shape an actual lab
session would use it: connect, record (with a live preview and a
performance monitor both attached to the same in-progress recording),
stop, validate, read frames back, export, track particles, and find the
recording again through the experiment manager.

Runs against pypylon's built-in camera emulation transport layer
(PYLON_CAMEMU, set in conftest.py) rather than physical hardware. If
pypylon is not installed, or no emulated/real camera is reachable in this
environment, the whole module is skipped.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pypylon = pytest.importorskip("pypylon")

from glas.analysis import track_dataset  # noqa: E402
from glas.camera_info import detect_cameras  # noqa: E402
from glas.camera_validator import ROI  # noqa: E402
from glas.controller import RecorderController  # noqa: E402
from glas.dataset import iter_frames, validate_dataset  # noqa: E402
from glas.experiment import ExperimentManager  # noqa: E402
from glas.export import export_dataset  # noqa: E402
from glas.monitor import PerformanceMonitor  # noqa: E402
from glas.preview import Preview  # noqa: E402

_cameras = detect_cameras()
if not _cameras:
    pytest.skip(
        "No Basler camera (real or emulated) detected in this environment.",
        allow_module_level=True,
    )

_FRAME_WIDTH = 64
_FRAME_HEIGHT = 48


def test_full_pipeline_record_preview_monitor_export_and_find(tmp_path: Path) -> None:
    controller = RecorderController(tmp_path)
    controller.connect()
    try:
        controller.camera.roi = ROI(
            width=_FRAME_WIDTH, height=_FRAME_HEIGHT, offset_x=0, offset_y=0
        )

        recorder = controller.start_recording(
            name="integration sweep",
            tags=["brazil-nut", "integration-test"],
            notes="full-pipeline smoke test",
        )

        # A live preview and a performance monitor both attach to the same
        # in-progress recording's buffer -- the whole point of Phase 6/7's
        # peek()/stats()-only design is that neither can ever interfere
        # with the dataset writer or with each other.
        preview = Preview(recorder.buffer)
        monitor = PerformanceMonitor(recorder.buffer, data_dir=recorder.dataset.folder)

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            preview.update()
            monitor.sample()
            time.sleep(0.01)

        assert preview.fps() >= 0.0  # exercised without raising; frames may or may not have ticked
        snapshot = monitor.sample()
        assert snapshot.cpu_percent >= 0.0
        assert snapshot.disk_free_gb >= 0.0

        metadata = controller.stop_recording()
    finally:
        controller.disconnect()

    # The dataset writer's output is authoritative and unaffected by the
    # concurrent, non-destructive preview/monitor readers above.
    assert metadata.frame_count > 0
    folder = recorder.dataset.folder
    buffer_stats = recorder.buffer.stats()
    assert metadata.frame_count + buffer_stats.dropped == buffer_stats.pushed

    # Phase 4: on-disk integrity.
    validation = validate_dataset(folder)
    assert validation.valid, validation.errors

    # Phase 8: read every frame back, in order, with no gaps.
    frames = list(iter_frames(folder))
    assert len(frames) == metadata.frame_count
    assert [frame.frame_id for frame in frames] == list(range(metadata.frame_count))
    assert all(frame.image.shape == (_FRAME_HEIGHT, _FRAME_WIDTH) for frame in frames)

    # Phase 8: export to both an image sequence and a video format.
    png_result = export_dataset(folder, tmp_path / "frames_png", "png")
    assert png_result.frame_count == metadata.frame_count
    assert len(list((tmp_path / "frames_png").glob("*.png"))) == metadata.frame_count

    mp4_result = export_dataset(folder, tmp_path / "clip.mp4", "mp4", fps=15.0)
    assert mp4_result.frame_count == metadata.frame_count
    assert (tmp_path / "clip.mp4").is_file()

    # Phase 11: detect and track particles across the recording. The
    # emulated camera's test pattern isn't real granular media, so this
    # only proves the pipeline runs end to end without raising -- not
    # that any particular number of particles is found.
    history = track_dataset(folder)
    assert isinstance(history, dict)
    for observations in history.values():
        assert all(0 <= obs.frame_id < metadata.frame_count for obs in observations)

    # Phase 9: find the same recording back by tag and by name.
    manager = ExperimentManager(tmp_path)
    by_tag = manager.search_experiments(tag="brazil-nut")
    assert [s.run_id for s in by_tag] == [folder.name]

    summary = manager.get_experiment(folder.name)
    assert summary.name == "integration sweep"
    assert summary.tags == ["brazil-nut", "integration-test"]
    assert summary.notes == "full-pipeline smoke test"
    assert summary.frame_count == metadata.frame_count

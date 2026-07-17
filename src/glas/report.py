"""Self-contained HTML experiment reports: every analysis, in one file, for one recording.

    glas.metadata.load_metadata_json() -> generate_report() -> report.html

Runs every analysis GLAS has for a recording (tracking, Brazil nut,
convection, packing, segregation, and optionally vibration if an
accelerometer CSV is given), embeds each one's summary statistics and
plot (base64-encoded directly into the HTML -- no separate image files
to lose track of) into a single portable file. Nothing here computes
anything new: every number and every plot comes straight from the exact
same :mod:`glas.analysis`/:mod:`glas.accelerometer` functions the CLI and
GUI already use, styled through :mod:`glas.plotting` the same way.

An individual analysis failing (too few particles for Brazil nut, too
few frames for convection, etc.) does not abort the report -- that
section is shown as "Skipped: <reason>" instead, since the operator still
wants the report for everything that *did* work. Only a total failure
(the dataset's own frames can't be read at all) raises
:class:`~glas.exceptions.ReportError`, since nothing in the report would
be meaningful in that case.
"""

from __future__ import annotations

import base64
import html
import tempfile
from dataclasses import dataclass
from pathlib import Path

from glas.accelerometer import analyze_vibration
from glas.analysis import (
    analyze_brazil_nut,
    analyze_convection,
    analyze_packing,
    analyze_segregation,
    plot_packing_summary,
    plot_velocity_heatmap,
    track_dataset,
)
from glas.analysis.particle_tracking import DEFAULT_MAX_DISTANCE, DEFAULT_MAX_GAP
from glas.analysis.tracking_utils import DEFAULT_MIN_AREA
from glas.exceptions import GLASError, ReportError
from glas.experiment import NAME_KEY, TAGS_KEY, PhysicalParameters, get_physical_parameters
from glas.metadata import DatasetMetadata, load_metadata_json

_METADATA_FILENAME = "metadata.json"


@dataclass
class _ReportSection:
    title: str
    body_html: str


def _image_tag(path: Path, alt: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}" '
        f'style="max-width:100%;">'
    )


def _paragraph(lines: list[str]) -> str:
    return "<p>" + "<br>".join(html.escape(line) for line in lines) + "</p>"


def _skipped(title: str, exc: Exception) -> _ReportSection:
    return _ReportSection(title, f"<p><em>Skipped: {html.escape(str(exc))}</em></p>")


def _tracking_section(
    folder: Path, *, max_distance: float, max_gap: int, min_area: float
) -> _ReportSection:
    try:
        history = track_dataset(
            folder, max_distance=max_distance, max_gap=max_gap, min_area=min_area
        )
    except GLASError as exc:
        raise ReportError(f"Could not read frames from {folder}: {exc}") from exc

    if not history:
        return _ReportSection("Tracking", "<p><em>No particles detected.</em></p>")

    lengths = [len(observations) for observations in history.values()]
    lines = [
        f"Tracks: {len(history)}",
        f"Mean track length: {sum(lengths) / len(lengths):.1f} frame(s)",
        f"Longest track: {max(lengths)} frame(s)",
    ]
    return _ReportSection("Tracking", _paragraph(lines))


def _brazil_nut_section(
    folder: Path, tmp_dir: Path, *, max_distance: float, max_gap: int, min_area: float
) -> _ReportSection:
    plot_path = tmp_dir / "brazil_nut.png"
    try:
        trajectory = analyze_brazil_nut(
            folder,
            max_distance=max_distance,
            max_gap=max_gap,
            min_area=min_area,
            plot_path=plot_path,
        )
    except GLASError as exc:
        return _skipped("Brazil Nut", exc)

    lines = [f"Track ID: {trajectory.track_id}", f"Observations: {len(trajectory.frame_ids)}"]
    if trajectory.rise_time_s is not None:
        lines.append(f"Rise time: {trajectory.rise_time_s:.2f} s")
    if trajectory.heights_px:
        lines.append(f"Final height: {trajectory.heights_px[-1]:.1f} px")
    return _ReportSection(
        "Brazil Nut", _paragraph(lines) + _image_tag(plot_path, "Brazil nut trajectory")
    )


def _convection_section(folder: Path, tmp_dir: Path) -> _ReportSection:
    try:
        summary = analyze_convection(folder)
    except GLASError as exc:
        return _skipped("Convection", exc)

    lines = [f"Frame pairs: {len(summary.frame_ids)}"]
    image_html = ""
    if summary.circulations:
        lines.append(f"Final circulation: {summary.circulations[-1]:.2f} px^2/s")
        mean_circulation = sum(summary.circulations) / len(summary.circulations)
        lines.append(f"Mean circulation: {mean_circulation:.2f} px^2/s")
    if summary.fields:
        plot_path = tmp_dir / "convection.png"
        plot_velocity_heatmap(summary.fields[-1], plot_path)
        image_html = _image_tag(plot_path, "Velocity field")
    return _ReportSection("Convection", _paragraph(lines) + image_html)


def _packing_section(folder: Path, tmp_dir: Path, *, min_area: float) -> _ReportSection:
    try:
        summary = analyze_packing(folder, min_area=min_area)
    except GLASError as exc:
        return _skipped("Packing", exc)

    lines = [f"Frames: {len(summary.frame_ids)}"]
    if summary.metrics:
        final = summary.metrics[-1]
        lines.append(f"Final packing fraction: {final.packing_fraction:.3f}")
        lines.append(f"Final particle count: {final.particle_count}")
    plot_path = tmp_dir / "packing.png"
    plot_packing_summary(summary, plot_path)
    return _ReportSection("Packing", _paragraph(lines) + _image_tag(plot_path, "Packing summary"))


def _segregation_section(folder: Path, tmp_dir: Path, *, min_area: float) -> _ReportSection:
    plot_path = tmp_dir / "segregation.png"
    try:
        summary = analyze_segregation(folder, min_area=min_area, plot_path=plot_path)
    except GLASError as exc:
        return _skipped("Segregation", exc)

    lines = [f"Frames: {len(summary.frame_ids)}"]
    if summary.metrics:
        final = summary.metrics[-1]
        lines.append(f"Final segregation index: {final.segregation_index:.3f}")
        lines.append(f"Final mixing entropy: {final.mixing_entropy:.3f}")
    return _ReportSection(
        "Segregation", _paragraph(lines) + _image_tag(plot_path, "Segregation summary")
    )


def _vibration_section(accelerometer_csv: Path, tmp_dir: Path) -> _ReportSection:
    plot_path = tmp_dir / "vibration.png"
    try:
        metrics = analyze_vibration(accelerometer_csv, plot_path=plot_path)
    except GLASError as exc:
        return _skipped("Vibration", exc)

    lines = [
        f"Frequency: {metrics.frequency_hz:.2f} Hz",
        f"Amplitude: {metrics.amplitude_m * 1000:.3f} mm",
        f"Gamma: {metrics.gamma:.2f}",
        f"Peak acceleration: {metrics.peak_acceleration_g:.2f} g",
    ]
    return _ReportSection(
        "Vibration", _paragraph(lines) + _image_tag(plot_path, "Accelerometer signal")
    )


def _metadata_rows(folder: Path, metadata: DatasetMetadata) -> list[tuple[str, str]]:
    rows = [
        ("Run", folder.name),
        ("Camera", f"{metadata.camera_model} (serial {metadata.camera_serial})"),
        ("Pixel format", metadata.pixel_format),
        ("Frame size", f"{metadata.width} x {metadata.height}"),
        ("ROI offset", f"({metadata.roi_offset_x}, {metadata.roi_offset_y})"),
        ("Frame count", str(metadata.frame_count)),
        ("Created (UTC)", metadata.created_at_utc),
        ("GLAS version", metadata.glas_version),
    ]
    if metadata.exposure_time_us is not None:
        rows.append(("Exposure", f"{metadata.exposure_time_us:.1f} us"))
    if metadata.gain_db is not None:
        rows.append(("Gain", f"{metadata.gain_db:.2f} dB"))
    if metadata.frame_rate_hz is not None:
        rows.append(("Frame rate", f"{metadata.frame_rate_hz:.2f} Hz"))
    if metadata.camera_settings:
        settings_text = ", ".join(
            f"{key}={value}" for key, value in sorted(metadata.camera_settings.items())
        )
        rows.append(("Camera settings", settings_text))

    tags = metadata.extra.get(TAGS_KEY, [])
    if isinstance(tags, list) and tags:
        rows.append(("Tags", ", ".join(str(tag) for tag in tags)))
    if metadata.notes:
        rows.append(("Notes", metadata.notes))

    physical = get_physical_parameters(metadata)
    if physical != PhysicalParameters():
        for field_name, value in physical.model_dump().items():
            if value:
                rows.append((field_name.replace("_", " ").title(), str(value)))

    return rows


def _render_html(folder: Path, metadata: DatasetMetadata, sections: list[_ReportSection]) -> str:
    experiment_name = str(metadata.extra.get(NAME_KEY, ""))
    title = folder.name + (f" -- {experiment_name}" if experiment_name else "")

    metadata_html = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in _metadata_rows(folder, metadata)
    )
    sections_html = "".join(
        f"<section><h2>{html.escape(section.title)}</h2>{section.body_html}</section>"
        for section in sections
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
  color: #1a1a1a; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
h2 {{ margin-top: 2rem; }}
table {{ border-collapse: collapse; margin-bottom: 1rem; }}
th, td {{ text-align: left; padding: 0.25rem 1rem 0.25rem 0; vertical-align: top; }}
th {{ font-weight: 600; color: #555; }}
img {{ display: block; margin-top: 0.5rem; border: 1px solid #ddd; }}
section {{ border-top: 1px solid #eee; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<table>{metadata_html}</table>
{sections_html}
</body>
</html>
"""


def generate_report(
    folder: Path,
    output_path: Path,
    *,
    accelerometer_csv: Path | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    max_gap: int = DEFAULT_MAX_GAP,
    min_area: float = DEFAULT_MIN_AREA,
) -> Path:
    """Generate a self-contained HTML report covering every analysis for one recording.

    Parameters
    ----------
    folder : pathlib.Path
        A finalized dataset folder.
    output_path : pathlib.Path
        Destination HTML file. Parent directories are created if
        missing.
    accelerometer_csv : pathlib.Path, optional
        If given, an accelerometer CSV to include a Vibration section
        for (see :func:`glas.accelerometer.analyze_vibration`). Omitted
        entirely if not given -- there is no way to auto-discover an
        accelerometer recording from a dataset folder alone.
    max_distance, max_gap, min_area : see
        :func:`glas.analysis.track_dataset`/:func:`~glas.analysis.tracking_utils.detect_particles`.
        Used for every particle-detection-based section (Tracking,
        Brazil Nut, Packing, Segregation).

    Returns
    -------
    pathlib.Path
        ``output_path``, for chaining.

    Raises
    ------
    ReportError
        If the dataset's metadata or frames cannot be read at all. An
        individual analysis failing partway through (too few particles,
        too few frames, etc.) does not raise this -- that section is
        shown as skipped instead.
    """
    try:
        metadata = load_metadata_json(folder / _METADATA_FILENAME)
    except GLASError as exc:
        raise ReportError(f"Could not read metadata for {folder}: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        sections = [
            _tracking_section(
                folder, max_distance=max_distance, max_gap=max_gap, min_area=min_area
            ),
            _brazil_nut_section(
                folder, tmp_dir, max_distance=max_distance, max_gap=max_gap, min_area=min_area
            ),
            _convection_section(folder, tmp_dir),
            _packing_section(folder, tmp_dir, min_area=min_area),
            _segregation_section(folder, tmp_dir, min_area=min_area),
        ]
        if accelerometer_csv is not None:
            sections.append(_vibration_section(accelerometer_csv, tmp_dir))

        html_text = _render_html(folder, metadata, sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path

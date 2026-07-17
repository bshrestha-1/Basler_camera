"""GLAS analysis engine.

Detects and tracks particles across a recorded dataset::

    glas.dataset.iter_frames() -> detect_particles() -> ParticleTracker -> trajectories

This is the foundation later analysis phases (Brazil nut effect,
convection, packing, segregation) build their own measurements on top of.
"""

from __future__ import annotations

from glas.analysis.brazil_nut import (
    BrazilNutTrajectory,
    analyze_brazil_nut,
    compute_brazil_nut_trajectory,
    identify_brazil_nut,
    plot_brazil_nut_trajectory,
)
from glas.analysis.convection import (
    ConvectionSummary,
    VelocityField,
    analyze_convection,
    compute_optical_flow,
    compute_vorticity,
    plot_velocity_heatmap,
    total_circulation,
)
from glas.analysis.packing import (
    PackingField,
    PackingMetrics,
    PackingSummary,
    analyze_packing,
    compute_packing_field,
    compute_packing_metrics,
    plot_packing_heatmap,
    plot_packing_summary,
)
from glas.analysis.particle_tracking import (
    ParticleTracker,
    TrackedParticle,
    export_tracks_csv,
    track_dataset,
)
from glas.analysis.segregation import (
    SegregationMetrics,
    SegregationSummary,
    analyze_segregation,
    compute_segregation_metrics,
    plot_segregation_summary,
)
from glas.analysis.tracking_utils import Detection, detect_particles, link_nearest

__all__ = [
    "Detection",
    "detect_particles",
    "link_nearest",
    "ParticleTracker",
    "TrackedParticle",
    "track_dataset",
    "export_tracks_csv",
    "BrazilNutTrajectory",
    "identify_brazil_nut",
    "compute_brazil_nut_trajectory",
    "plot_brazil_nut_trajectory",
    "analyze_brazil_nut",
    "VelocityField",
    "ConvectionSummary",
    "compute_optical_flow",
    "compute_vorticity",
    "total_circulation",
    "plot_velocity_heatmap",
    "analyze_convection",
    "PackingMetrics",
    "PackingSummary",
    "PackingField",
    "compute_packing_metrics",
    "compute_packing_field",
    "plot_packing_heatmap",
    "plot_packing_summary",
    "analyze_packing",
    "SegregationMetrics",
    "SegregationSummary",
    "compute_segregation_metrics",
    "plot_segregation_summary",
    "analyze_segregation",
]

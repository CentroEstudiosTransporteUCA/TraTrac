"""Convert authored exclusion zones into global-frame polygons for masking.

Each zone is authored on a reference frame; this pushes its vertices through that
frame's ego-motion pose once, so the runtime ``DetectionMask`` only ever maps
global -> current-raw. For a static run every pose is the identity and the global
polygons equal the authored raw polygons. See vault/21_exclusion_zones.md.
"""

from __future__ import annotations

from collections.abc import Callable

from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.geometry import Point2D, Transform2D


def to_global_polygons(
	zones: ExclusionZones, pose_for: Callable[[int], Transform2D]
) -> tuple[tuple[Point2D, ...], ...]:
	"""Map each zone's polygon from its reference frame into the global frame.

	``pose_for(reference_frame)`` returns that frame's pose (raw -> global).
	"""
	return tuple(
		tuple(pose_for(zone.reference_frame).apply(v) for v in zone.polygon.vertices)
		for zone in zones.zones
	)

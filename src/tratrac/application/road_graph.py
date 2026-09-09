"""Convert authored link zones into global-frame polygons and classify points by link.

Mirrors ``application/exclusion.py``'s reference-frame -> global-frame conversion,
but classification is per-observation (a vehicle can cross links mid-track, and
SSAM's Link ID is a per-VEHICLE-RECORD field, i.e. per frame) rather than the
track-level majority vote ``excluded_track_ids`` uses. See
``docs/IMPLEMENTATION_PLAN.md`` Group C1.
"""

from __future__ import annotations

from collections.abc import Callable

from tratrac.domain.geometry import Point2D, Transform2D, point_in_polygon
from tratrac.domain.road_graph import LinkZones

# One link zone's global-frame polygon, paired with its label.
GlobalLinkPolygon = tuple[int, tuple[Point2D, ...]]


def to_global_link_polygons(
	zones: LinkZones, pose_for: Callable[[int], Transform2D]
) -> tuple[GlobalLinkPolygon, ...]:
	"""Map each zone's polygon from its reference frame into the global frame.

	``pose_for(reference_frame)`` returns that frame's pose (raw -> global).
	Order is preserved from ``zones`` — first match wins on overlapping zones.
	"""
	return tuple(
		(
			zone.link_id,
			tuple(pose_for(zone.reference_frame).apply(v) for v in zone.polygon.vertices),
		)
		for zone in zones.zones
	)


def link_id_for_point(point: Point2D, global_link_polygons: tuple[GlobalLinkPolygon, ...]) -> int:
	"""The link id of the first zone containing ``point``, or ``0`` (unassigned) if none.

	``0`` matches ``VehicleState.link_id``'s existing "unknown" sentinel default.
	"""
	for link_id, polygon in global_link_polygons:
		if point_in_polygon(point, polygon):
			return link_id
	return 0

"""Convert authored link/lane zones into global-frame polygons and classify points.

Mirrors ``application/exclusion.py``'s reference-frame -> global-frame conversion,
but classification is per-observation (a vehicle can cross links/lanes mid-track,
and SSAM's Link ID / Lane ID are per-VEHICLE-RECORD fields, i.e. per frame) rather
than the track-level majority vote ``excluded_track_ids`` uses. See
``docs/IMPLEMENTATION_PLAN.md`` Groups C1/C2.
"""

from __future__ import annotations

from collections.abc import Callable

from tratrac.domain.geometry import Point2D, Transform2D, point_in_polygon
from tratrac.domain.road_graph import LaneZones, LinkZones

# One zone's global-frame polygon, paired with its label (link_id or lane_id).
GlobalLabeledPolygon = tuple[int, tuple[Point2D, ...]]
GlobalLinkPolygon = GlobalLabeledPolygon
GlobalLanePolygon = GlobalLabeledPolygon


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
	return _label_for_point(point, global_link_polygons)


def to_global_lane_polygons(
	zones: LaneZones, pose_for: Callable[[int], Transform2D]
) -> tuple[GlobalLanePolygon, ...]:
	"""Map each lane zone's polygon from its reference frame into the global frame.

	Mirrors ``to_global_link_polygons``; the label carried is ``lane_id``, not ``link_id``.
	"""
	return tuple(
		(
			zone.lane_id,
			tuple(pose_for(zone.reference_frame).apply(v) for v in zone.polygon.vertices),
		)
		for zone in zones.zones
	)


def lane_id_for_point(point: Point2D, global_lane_polygons: tuple[GlobalLanePolygon, ...]) -> int:
	"""The lane id of the first zone containing ``point``, or ``0`` (unassigned) if none.

	``0`` matches ``VehicleState.lane_id``'s existing "unknown" sentinel default.
	"""
	return _label_for_point(point, global_lane_polygons)


def _label_for_point(point: Point2D, global_polygons: tuple[GlobalLabeledPolygon, ...]) -> int:
	for label, polygon in global_polygons:
		if point_in_polygon(point, polygon):
			return label
	return 0

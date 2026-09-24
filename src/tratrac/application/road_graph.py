"""Convert authored link/lane/plane zones into global-frame polygons and classify points.

Mirrors ``application/exclusion.py``'s reference-frame -> global-frame conversion,
but classification is per-observation (a vehicle can cross links/lanes/planes
mid-track) rather than the track-level majority vote ``excluded_track_ids`` uses.
Link/Lane classification feeds SSAM's per-VEHICLE-RECORD fields; Plane
classification feeds `MultiPlaneTransform` (Group C5) instead — see
GitHub Issues Groups C1/C2/C5.
"""

from __future__ import annotations

from tratrac.domain.geometry import Point2D, point_in_polygon
from tratrac.domain.ports import CoordinateTransform
from tratrac.domain.road_graph import LaneZones, LinkZones, PlaneZones

# One zone's global-frame polygon, paired with its label (link_id, lane_id, or plane_id).
GlobalLabeledPolygon = tuple[int, tuple[Point2D, ...]]
GlobalLinkPolygon = GlobalLabeledPolygon
GlobalLanePolygon = GlobalLabeledPolygon
GlobalPlanePolygon = GlobalLabeledPolygon


def to_global_link_polygons(
	zones: LinkZones, pose: CoordinateTransform
) -> tuple[GlobalLinkPolygon, ...]:
	"""Map each zone's polygon from its reference frame into the global frame.

	``pose.apply(vertex, reference_frame)`` maps a vertex authored on that frame
	(raw -> global). Order is preserved from ``zones`` — first match wins on
	overlapping zones. Raises ``ValueError`` if a zone's ``reference_frame`` isn't
	an anchor ``pose`` knows about.
	"""
	try:
		return tuple(
			(
				zone.link_id,
				tuple(pose.apply(v, zone.reference_frame) for v in zone.polygon.vertices),
			)
			for zone in zones.zones
		)
	except KeyError as exc:
		raise ValueError(
			f"link zone reference_frame is not a frame in the transforms file: {exc}"
		) from exc


def link_id_for_point(point: Point2D, global_link_polygons: tuple[GlobalLinkPolygon, ...]) -> int:
	"""The link id of the first zone containing ``point``, or ``0`` (unassigned) if none.

	``0`` matches ``VehicleState.link_id``'s existing "unknown" sentinel default.
	"""
	return _label_for_point(point, global_link_polygons)


def to_global_lane_polygons(
	zones: LaneZones, pose: CoordinateTransform
) -> tuple[GlobalLanePolygon, ...]:
	"""Map each lane zone's polygon from its reference frame into the global frame.

	Mirrors ``to_global_link_polygons``; the label carried is ``lane_id``, not ``link_id``.
	"""
	try:
		return tuple(
			(
				zone.lane_id,
				tuple(pose.apply(v, zone.reference_frame) for v in zone.polygon.vertices),
			)
			for zone in zones.zones
		)
	except KeyError as exc:
		raise ValueError(
			f"lane zone reference_frame is not a frame in the transforms file: {exc}"
		) from exc


def lane_id_for_point(point: Point2D, global_lane_polygons: tuple[GlobalLanePolygon, ...]) -> int:
	"""The lane id of the first zone containing ``point``, or ``0`` (unassigned) if none.

	``0`` matches ``VehicleState.lane_id``'s existing "unknown" sentinel default.
	"""
	return _label_for_point(point, global_lane_polygons)


def to_global_plane_polygons(
	zones: PlaneZones, pose: CoordinateTransform
) -> tuple[GlobalPlanePolygon, ...]:
	"""Map each plane zone's polygon from its reference frame into the global frame.

	Mirrors ``to_global_link_polygons``; the label carried is ``plane_id``.
	"""
	try:
		return tuple(
			(
				zone.plane_id,
				tuple(pose.apply(v, zone.reference_frame) for v in zone.polygon.vertices),
			)
			for zone in zones.zones
		)
	except KeyError as exc:
		raise ValueError(
			f"plane zone reference_frame is not a frame in the transforms file: {exc}"
		) from exc


def plane_id_for_point(
	point: Point2D, global_plane_polygons: tuple[GlobalPlanePolygon, ...]
) -> int:
	"""The plane id of the first zone containing ``point``, or ``0`` if none.

	Unlike Link/Lane, ``0`` here is not a reserved "unknown" sentinel — plane zones commonly
	partition the *whole* scene (ground plus every bridge/overpass), so ``0`` is a legitimate
	default/ground-plane label an operator can rely on for points outside every explicit zone.
	It is the caller's responsibility to fit a homography for whichever plane ids actually
	occur (``MultiPlaneTransform`` raises clearly if one is missing).
	"""
	return _label_for_point(point, global_plane_polygons)


def _label_for_point(point: Point2D, global_polygons: tuple[GlobalLabeledPolygon, ...]) -> int:
	for label, polygon in global_polygons:
		if point_in_polygon(point, polygon):
			return label
	return 0

"""Convert authored exclusion zones into global-frame polygons for masking.

Each zone is authored on a reference frame; this pushes its vertices through that
frame's ego-motion pose once, so the runtime ``DetectionMask`` only ever maps
global -> current-raw. For a static run every pose is the identity and the global
polygons equal the authored raw polygons. See src/tratrac/application/EXCLUSION_ZONES.md.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.geometry import Point2D, point_in_polygon
from tratrac.domain.ports import CoordinateTransform


def to_global_polygons(
	zones: ExclusionZones, pose: CoordinateTransform
) -> tuple[tuple[Point2D, ...], ...]:
	"""Map each zone's polygon from its reference frame into the global frame.

	``pose.apply(vertex, reference_frame)`` maps a vertex authored on that frame
	(raw -> global). Raises ``ValueError`` if a zone's ``reference_frame`` isn't an
	anchor ``pose`` knows about (re-wrapping the ``PerFrameTransform`` ``KeyError``).
	"""
	try:
		return tuple(
			tuple(pose.apply(v, zone.reference_frame) for v in zone.polygon.vertices)
			for zone in zones.zones
		)
	except KeyError as exc:
		raise ValueError(
			f"exclusion zone reference_frame is not an anchor in the manifest: {exc}"
		) from exc


def excluded_track_ids(
	centroids: Iterable[tuple[int, Point2D]],
	global_polygons: tuple[tuple[Point2D, ...], ...],
	*,
	min_fraction: float,
) -> set[int]:
	"""Track ids to drop: those whose fraction of observations inside any zone ≥ ``min_fraction``.

	Track-aware exclusion (src/tratrac/application/EXCLUSION_ZONES.md): "objects passing here don't interest
	me" is about the object, so a whole track is dropped once enough of its life is spent in a
	zone. ``centroids`` are ``(track_id, centroid)`` per observation, in the global frame (the
	same frame ``global_polygons`` live in). With no polygons, nothing is excluded.
	"""
	if not global_polygons:
		return set()
	inside: dict[int, int] = defaultdict(int)
	total: dict[int, int] = defaultdict(int)
	for track_id, point in centroids:
		total[track_id] += 1
		if any(point_in_polygon(point, polygon) for polygon in global_polygons):
			inside[track_id] += 1
	return {tid for tid, n in total.items() if n > 0 and inside[tid] / n >= min_fraction}

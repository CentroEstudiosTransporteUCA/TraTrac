"""Convert authored exclusion zones into global-frame polygons for masking.

Each zone is authored on a reference frame; this pushes its vertices through that
frame's ego-motion pose once, so the runtime ``DetectionMask`` only ever maps
global -> current-raw. For a static run every pose is the identity and the global
polygons equal the authored raw polygons. See vault/21_exclusion_zones.md.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.geometry import Point2D, Transform2D, point_in_polygon


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


def excluded_track_ids(
	centroids: Iterable[tuple[int, Point2D]],
	global_polygons: tuple[tuple[Point2D, ...], ...],
	*,
	min_fraction: float,
) -> set[int]:
	"""Track ids to drop: those whose fraction of observations inside any zone ≥ ``min_fraction``.

	Track-aware exclusion (vault/21_exclusion_zones.md): "objects passing here don't interest
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

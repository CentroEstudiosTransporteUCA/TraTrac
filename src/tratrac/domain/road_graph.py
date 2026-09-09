"""Road-topology zones: image-space polygons labeled with a Link ID or Lane ID.

Mirrors ``domain/exclusion.py``'s shape (a set of polygons, each authored on a
*reference frame*) but carries a label instead of a uniform drop decision — a
vehicle inside a zone is classified into that zone's ``link_id``/``lane_id``
rather than excluded. See ``docs/roadmap/road_topology.md`` for the sourcing
strategy and ``VehicleState.link_id``/``lane_id`` (``domain/vehicle.py``) for
the SSAM fields these feed.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Polygon


@dataclass(frozen=True, slots=True)
class LinkZone:
	"""One road-link polygon, its label, and the reference frame it's authored on.

	``reference_frame`` is the source ``Frame.index`` the polygon's pixel
	coordinates are expressed in (``0`` for a static camera / single-frame
	authoring), exactly like ``ExclusionZone.reference_frame``.
	"""

	link_id: int
	reference_frame: int
	polygon: Polygon

	def __post_init__(self) -> None:
		if self.link_id <= 0:
			raise ValueError(
				f"link_id must be positive (0 is the SSAM 'unknown' sentinel), got {self.link_id}."
			)


@dataclass(frozen=True, slots=True)
class LinkZones:
	"""A collection of link zones. Pure data; empty is legal (assigns nothing)."""

	zones: tuple[LinkZone, ...]


@dataclass(frozen=True, slots=True)
class LaneZone:
	"""One lane-strip polygon within a link, and the reference frame it's authored on.

	``link_id`` records which link this lane belongs to (operator documentation / future
	cross-checking — see `docs/IMPLEMENTATION_PLAN.md` Group C2); classification itself
	(``application/road_graph.lane_id_for_point``) is plain point-in-polygon over the lane
	zones, independent of any separately-computed Link ID.
	"""

	link_id: int
	lane_id: int
	reference_frame: int
	polygon: Polygon

	def __post_init__(self) -> None:
		if self.link_id <= 0:
			raise ValueError(
				f"link_id must be positive (0 is the SSAM 'unknown' sentinel), got {self.link_id}."
			)
		if not 1 <= self.lane_id <= 255:
			raise ValueError(
				"lane_id must be in [1, 255] (0 is the SSAM 'unknown' sentinel, and it's a Byte "
				f"field), got {self.lane_id}."
			)


@dataclass(frozen=True, slots=True)
class LaneZones:
	"""A collection of lane zones. Pure data; empty is legal (assigns nothing)."""

	zones: tuple[LaneZone, ...]

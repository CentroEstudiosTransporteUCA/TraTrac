"""Image-space exclusion zones: regions whose detections must not be analyzed.

A set of pixel polygons (see ``vault/21_exclusion_zones.md``), each authored on a
*reference frame*. Pure data: the drop is computed by an infrastructure
``DetectionMask`` that rasterizes the zones' union into a per-frame mask and drops
any detection mostly (>50% of its area) covered. Rasterizing handles concave
polygons and overlapping/adjacent zones (true union) that an analytic per-polygon
test could not.

For a moving drone, a zone is authored on the ORB keyframe anchor where its scene
region is visible and converted once into the continuous global stabilization
frame (using that frame's pose); at runtime it is mapped back into each raw frame,
so the zone tracks the scene. A static camera is the degenerate case:
``reference_frame = 0`` and an identity pose.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import Polygon


@dataclass(frozen=True, slots=True)
class ExclusionZone:
	"""One exclusion polygon and the reference frame its coordinates live in.

	``reference_frame`` is the source ``Frame.index`` the polygon's pixel
	coordinates are expressed in (``0`` for a static camera / single-frame
	authoring).
	"""

	reference_frame: int
	polygon: Polygon


@dataclass(frozen=True, slots=True)
class ExclusionZones:
	"""A collection of exclusion zones. Pure data; empty is legal (excludes nothing)."""

	zones: tuple[ExclusionZone, ...]

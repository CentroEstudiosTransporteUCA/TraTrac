"""Image-space exclusion zones: regions whose detections must not be analyzed.

A set of pixel polygons (see ``vault/21_exclusion_zones.md``). A detection is
excluded when a *majority* of its bounding box — strictly more than half its
area — falls inside any one zone. Pure value object: the drop is applied by a
``Detector`` decorator at the detector seam, and the same zones are masked out
of ORB ego-motion feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from tratrac.domain.geometry import BoundingBox, Polygon

# "Majority" overlap: strictly more than half the bbox area inside a single zone.
# This is the definition of the feature, not a tunable knob.
_MAJORITY = 0.5


@dataclass(frozen=True, slots=True)
class ExclusionZones:
	"""A collection of image-space polygons to exclude from analysis.

	Empty is legal and excludes nothing, so the feature can be wired in
	unconditionally with an empty instance standing for "off".
	"""

	zones: tuple[Polygon, ...]

	def excludes(self, bbox: BoundingBox) -> bool:
		"""Whether ``bbox`` lies mostly (>50% of its area) inside any single zone.

		Tested per zone (max), not against the union of zones, so a box straddling
		two adjacent zones can survive even when their combined coverage exceeds half
		— acceptable for distinct drawn regions (see vault/21_exclusion_zones.md).
		"""
		return any(zone.overlap_fraction(bbox) > _MAJORITY for zone in self.zones)

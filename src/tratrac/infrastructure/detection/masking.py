"""A ``Detector`` decorator that drops detections inside image-space exclusion zones.

Sits at the detector seam — the only point where detections are still in raw
pixel coordinates, before the ego-motion transform maps them into the
stabilization frame (see vault/21_exclusion_zones.md). Wrapping the real
detector keeps the pipeline untouched, matching the decorator idiom used for
step timing and transform recording.
"""

from __future__ import annotations

from tratrac.domain.detection import Detection
from tratrac.domain.exclusion import ExclusionZones
from tratrac.domain.frame import Frame
from tratrac.domain.ports import Detector


class MaskingDetector:
	"""Implements ``Detector`` by filtering an inner detector's output.

	Drops every detection a majority of whose bounding box lies inside an
	exclusion zone; passes the rest through unchanged.
	"""

	def __init__(self, inner: Detector, zones: ExclusionZones) -> None:
		self._inner = inner
		self._zones = zones

	def detect(self, frame: Frame) -> list[Detection]:
		return [d for d in self._inner.detect(frame) if not self._zones.excludes(d.bbox)]

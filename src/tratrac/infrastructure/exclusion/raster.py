"""RasterExclusionMask: drop detections inside image-space exclusion zones.

Implements the ``DetectionMask`` port by rasterizing the zones' union into a
per-frame binary mask and dropping any detection whose bounding box is mostly
(>50% of its in-frame area) covered. ``cv2.fillPoly`` fills the true union, so
concave polygons and overlapping/adjacent zones are handled correctly — unlike an
analytic per-polygon test. See vault/21_exclusion_zones.md.

The zones are held as polygons in the **global** stabilization frame; each call
maps them back into the current raw frame via the inverse ego-motion transform, so
a zone tracks its scene region under a moving drone. With stabilization off the
transform is the identity and the global polygons are the raw frame-0 polygons.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from tratrac.domain.detection import Detection
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Point2D, Transform2D

# Strictly more than half the bounding box covered by the masked union -> drop.
_MAJORITY = 0.5


class RasterExclusionMask:
	"""Implements ``DetectionMask`` by rasterizing global-frame exclusion polygons."""

	def __init__(self, global_polygons: tuple[tuple[Point2D, ...], ...]) -> None:
		self._global_polygons = global_polygons

	def filter(
		self, detections: list[Detection], transform: Transform2D, frame: Frame
	) -> list[Detection]:
		if not self._global_polygons:
			return detections
		height, width = frame.pixels.shape[:2]
		mask = self._raw_mask(transform, height, width)
		return [d for d in detections if not self._mostly_covered(mask, d, width, height)]

	def _raw_mask(self, transform: Transform2D, height: int, width: int) -> NDArray[np.uint8]:
		"""Union mask (255 = excluded) of the zones mapped global -> raw for this frame."""
		to_raw = transform.inverse()
		mask: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
		polygons = [
			np.array([_xy(to_raw.apply(v)) for v in polygon], dtype=np.int32)
			for polygon in self._global_polygons
		]
		cv2.fillPoly(mask, polygons, 255)
		return mask

	def _mostly_covered(
		self, mask: NDArray[np.uint8], detection: Detection, width: int, height: int
	) -> bool:
		box = detection.bbox
		x0 = max(0, math.floor(box.x))
		x1 = min(width, math.ceil(box.x + box.width))
		y0 = max(0, math.floor(box.y))
		y1 = min(height, math.ceil(box.y + box.height))
		if x1 <= x0 or y1 <= y0:
			return False  # bbox entirely outside the frame; nothing to cover
		# Mean over a 0/255 mask = covered fraction * 255, against the clamped (in-frame)
		# bbox area.
		return bool(mask[y0:y1, x0:x1].mean() > _MAJORITY * 255.0)


def _xy(point: Point2D) -> tuple[float, float]:
	return (point.x, point.y)

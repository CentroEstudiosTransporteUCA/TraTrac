"""Default (no-op) detection mask for the application layer.

A Null Object so ``TrajectoryPipeline`` always has a mask to apply and never needs
to guard against ``None``. The real mask (``RasterExclusionMask``) is supplied when
exclusion zones are configured; every other run uses this silent default. See
vault/21_exclusion_zones.md.
"""

from __future__ import annotations

from tratrac.domain.detection import Detection
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D


class NullDetectionMask:
	"""Passes every detection through unchanged. The pipeline's default mask."""

	def filter(
		self, detections: list[Detection], transform: Transform2D, frame: Frame
	) -> list[Detection]:
		del transform, frame  # intentionally ignored
		return detections

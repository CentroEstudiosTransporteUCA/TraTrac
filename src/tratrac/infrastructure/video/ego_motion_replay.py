"""ReplayEgoMotionEstimator: serve a previously recorded ego-motion schedule.

Implements ``EgoMotionEstimator`` by returning, for each frame, the transform a
scout pass recorded for that frame index (read from a transform CSV). The real run
replays the scout's poses verbatim instead of recomputing ORB, so exclusion zones
(placed with the scout's poses) and detections (placed with the run's poses) share
one coordinate frame — a correctness requirement, not just a speed-up. See
vault/21_exclusion_zones.md.
"""

from __future__ import annotations

from collections.abc import Mapping

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D


class ReplayEgoMotionEstimator:
	"""Implements ``EgoMotionEstimator`` from a ``{frame_index: Transform2D}`` map."""

	def __init__(self, transforms: Mapping[int, Transform2D]) -> None:
		self._transforms = dict(transforms)
		self._current = Transform2D.identity()

	@property
	def current_transform(self) -> Transform2D:
		"""The last transform returned by ``estimate`` (current frame → global)."""
		return self._current

	def estimate(self, frame: Frame) -> Transform2D:
		try:
			self._current = self._transforms[frame.index]
		except KeyError:
			raise ValueError(
				f"no recorded transform for frame {frame.index}; the transforms CSV does "
				"not cover this run (re-scout the same video/window)."
			) from None
		return self._current

"""Default (no-op) detection observer for the application layer.

A Null Object so ``TrajectoryPipeline`` always has an observer to message and
never needs to guard against ``None``. The stabilizer supplies the real observer
(the ORB ego-motion estimator) when vehicle-masked stabilization is active; every
other run uses this silent default. See vault/05_75_mvp1_9.md.
"""

from __future__ import annotations

from tratrac.domain.detection import Detection


class NullDetectionObserver:
	"""Discards every detection batch. The pipeline's default observer."""

	def observe(self, detections: list[Detection]) -> None:
		del detections  # intentionally ignored

"""AnchorRecordingEgoMotionEstimator: tees each frame's new anchors to an AnchorSink.

The ``EgoMotionEstimator`` analog of ``RecordingEgoMotionEstimator`` (GoF Decorator). The
ORB estimator notifies an ``anchor_observer`` synchronously during ``estimate(frame)`` when
a frame becomes a new keyframe anchor; that observer appends the pose to a ``pending`` list
this decorator owns. The decorator drains it right after the call — while ``frame`` is still
the frame that became the anchor — and records ``(frame, pose)`` to the sink. The pipeline
is untouched: it just drives the decorated estimator. See vault/21_exclusion_zones.md.
"""

from __future__ import annotations

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import AnchorSink, EgoMotionEstimator


class AnchorRecordingEgoMotionEstimator:
	"""``EgoMotionEstimator`` wrapper that exports each new anchor frame to an ``AnchorSink``.

	``pending`` is the list the inner estimator's ``anchor_observer`` appends an anchor pose
	to; the caller wires the observer to it (``anchor_observer=lambda i, p: pending.append(p)``)
	so this decorator and the estimator share one queue, the same pattern the scout used.
	"""

	def __init__(
		self, inner: EgoMotionEstimator, pending: list[Transform2D], sink: AnchorSink
	) -> None:
		self._inner = inner
		self._pending = pending
		self._sink = sink

	def estimate(self, frame: Frame) -> Transform2D:
		self._pending.clear()
		transform = self._inner.estimate(frame)
		for pose in self._pending:
			self._sink.record(frame, pose)
		return transform

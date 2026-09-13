"""AnchorRecordingEgoMotionEstimator: tees each new anchor frame to an AnchorSink.

The ``EgoMotionEstimator`` analog of ``RecordingEgoMotionEstimator`` (GoF Decorator). The
ORB estimator notifies an ``anchor_observer`` synchronously during ``estimate(frame)`` when
a frame becomes a new keyframe anchor; that observer appends to a ``pending`` list this
decorator owns, and the decorator drains it right after the call — while ``frame`` is still
the frame that became the anchor — recording ``frame`` to the sink. The pipeline is
untouched: it just drives the decorated estimator. See src/tratrac/application/EXCLUSION_ZONES.md.
"""

from __future__ import annotations

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import AnchorSink, EgoMotionEstimator


class AnchorRecordingEgoMotionEstimator:
	"""``EgoMotionEstimator`` wrapper that exports each new anchor frame to an ``AnchorSink``.

	``pending`` is the list the inner estimator's ``anchor_observer`` appends to (one
	entry per new anchor within the ``estimate`` call) so this decorator and the
	estimator share one queue, the same pattern the scout used. Only the *count* of
	new anchors matters here — the pose itself already lives in the transform
	sidecar under this frame's index, so it isn't threaded through to the sink.
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
		for _ in self._pending:
			self._sink.record(frame)
		return transform

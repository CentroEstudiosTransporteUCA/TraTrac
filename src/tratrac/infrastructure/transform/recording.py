"""RecordingEgoMotionEstimator: tees each frame's ego-motion transform to a sink.

The ``EgoMotionEstimator`` analog of the ``Timed*`` decorators (GoF Decorator): it
implements the port, forwards ``estimate`` unchanged, and reports the returned
transform to a ``TransformSink`` before handing it back. The pipeline is untouched
— persisting transforms is opt-in instrumentation wrapped around the port, exactly
as timing wraps the other ports (vault/15_step_timing.md). See vault/05_75_mvp1_9.md.
"""

from __future__ import annotations

from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import EgoMotionEstimator, TransformSink
from tratrac.domain.stabilization import FrameTransform


class RecordingEgoMotionEstimator:
	"""``EgoMotionEstimator`` wrapper that records each frame's transform to a sink."""

	def __init__(self, inner: EgoMotionEstimator, sink: TransformSink) -> None:
		self._inner = inner
		self._sink = sink

	def estimate(self, frame: Frame) -> Transform2D:
		transform = self._inner.estimate(frame)
		self._sink.record(FrameTransform(frame_index=frame.index, transform=transform))
		return transform

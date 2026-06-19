"""RecordingTracker: tees each frame's tracked detections to a TrackSink.

The ``Tracker`` analog of ``RecordingEgoMotionEstimator`` (GoF Decorator): it
implements the port, forwards ``update`` unchanged, and reports the resulting
tracked detections to a ``TrackSink`` before handing them back. The pipeline is
untouched — persisting the track-observation file ("export B") is opt-in
instrumentation wrapped around the port. See vault/22_smoothing.md.
"""

from __future__ import annotations

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame
from tratrac.domain.ports import Tracker, TrackSink


class RecordingTracker:
	"""``Tracker`` wrapper that records each frame's tracked detections to a sink."""

	def __init__(self, inner: Tracker, sink: TrackSink) -> None:
		self._inner = inner
		self._sink = sink

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]:
		tracked = self._inner.update(frame, detections)
		self._sink.record(frame.index, tracked)
		return tracked

"""Pipeline orchestrator: wires VideoSource -> Detector -> Tracker -> Orientation -> Exporter."""

from __future__ import annotations

from tratrac.application.orientation import OrientationEstimator
from tratrac.domain.ports import Detector, Tracker, TrajectoryExporter, VideoSource


class TrajectoryPipeline:
	"""Drives the per-frame loop. Caller opens the video; pipeline owns the exporter lifecycle."""

	def __init__(
		self,
		*,
		video: VideoSource,
		detector: Detector,
		tracker: Tracker,
		exporter: TrajectoryExporter,
		orientation: OrientationEstimator,
	) -> None:
		self._video = video
		self._detector = detector
		self._tracker = tracker
		self._exporter = exporter
		self._orientation = orientation

	def run(self) -> int:
		"""Process every frame from the (already-open) video. Returns the number of frames processed."""
		fps = self._video.metadata.fps
		count = 0
		with self._exporter:
			for frame in self._video.frames():
				timestamp = frame.timestamp_seconds(fps)
				detections = self._detector.detect(frame)
				tracked = self._tracker.update(frame, detections)
				states = [self._orientation.estimate(t, timestamp) for t in tracked]
				self._exporter.emit_frame(timestamp, states)
				count += 1
		return count

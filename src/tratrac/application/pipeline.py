"""Pipeline orchestrator: wires VideoSource -> Detector -> Tracker -> Orientation -> Exporter."""

from __future__ import annotations

from tratrac.application.progress import NullProgressReporter
from tratrac.domain.ports import (
	Detector,
	OrientationEstimator,
	ProgressReporter,
	Tracker,
	TrajectoryExporter,
	VideoSource,
)
from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
)


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
		reporter: ProgressReporter | None = None,
	) -> None:
		self._video = video
		self._detector = detector
		self._tracker = tracker
		self._exporter = exporter
		self._orientation = orientation
		# Null Object default: the pipeline always has a reporter to message and
		# never has to guard against None.
		self._reporter: ProgressReporter = reporter or NullProgressReporter()

	def run(self) -> int:
		"""Process every frame from the (already-open) video.

		Emits a progress stream to the reporter: one ``ProcessingStarted`` before
		the loop, a ``FrameProcessed`` after each frame, a ``ProcessingFinished``
		at the end. If a frame raises, a ``ProcessingFailed`` is emitted and the
		error re-raises. Returns the number of frames processed.
		"""
		metadata = self._video.metadata
		fps = metadata.fps
		total = metadata.total_frames
		count = 0
		self._reporter.receive(ProcessingStarted(metadata=metadata))
		with self._exporter:
			for frame in self._video.frames():
				try:
					timestamp = frame.timestamp_seconds(fps)
					detections = self._detector.detect(frame)
					tracked = self._tracker.update(frame, detections)
					states = self._orientation.estimate(tracked, timestamp)
					self._exporter.emit_frame(timestamp, states)
				except Exception as exc:
					self._reporter.receive(
						ProcessingFailed(frame_index=frame.index, error=repr(exc))
					)
					raise
				count += 1
				self._reporter.receive(
					FrameProcessed(
						frame_index=frame.index,
						total_frames=total,
						timestamp_seconds=timestamp,
						active_tracks=len(states),
					)
				)
		self._reporter.receive(ProcessingFinished(frames_processed=count))
		return count

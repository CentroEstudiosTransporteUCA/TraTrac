"""Pipeline orchestrator: wires VideoSource -> Detector -> Tracker -> Orientation -> Exporter."""

from __future__ import annotations

from tratrac.application.detection_observer import NullDetectionObserver
from tratrac.application.progress import NullProgressReporter
from tratrac.application.stabilization import apply_transform
from tratrac.domain.ports import (
	DetectionObserver,
	Detector,
	EgoMotionEstimator,
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
		detection_observer: DetectionObserver | None = None,
		ego_motion: EgoMotionEstimator | None = None,
	) -> None:
		self._video = video
		self._detector = detector
		self._tracker = tracker
		self._exporter = exporter
		self._orientation = orientation
		# Coordinate stabilization (MVP1.9, vault/05_75_mvp1_9.md): when present, each
		# frame's detections are mapped into the keyframe-anchored global frame BEFORE
		# tracking, so detection/tracking run on the raw, full-resolution frame (no
		# black-border cropping) while trajectories stay free of drone ego-motion.
		# None => no stabilization (identity); detections pass through unchanged.
		self._ego_motion = ego_motion
		# Null Object default: the pipeline always has a reporter to message and
		# never has to guard against None.
		self._reporter: ProgressReporter = reporter or NullProgressReporter()
		# Same Null Object treatment: the masked-ORB stabilizer subscribes here to
		# reuse each frame's detections; every other run gets the silent default.
		self._detection_observer: DetectionObserver = detection_observer or NullDetectionObserver()

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
					# Hand the raw-frame detections to any upstream subscriber (the
					# stabilizer) before estimating motion — it masks them out of ORB
					# feature extraction on this same frame.
					self._detection_observer.observe(detections)
					if self._ego_motion is not None:
						# Stabilize coordinates, not pixels: map each detection into the
						# global frame so the tracker associates ego-motion-free boxes.
						transform = self._ego_motion.estimate(frame)
						detections = [apply_transform(d, transform) for d in detections]
					tracked = self._tracker.update(frame, detections)
					states = self._orientation.estimate(tracked, timestamp)
					# Pass the raw frame so pixel exporters can render the overlay (they
					# map states back to raw via the ego-motion transform); data
					# exporters ignore it.
					self._exporter.emit_frame(timestamp, states, frame)
				except Exception as exc:
					self._reporter.receive(
						ProcessingFailed(frame_index=frame.index, error=repr(exc))
					)
					raise
				count += 1
				self._reporter.receive(
					FrameProcessed(
						frame_index=frame.index,
						frames_done=count,
						total_frames=total,
						timestamp_seconds=timestamp,
						active_tracks=len(states),
					)
				)
		self._reporter.receive(ProcessingFinished(frames_processed=count))
		return count

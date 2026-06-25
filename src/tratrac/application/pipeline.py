"""Pipeline orchestrator: wires VideoSource -> Detector -> Tracker -> TrackSink.

Perception only: it detects, (optionally) removes ego-motion, masks, and tracks, then
records the raw tracked measurements to a ``TrackSink`` — the canonical run output
("export B", vault/01). Kinematics (orientation/speed/accel) and the SSAM ``.trj`` are
**not** produced here; they are derived offline by ``tratrac-smooth`` from the record
(vault/22). Keeping the pipeline to raw measurements is what lets the smoother de-jitter
position instead of re-smoothing already-derived kinematics.
"""

from __future__ import annotations

from tratrac.application.detection_mask import NullDetectionMask
from tratrac.application.detection_observer import NullDetectionObserver
from tratrac.application.progress import NullProgressReporter
from tratrac.application.stabilization import apply_transform
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import (
	DetectionMask,
	DetectionObserver,
	Detector,
	EgoMotionEstimator,
	ProgressReporter,
	Tracker,
	TrackSink,
	VideoSource,
)
from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
)


class TrajectoryPipeline:
	"""Drives the per-frame loop. Caller opens the video; pipeline owns the sink lifecycle."""

	def __init__(
		self,
		*,
		video: VideoSource,
		detector: Detector,
		tracker: Tracker,
		sink: TrackSink,
		reporter: ProgressReporter | None = None,
		detection_observer: DetectionObserver | None = None,
		detection_mask: DetectionMask | None = None,
		ego_motion: EgoMotionEstimator | None = None,
	) -> None:
		self._video = video
		self._detector = detector
		self._tracker = tracker
		# The run's primary output: raw tracked measurements per frame. The pipeline
		# owns its lifecycle (as it used to own the exporter's).
		self._sink = sink
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
		# Exclusion zones drop detections in raw pixel space, after ego-motion is
		# estimated (so zones can be mapped into the current frame) but before the
		# detections are stabilized. Null Object default => no exclusions.
		self._detection_mask: DetectionMask = detection_mask or NullDetectionMask()

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
		with self._sink:
			for frame in self._video.frames():
				try:
					timestamp = frame.timestamp_seconds(fps)
					detections = self._detector.detect(frame)
					# Hand the raw-frame detections to any upstream subscriber (the
					# stabilizer) before estimating motion — it masks them out of ORB
					# feature extraction on this same frame.
					self._detection_observer.observe(detections)
					# The frame's ego-motion pose (raw -> global); identity when off.
					transform = (
						self._ego_motion.estimate(frame)
						if self._ego_motion is not None
						else Transform2D.identity()
					)
					# Drop detections inside exclusion zones, in raw pixel space, using
					# the pose to map each zone into this frame (vault/21). No-op default.
					detections = self._detection_mask.filter(detections, transform, frame)
					if self._ego_motion is not None:
						# Stabilize coordinates, not pixels: map each detection into the
						# global frame so the tracker associates ego-motion-free boxes.
						detections = [apply_transform(d, transform) for d in detections]
					tracked = self._tracker.update(frame, detections)
					# Record the raw measurements; this is the run's output.
					self._sink.record(frame.index, tracked)
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
						active_tracks=len(tracked),
					)
				)
		self._reporter.receive(ProcessingFinished(frames_processed=count))
		return count

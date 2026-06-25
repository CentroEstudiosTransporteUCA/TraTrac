"""Ports: abstract interfaces the application layer talks to. No infrastructure here."""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType
from typing import Protocol

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import Transform2D
from tratrac.domain.progress import ProgressEvent
from tratrac.domain.stabilization import FrameTransform
from tratrac.domain.timing import StepTiming
from tratrac.domain.vehicle import VehicleState


class VideoSource(Protocol):
	"""Streams decoded frames from some video container. Use as a context manager."""

	@property
	def metadata(self) -> VideoMetadata: ...

	def frames(self) -> Iterator[Frame]: ...

	def __enter__(self) -> VideoSource: ...

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None: ...


class EgoMotionEstimator(Protocol):
	"""Estimates camera ego-motion per frame for coordinate stabilization.

	Stateful: ``estimate`` is called once per frame in stream order and returns the
	transform mapping *this frame's* pixel coordinates into a continuous global
	frame (anchored to the first frame). The first call returns the identity.
	Implementations may match against a keyframe anchor and compose anchor poses
	internally. The pipeline applies the returned transform to detections (not
	pixels). See vault/05_75_mvp1_9.md.
	"""

	def estimate(self, frame: Frame) -> Transform2D: ...


class StabilizationTransformSource(Protocol):
	"""Exposes the latest ego-motion transform (current frame -> global).

	Both the live ORB estimator and the replay estimator publish this so the overlay
	video can map stabilized coordinates back onto the raw frame. Read-only view,
	distinct from ``EgoMotionEstimator`` (which the pipeline drives)."""

	@property
	def current_transform(self) -> Transform2D: ...


class Detector(Protocol):
	"""Detects vehicles in a single frame."""

	def detect(self, frame: Frame) -> list[Detection]: ...


class DetectionObserver(Protocol):
	"""Receives each frame's detections after the detector runs, in stream order.

	A backward channel from the pipeline to an upstream collaborator that wants to
	reuse detections the pipeline already computed instead of detecting again. The
	masked-ORB ego-motion path uses it: the stabilizer keeps the latest batch and,
	on the next frame, masks those vehicles out of ORB feature extraction so the
	moving foreground cannot bias the ego-motion fit. See vault/05_75_mvp1_9.md.
	"""

	def observe(self, detections: list[Detection]) -> None: ...


class DetectionStabilizer(Protocol):
	"""Maps a frame's detections into the global stabilization frame via its ego-motion pose.

	Applied after ego-motion estimation, before tracking, so the tracker associates
	ego-motion-free boxes. The Null Object (no stabilization) returns the detections
	unchanged. A port so the step is decoratable/timeable like the others (vault/15).
	"""

	def stabilize(self, detections: list[Detection], transform: Transform2D) -> list[Detection]: ...


class Tracker(Protocol):
	"""Assigns stable identities to detections across frames."""

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]: ...


class TrajectoryExporter(Protocol):
	"""
	Writes per-timestep vehicle states to some trajectory output.

	Used as a context manager so the exporter can write headers on enter and
	flush/close on exit.

	This is a pure data port — it carries no pixels. Visualization (drawing
	trajectories over the footage) is a post-hoc concern handled by the standalone
	``OverlayVideoExporter`` / ``tratrac-render``, not a pipeline exporter (see
	vault/20_video_export.md).
	"""

	def emit_frame(self, timestamp_seconds: float, states: list[VehicleState]) -> None: ...

	def __enter__(self) -> TrajectoryExporter: ...

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None: ...


class ProgressReporter(Protocol):
	"""Receives a stream of progress events while a video is processed.

	Single-channel messaging: the pipeline sends ``ProgressEvent``s via
	``receive``; each reporter dispatches on the concrete type and silently
	ignores events it does not handle. This keeps the event vocabulary open for
	extension (see ``tratrac.domain.progress``).
	"""

	def receive(self, event: ProgressEvent) -> None: ...


class TimingSink(Protocol):
	"""Receives per-step timing records while a video is processed.

	One record per step per frame. Adapters render them (CSV now, a telemetry
	POST later); see ``tratrac.domain.timing`` and vault/15_step_timing.md.
	"""

	def record(self, timing: StepTiming) -> None: ...


class TrackSink(Protocol):
	"""Receives each frame's tracked detections while a video is processed.

	One call per processed frame, with the absolute ``frame_index`` and that frame's
	tracked detections (in the tracker's coordinate frame — stabilized pixels when
	ego-motion is on). Adapters persist them as the track-observation file — the
	canonical run output (the internal record of the dual-export architecture), which
	the offline ``tratrac-smooth`` pass reads to run the Kalman/RTS smoother and produce
	the SSAM ``.trj``. Used as a context manager (write a header on enter, flush/close on
	exit), since it is the pipeline's primary output. See vault/22_smoothing.md.
	"""

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None: ...

	def __enter__(self) -> TrackSink: ...

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None: ...


class TransformSink(Protocol):
	"""Receives each frame's ego-motion transform while a video is processed.

	One record per processed frame (current frame -> global stabilization frame).
	Adapters persist them (CSV now) so a downstream tool can invert each to map
	stabilized coordinates back onto the raw frame. Streaming, like ``TimingSink``;
	see ``tratrac.domain.stabilization`` and vault/05_75_mvp1_9.md.
	"""

	def record(self, frame_transform: FrameTransform) -> None: ...


class AnchorSink(Protocol):
	"""Receives each ORB keyframe **anchor** as the run discovers it.

	One record per re-anchor: the anchor ``frame`` (its pixels are exported as a PNG
	for the operator to draw exclusion zones on) and its global pose (raw -> global).
	Adapters persist the images + a manifest of ``(frame_index, pose, image)``, which
	the post-process pass reads to map zones authored on an anchor into the global
	frame. Used as a context manager (the manifest is written on exit). See
	vault/21_exclusion_zones.md.
	"""

	def record(self, frame: Frame, pose: Transform2D) -> None: ...

	def __enter__(self) -> AnchorSink: ...

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None: ...

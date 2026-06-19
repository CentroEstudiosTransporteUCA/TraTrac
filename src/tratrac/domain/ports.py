"""Ports: abstract interfaces the application layer talks to. No infrastructure here."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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


class DetectionMask(Protocol):
	"""Drops detections that fall inside masked image regions (exclusion zones).

	Applied after detection and ego-motion estimation but BEFORE the detections are
	mapped into the global frame, so it tests raw-pixel boxes. ``transform`` is the
	frame's ego-motion pose (raw -> global; identity when stabilization is off): an
	implementation maps its masked regions back into the raw frame with the inverse
	and drops detections mostly covered by them, so the zones track the scene under
	a moving drone. See vault/21_exclusion_zones.md.
	"""

	def filter(
		self, detections: list[Detection], transform: Transform2D, frame: Frame
	) -> list[Detection]: ...


class Tracker(Protocol):
	"""Assigns stable identities to detections across frames."""

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]: ...


class OrientationEstimator(Protocol):
	"""Turns a frame's tracked detections into vehicle states.

	Batch (one call per frame) so it is a uniform pipeline step alongside the
	other ports, decoratable the same way — see vault/15_step_timing.md.
	"""

	def estimate(
		self, tracked: Sequence[TrackedDetection], timestamp_seconds: float
	) -> list[VehicleState]: ...


class TrajectoryExporter(Protocol):
	"""
	Writes per-timestep vehicle states to some trajectory output.

	Used as a context manager so the exporter can write headers on enter and
	flush/close on exit.

	``emit_frame`` also receives the ``Frame`` the states were derived from — the
	(already-stabilized, if stabilization is on) pixels the pipeline processed.
	Pure data exporters (SSAM ``.trj``) ignore it; pixel exporters (the overlay
	video writer) render onto it. Carrying the frame here keeps every output a
	single uniform port so they compose behind one ``CompositeTrajectoryExporter``.
	"""

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None: ...

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
	ego-motion is on). Adapters persist them as the "export B" track-observation file
	(the extended internal format of the dual-export architecture), which the offline
	``tratrac-smooth`` pass reads to run the Kalman/RTS smoother. Streaming, like
	``TransformSink``; see vault/22_smoothing.md.
	"""

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None: ...


class TransformSink(Protocol):
	"""Receives each frame's ego-motion transform while a video is processed.

	One record per processed frame (current frame -> global stabilization frame).
	Adapters persist them (CSV now) so a downstream tool can invert each to map
	stabilized coordinates back onto the raw frame. Streaming, like ``TimingSink``;
	see ``tratrac.domain.stabilization`` and vault/05_75_mvp1_9.md.
	"""

	def record(self, frame_transform: FrameTransform) -> None: ...

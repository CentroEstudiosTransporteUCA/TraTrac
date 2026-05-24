"""Ports: abstract interfaces the application layer talks to. No infrastructure here."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Protocol

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.progress import ProgressEvent
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


class Detector(Protocol):
	"""Detects vehicles in a single frame."""

	def detect(self, frame: Frame) -> list[Detection]: ...


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
	Writes per-timestep vehicle states to disk in some trajectory format.

	Used as a context manager so the exporter can write headers on enter and
	flush/close on exit.
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

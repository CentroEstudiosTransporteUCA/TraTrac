"""Ports: abstract interfaces the application layer talks to. No infrastructure here."""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType
from typing import Protocol

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame, VideoMetadata
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

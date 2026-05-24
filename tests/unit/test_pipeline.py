"""Pipeline orchestration tests using fake adapters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType

import numpy as np
import pytest

from tratrac.application.orientation import EmaOrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
	ProgressEvent,
)
from tratrac.domain.vehicle import VehicleState


class _FakeVideoSource:
	def __init__(self, n_frames: int, fps: float = 30.0) -> None:
		self._n = n_frames
		self._metadata = VideoMetadata(width=100, height=100, fps=fps, total_frames=n_frames)

	@property
	def metadata(self) -> VideoMetadata:
		return self._metadata

	def frames(self) -> Iterator[Frame]:
		for i in range(self._n):
			yield Frame(index=i, pixels=np.zeros((100, 100, 3), dtype=np.uint8))

	def __enter__(self) -> _FakeVideoSource:
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		return None


class _FixedDetector:
	def __init__(self, per_frame: list[list[Detection]]) -> None:
		self._per_frame = per_frame

	def detect(self, frame: Frame) -> list[Detection]:
		if frame.index >= len(self._per_frame):
			return []
		return self._per_frame[frame.index]


class _IdentityTracker:
	"""Trivial tracker: assigns track_id = detection index."""

	def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedDetection]:
		return [TrackedDetection(track_id=i, detection=d) for i, d in enumerate(detections)]


class _CapturingExporter:
	def __init__(self) -> None:
		self.emitted: list[tuple[float, list[VehicleState]]] = []
		self.opened = False
		self.closed = False

	def emit_frame(self, timestamp_seconds: float, states: list[VehicleState]) -> None:
		self.emitted.append((timestamp_seconds, states))

	def __enter__(self) -> _CapturingExporter:
		self.opened = True
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self.closed = True


class _RecordingReporter:
	def __init__(self) -> None:
		self.events: list[ProgressEvent] = []

	def receive(self, event: ProgressEvent) -> None:
		self.events.append(event)


def _bbox(x: float) -> BoundingBox:
	return BoundingBox(x=x, y=10.0, width=4.0, height=2.0)


def _det(x: float) -> Detection:
	return Detection(bbox=_bbox(x), score=0.9, vehicle_class=VehicleClass.CAR)


class TestPipeline:
	def test_emits_one_record_per_frame_with_correct_timestamps(self) -> None:
		exporter = _CapturingExporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=3, fps=30.0),
			detector=_FixedDetector([[_det(0.0)], [_det(1.0)], [_det(2.0)]]),
			tracker=_IdentityTracker(),
			exporter=exporter,
			orientation=EmaOrientationEstimator(),
		)
		processed = pipeline.run()

		assert processed == 3
		assert len(exporter.emitted) == 3
		# Timestamps = frame_index / fps.
		assert exporter.emitted[0][0] == 0.0
		assert exporter.emitted[1][0] == 1.0 / 30.0
		assert exporter.emitted[2][0] == 2.0 / 30.0

	def test_opens_and_closes_the_exporter(self) -> None:
		exporter = _CapturingExporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=1),
			detector=_FixedDetector([[]]),
			tracker=_IdentityTracker(),
			exporter=exporter,
			orientation=EmaOrientationEstimator(),
		)
		pipeline.run()
		assert exporter.opened
		assert exporter.closed

	def test_zero_detections_emits_empty_state_list(self) -> None:
		exporter = _CapturingExporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=2),
			detector=_FixedDetector([[], []]),
			tracker=_IdentityTracker(),
			exporter=exporter,
			orientation=EmaOrientationEstimator(),
		)
		pipeline.run()
		assert all(states == [] for _, states in exporter.emitted)

	def test_states_carry_orientation_derived_attributes(self) -> None:
		exporter = _CapturingExporter()
		# Eastward motion across three frames.
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=3, fps=30.0),
			detector=_FixedDetector([[_det(0.0)], [_det(30.0)], [_det(60.0)]]),
			tracker=_IdentityTracker(),
			exporter=exporter,
			orientation=EmaOrientationEstimator(),
		)
		pipeline.run()
		# Final frame's state should have eastward heading.
		final_state = exporter.emitted[-1][1][0]
		assert final_state.heading.dx > 0.9
		assert final_state.speed > 0.0


class TestProgressEmission:
	def test_emits_started_frames_finished_in_order(self) -> None:
		reporter = _RecordingReporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=2, fps=30.0),
			detector=_FixedDetector([[_det(0.0)], [_det(1.0)]]),
			tracker=_IdentityTracker(),
			exporter=_CapturingExporter(),
			orientation=EmaOrientationEstimator(),
			reporter=reporter,
		)
		pipeline.run()

		assert isinstance(reporter.events[0], ProcessingStarted)
		frame_events = [e for e in reporter.events if isinstance(e, FrameProcessed)]
		assert [e.frame_index for e in frame_events] == [0, 1]

		finished = reporter.events[-1]
		assert isinstance(finished, ProcessingFinished)
		assert finished.frames_processed == 2

	def test_emits_failed_then_reraises_on_frame_error(self) -> None:
		class _BoomDetector:
			def detect(self, frame: Frame) -> list[Detection]:
				raise RuntimeError(f"boom at {frame.index}")

		reporter = _RecordingReporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=1),
			detector=_BoomDetector(),
			tracker=_IdentityTracker(),
			exporter=_CapturingExporter(),
			orientation=EmaOrientationEstimator(),
			reporter=reporter,
		)
		with pytest.raises(RuntimeError, match="boom"):
			pipeline.run()
		assert any(isinstance(e, ProcessingFailed) for e in reporter.events)

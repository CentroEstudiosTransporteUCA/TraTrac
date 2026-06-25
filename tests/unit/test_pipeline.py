"""Pipeline orchestration tests using fake adapters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType

import numpy as np
import pytest

from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.application.stabilization import EgoMotionStabilizer
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox, Transform2D
from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
	ProgressEvent,
)


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


class _CapturingSink:
	"""The run's output port: records each frame's tracked detections."""

	def __init__(self) -> None:
		self.records: list[tuple[int, list[TrackedDetection]]] = []
		self.opened = False
		self.closed = False

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		self.records.append((frame_index, tracked))

	def __enter__(self) -> _CapturingSink:
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


class _RecordingObserver:
	def __init__(self) -> None:
		self.batches: list[list[Detection]] = []

	def observe(self, detections: list[Detection]) -> None:
		self.batches.append(detections)


def _bbox(x: float) -> BoundingBox:
	return BoundingBox(x=x, y=10.0, width=4.0, height=2.0)


def _det(x: float) -> Detection:
	return Detection(bbox=_bbox(x), score=0.9, vehicle_class=VehicleClass.CAR)


class TestPipeline:
	def test_records_one_entry_per_frame_with_indices(self) -> None:
		sink = _CapturingSink()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=3, fps=30.0),
			detector=_FixedDetector([[_det(0.0)], [_det(1.0)], [_det(2.0)]]),
			tracker=_IdentityTracker(),
			sink=sink,
		)
		processed = pipeline.run()

		assert processed == 3
		assert [index for index, _ in sink.records] == [0, 1, 2]
		assert all(len(tracked) == 1 for _, tracked in sink.records)

	def test_opens_and_closes_the_sink(self) -> None:
		sink = _CapturingSink()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=1),
			detector=_FixedDetector([[]]),
			tracker=_IdentityTracker(),
			sink=sink,
		)
		pipeline.run()
		assert sink.opened
		assert sink.closed

	def test_zero_detections_records_empty_tracked(self) -> None:
		sink = _CapturingSink()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=2),
			detector=_FixedDetector([[], []]),
			tracker=_IdentityTracker(),
			sink=sink,
		)
		pipeline.run()
		assert all(tracked == [] for _, tracked in sink.records)

	def test_forwards_each_frames_detections_to_the_observer(self) -> None:
		observer = _RecordingObserver()
		per_frame = [[_det(0.0)], [_det(1.0), _det(2.0)], []]
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=3),
			detector=_FixedDetector(per_frame),
			tracker=_IdentityTracker(),
			sink=_CapturingSink(),
			detection_observer=observer,
		)
		pipeline.run()
		assert observer.batches == per_frame


class _CapturingTracker:
	"""Records the detections it receives; assigns track_id = index."""

	def __init__(self) -> None:
		self.received: list[list[Detection]] = []

	def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedDetection]:
		self.received.append(list(detections))
		return [TrackedDetection(track_id=i, detection=d) for i, d in enumerate(detections)]


class _StubEgoMotion:
	"""Returns a fixed transform; records the frames it is asked about."""

	def __init__(self, transform: Transform2D) -> None:
		self._transform = transform
		self.frames: list[Frame] = []

	def estimate(self, frame: Frame) -> Transform2D:
		self.frames.append(frame)
		return self._transform


class TestStabilization:
	def test_detections_are_transformed_before_reaching_the_tracker(self) -> None:
		# Pure translation by +100 in x; detection centre (2, 11) -> (102, 11).
		ego = _StubEgoMotion(Transform2D(a=1.0, b=0.0, tx=100.0, c=0.0, d=1.0, ty=0.0))
		tracker = _CapturingTracker()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=1),
			detector=_FixedDetector([[_det(0.0)]]),
			tracker=tracker,
			sink=_CapturingSink(),
			stabilizer=EgoMotionStabilizer(),
			ego_motion=ego,
		)
		pipeline.run()

		assert ego.frames[0].index == 0  # estimator saw the raw frame
		assert tracker.received[0][0].bbox.center.x == pytest.approx(102.0)
		assert tracker.received[0][0].bbox.center.y == pytest.approx(11.0)

	def test_without_ego_motion_detections_pass_through_unchanged(self) -> None:
		tracker = _CapturingTracker()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=1),
			detector=_FixedDetector([[_det(5.0)]]),
			tracker=tracker,
			sink=_CapturingSink(),
		)
		pipeline.run()
		# Detection centre is (5 + 2, 11) = (7, 11), untouched.
		assert tracker.received[0][0].bbox.center.x == pytest.approx(7.0)


class TestProgressEmission:
	def test_emits_started_frames_finished_in_order(self) -> None:
		reporter = _RecordingReporter()
		pipeline = TrajectoryPipeline(
			video=_FakeVideoSource(n_frames=2, fps=30.0),
			detector=_FixedDetector([[_det(0.0)], [_det(1.0)]]),
			tracker=_IdentityTracker(),
			sink=_CapturingSink(),
			reporter=reporter,
		)
		pipeline.run()

		assert isinstance(reporter.events[0], ProcessingStarted)
		frame_events = [e for e in reporter.events if isinstance(e, FrameProcessed)]
		assert [e.frame_index for e in frame_events] == [0, 1]
		assert [e.frames_done for e in frame_events] == [1, 2]  # 1-based progress count
		assert [e.active_tracks for e in frame_events] == [1, 1]

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
			sink=_CapturingSink(),
			reporter=reporter,
		)
		with pytest.raises(RuntimeError, match="boom"):
			pipeline.run()
		assert any(isinstance(e, ProcessingFailed) for e in reporter.events)

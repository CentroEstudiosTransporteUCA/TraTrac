"""Tests for the timing stopwatch, decorators, and CSV sink."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import numpy as np

from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox, Transform2D
from tratrac.domain.timing import PipelineStep, StepTiming
from tratrac.infrastructure.timing.csv import CsvTimingSink
from tratrac.infrastructure.timing.decorators import (
	StepStopwatch,
	TimedDetectionObserver,
	TimedDetector,
	TimedEgoMotion,
	TimedStabilizer,
	TimedTracker,
	TimedTrackSink,
)


class _RecordingSink:
	def __init__(self) -> None:
		self.records: list[StepTiming] = []

	def record(self, timing: StepTiming) -> None:
		self.records.append(timing)


class _StubClock:
	"""Returns the supplied values in order, one per call."""

	def __init__(self, values: list[float]) -> None:
		self._values: Iterator[float] = iter(values)

	def __call__(self) -> float:
		return next(self._values)


class _ConstDetector:
	def __init__(self, result: list[Detection]) -> None:
		self._result = result

	def detect(self, frame: Frame) -> list[Detection]:
		return self._result


class _PerDetectionTracker:
	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]:
		return [TrackedDetection(track_id=i, detection=d) for i, d in enumerate(detections)]


class _ConstTracker:
	def __init__(self, result: list[TrackedDetection]) -> None:
		self._result = result

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]:
		return self._result


class _StubTrackSink:
	"""A TrackSink that discards records; satisfies the pipeline's output port."""

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		return None

	def __enter__(self) -> _StubTrackSink:
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		return None


class _StubVideo:
	def __init__(self, n_frames: int) -> None:
		self._n = n_frames
		self._metadata = VideoMetadata(width=10, height=10, fps=30.0, total_frames=n_frames)

	@property
	def metadata(self) -> VideoMetadata:
		return self._metadata

	def frames(self) -> Iterator[Frame]:
		for i in range(self._n):
			yield _frame(i)

	def __enter__(self) -> _StubVideo:
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		return None


def _frame(index: int = 0) -> Frame:
	return Frame(index=index, pixels=np.zeros((4, 4, 3), dtype=np.uint8))


def _det() -> Detection:
	return Detection(
		bbox=BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0),
		score=0.9,
		vehicle_class=VehicleClass.CAR,
	)


class TestStepStopwatch:
	def test_records_step_ordinal_and_elapsed(self) -> None:
		sink = _RecordingSink()
		watch = StepStopwatch(PipelineStep.DETECT, sink, clock=_StubClock([10.0, 10.5]))
		assert watch.time(lambda: "value") == "value"
		assert sink.records == [StepTiming(PipelineStep.DETECT, 0, 0.5)]

	def test_ordinal_advances_per_call(self) -> None:
		sink = _RecordingSink()
		watch = StepStopwatch(PipelineStep.TRACK, sink, clock=_StubClock([0.0, 1.0, 5.0, 5.25]))
		watch.time(lambda: None)
		watch.time(lambda: None)
		assert [r.frame_ordinal for r in sink.records] == [0, 1]
		assert [r.seconds for r in sink.records] == [1.0, 0.25]


class TestTimedDetector:
	def test_forwards_result_and_records_detect(self) -> None:
		result = [_det()]
		sink = _RecordingSink()
		timed = TimedDetector(_ConstDetector(result), sink, clock=_StubClock([1.0, 1.5]))
		assert timed.detect(_frame()) is result
		assert sink.records == [StepTiming(PipelineStep.DETECT, 0, 0.5)]


class TestTimedTracker:
	def test_forwards_result_and_records_track(self) -> None:
		result = [TrackedDetection(track_id=1, detection=_det())]
		sink = _RecordingSink()
		timed = TimedTracker(_ConstTracker(result), sink, clock=_StubClock([0.0, 0.25]))
		assert timed.update(_frame(), [_det()]) is result
		assert sink.records == [StepTiming(PipelineStep.TRACK, 0, 0.25)]


class TestTimedDetectionObserver:
	def test_forwards_and_records_observe(self) -> None:
		seen: list[list[Detection]] = []
		sink = _RecordingSink()
		timed = TimedDetectionObserver(
			type("_Obs", (), {"observe": lambda _self, d: seen.append(d)})(),
			sink,
			clock=_StubClock([0.0, 0.01]),
		)
		timed.observe([_det()])
		assert len(seen) == 1
		assert sink.records == [StepTiming(PipelineStep.OBSERVE, 0, 0.01)]


class TestTimedEgoMotion:
	def test_forwards_result_and_records_ego_motion(self) -> None:
		transform = Transform2D(a=1.0, b=0.0, tx=5.0, c=0.0, d=1.0, ty=0.0)
		sink = _RecordingSink()
		timed = TimedEgoMotion(
			type("_Ego", (), {"estimate": lambda _self, _f: transform})(),
			sink,
			clock=_StubClock([0.0, 0.3]),
		)
		assert timed.estimate(_frame()) is transform
		assert sink.records == [StepTiming(PipelineStep.EGOMOTION, 0, 0.3)]


class TestTimedStabilizer:
	def test_forwards_result_and_records_stabilize(self) -> None:
		result = [_det()]
		sink = _RecordingSink()
		timed = TimedStabilizer(
			type("_Stab", (), {"stabilize": lambda _self, _d, _t: result})(),
			sink,
			clock=_StubClock([0.0, 0.02]),
		)
		assert timed.stabilize([], Transform2D.identity()) is result
		assert sink.records == [StepTiming(PipelineStep.STABILIZE, 0, 0.02)]


class _RecordingTrackSink:
	def __init__(self) -> None:
		self.recorded: list[int] = []
		self.entered = False
		self.exited = False

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		self.recorded.append(frame_index)

	def __enter__(self) -> _RecordingTrackSink:
		self.entered = True
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self.exited = True


class TestTimedTrackSink:
	def test_times_record_and_delegates_context_manager(self) -> None:
		inner = _RecordingTrackSink()
		sink = _RecordingSink()
		timed = TimedTrackSink(inner, sink, clock=_StubClock([0.0, 0.001]))
		with timed:
			timed.record(5, [])
		assert inner.recorded == [5]
		assert inner.entered
		assert inner.exited
		assert sink.records == [StepTiming(PipelineStep.RECORD, 0, 0.001)]


class TestCsvTimingSink:
	def test_writes_one_wide_row_per_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "timings.csv"
		with CsvTimingSink(path) as sink:
			for ordinal in (0, 1):
				sink.record(StepTiming(PipelineStep.DETECT, ordinal, 0.5))
				sink.record(StepTiming(PipelineStep.OBSERVE, ordinal, 0.01))
				sink.record(StepTiming(PipelineStep.EGOMOTION, ordinal, 0.3))
				sink.record(StepTiming(PipelineStep.STABILIZE, ordinal, 0.02))
				sink.record(StepTiming(PipelineStep.TRACK, ordinal, 0.25))
				sink.record(StepTiming(PipelineStep.RECORD, ordinal, 0.001))
		lines = path.read_text().splitlines()
		assert lines[0] == "frame,detect,observe,ego_motion,stabilize,track,record"
		assert lines[1] == "0,0.5,0.01,0.3,0.02,0.25,0.001"
		assert lines[2] == "1,0.5,0.01,0.3,0.02,0.25,0.001"

	def test_stabilization_only_steps_blank_on_a_non_stabilized_run(self, tmp_path: Path) -> None:
		# detect/track/record present; observe/ego_motion/stabilize blank.
		path = tmp_path / "partial.csv"
		with CsvTimingSink(path) as sink:
			sink.record(StepTiming(PipelineStep.DETECT, 0, 0.5))
			sink.record(StepTiming(PipelineStep.TRACK, 0, 0.25))
			sink.record(StepTiming(PipelineStep.RECORD, 0, 0.001))
		lines = path.read_text().splitlines()
		assert lines[1] == "0,0.5,,,,0.25,0.001"


class TestDecoratedPipeline:
	def test_writes_one_row_per_processed_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "run.csv"
		with CsvTimingSink(path) as sink:
			pipeline = TrajectoryPipeline(
				video=_StubVideo(3),
				detector=TimedDetector(_ConstDetector([_det()]), sink),
				tracker=TimedTracker(_PerDetectionTracker(), sink),
				sink=_StubTrackSink(),
			)
			assert pipeline.run() == 3
		data = path.read_text().splitlines()[1:]
		assert [row.split(",")[0] for row in data] == ["0", "1", "2"]

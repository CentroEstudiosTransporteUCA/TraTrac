"""Tests for the timing stopwatch, decorators, and CSV sink."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType

import numpy as np

from tratrac.application.orientation import EmaOrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.domain.timing import PipelineStep, StepTiming
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.timing.csv import CsvTimingSink
from tratrac.infrastructure.timing.decorators import (
	StepStopwatch,
	TimedDetector,
	TimedExporter,
	TimedOrientation,
	TimedTracker,
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


class _ConstOrientation:
	def __init__(self, result: list[VehicleState]) -> None:
		self._result = result

	def estimate(
		self, tracked: Sequence[TrackedDetection], timestamp_seconds: float
	) -> list[VehicleState]:
		return self._result


class _RecordingExporter:
	def __init__(self) -> None:
		self.calls: list[str] = []

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None:
		self.calls.append("emit")

	def __enter__(self) -> _RecordingExporter:
		self.calls.append("enter")
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self.calls.append("exit")


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


class TestTimedOrientation:
	def test_forwards_result_and_records_orient(self) -> None:
		sink = _RecordingSink()
		timed = TimedOrientation(_ConstOrientation([]), sink, clock=_StubClock([0.0, 0.125]))
		assert timed.estimate([], 0.0) == []
		assert sink.records == [StepTiming(PipelineStep.ORIENT, 0, 0.125)]


class TestTimedExporter:
	def test_times_emit_and_delegates_context_manager(self) -> None:
		sink = _RecordingSink()
		inner = _RecordingExporter()
		timed = TimedExporter(inner, sink, clock=_StubClock([0.0, 0.5]))
		with timed:
			timed.emit_frame(1.0, [], _frame())
		assert inner.calls == ["enter", "emit", "exit"]
		assert sink.records == [StepTiming(PipelineStep.EXPORT, 0, 0.5)]


class TestCsvTimingSink:
	def test_writes_one_wide_row_per_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "timings.csv"
		with CsvTimingSink(path) as sink:
			for ordinal in (0, 1):
				sink.record(StepTiming(PipelineStep.DETECT, ordinal, 0.5))
				sink.record(StepTiming(PipelineStep.TRACK, ordinal, 0.25))
				sink.record(StepTiming(PipelineStep.ORIENT, ordinal, 0.125))
				sink.record(StepTiming(PipelineStep.EXPORT, ordinal, 0.0625))
		lines = path.read_text().splitlines()
		assert lines[0] == "frame,detect,track,orient,export"
		assert lines[1] == "0,0.5,0.25,0.125,0.0625"
		assert lines[2] == "1,0.5,0.25,0.125,0.0625"

	def test_partial_final_frame_flushes_with_blanks_on_close(self, tmp_path: Path) -> None:
		path = tmp_path / "partial.csv"
		with CsvTimingSink(path) as sink:
			sink.record(StepTiming(PipelineStep.DETECT, 0, 0.5))
			sink.record(StepTiming(PipelineStep.TRACK, 0, 0.25))
		lines = path.read_text().splitlines()
		assert lines[1] == "0,0.5,0.25,,"


class TestDecoratedPipeline:
	def test_writes_one_row_per_processed_frame(self, tmp_path: Path) -> None:
		path = tmp_path / "run.csv"
		with CsvTimingSink(path) as sink:
			pipeline = TrajectoryPipeline(
				video=_StubVideo(3),
				detector=TimedDetector(_ConstDetector([_det()]), sink),
				tracker=TimedTracker(_PerDetectionTracker(), sink),
				exporter=TimedExporter(_RecordingExporter(), sink),
				orientation=TimedOrientation(
					EmaOrientationEstimator(smoothing_window=5, meters_per_pixel=1.0), sink
				),
			)
			assert pipeline.run() == 3
		data = path.read_text().splitlines()[1:]
		assert [row.split(",")[0] for row in data] == ["0", "1", "2"]

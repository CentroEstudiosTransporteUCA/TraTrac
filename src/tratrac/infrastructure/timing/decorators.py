"""Timing decorators: wrap each pipeline port to measure its per-frame latency.

Each decorator implements the port it wraps, forwards the call unchanged, and
reports a ``StepTiming`` to a ``TimingSink``. Every step runs exactly once per
frame, so each decorator counts its own calls as the frame ordinal — they stay
aligned without sharing state. The full per-frame chain is detect → observe →
ego-motion → stabilize → track → record; the stabilization-only steps (observe,
ego-motion, stabilize) are only wrapped on a `--stabilize` run. See
vault/15_step_timing.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame
from tratrac.domain.geometry import Transform2D
from tratrac.domain.ports import (
	DetectionObserver,
	DetectionStabilizer,
	Detector,
	EgoMotionEstimator,
	TimingSink,
	Tracker,
	TrackSink,
)
from tratrac.domain.timing import PipelineStep, StepTiming

_T = TypeVar("_T")


class StepStopwatch:
	"""Times one pipeline step across frames and reports each duration.

	Owns the frame ordinal it stamps onto every record: one increment per timed
	call. A collaborator the timing decorators delegate to, rather than a base
	class, so timing lives in one place without inheritance.
	"""

	def __init__(
		self,
		step: PipelineStep,
		sink: TimingSink,
		*,
		clock: Callable[[], float] = time.perf_counter,
	) -> None:
		self._step = step
		self._sink = sink
		self._clock = clock
		self._ordinal = 0

	def time(self, call: Callable[[], _T]) -> _T:
		start = self._clock()
		result = call()
		self._sink.record(StepTiming(self._step, self._ordinal, self._clock() - start))
		self._ordinal += 1
		return result


class TimedDetector:
	"""``Detector`` wrapper that times ``detect``."""

	def __init__(
		self, inner: Detector, sink: TimingSink, *, clock: Callable[[], float] = time.perf_counter
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.DETECT, sink, clock=clock)

	def detect(self, frame: Frame) -> list[Detection]:
		return self._stopwatch.time(lambda: self._inner.detect(frame))


class TimedTracker:
	"""``Tracker`` wrapper that times ``update``."""

	def __init__(
		self, inner: Tracker, sink: TimingSink, *, clock: Callable[[], float] = time.perf_counter
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.TRACK, sink, clock=clock)

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]:
		return self._stopwatch.time(lambda: self._inner.update(frame, detections))


class TimedDetectionObserver:
	"""``DetectionObserver`` wrapper that times ``observe`` (the masked-ORB feedback tee)."""

	def __init__(
		self,
		inner: DetectionObserver,
		sink: TimingSink,
		*,
		clock: Callable[[], float] = time.perf_counter,
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.OBSERVE, sink, clock=clock)

	def observe(self, detections: list[Detection]) -> None:
		self._stopwatch.time(lambda: self._inner.observe(detections))


class TimedEgoMotion:
	"""``EgoMotionEstimator`` wrapper that times ``estimate`` (the ORB match + RANSAC)."""

	def __init__(
		self,
		inner: EgoMotionEstimator,
		sink: TimingSink,
		*,
		clock: Callable[[], float] = time.perf_counter,
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.EGOMOTION, sink, clock=clock)

	def estimate(self, frame: Frame) -> Transform2D:
		return self._stopwatch.time(lambda: self._inner.estimate(frame))


class TimedStabilizer:
	"""``DetectionStabilizer`` wrapper that times ``stabilize``."""

	def __init__(
		self,
		inner: DetectionStabilizer,
		sink: TimingSink,
		*,
		clock: Callable[[], float] = time.perf_counter,
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.STABILIZE, sink, clock=clock)

	def stabilize(self, detections: list[Detection], transform: Transform2D) -> list[Detection]:
		return self._stopwatch.time(lambda: self._inner.stabilize(detections, transform))


class TimedTrackSink:
	"""``TrackSink`` wrapper: times ``record``, delegates the context manager."""

	def __init__(
		self, inner: TrackSink, sink: TimingSink, *, clock: Callable[[], float] = time.perf_counter
	) -> None:
		self._inner = inner
		self._stopwatch = StepStopwatch(PipelineStep.RECORD, sink, clock=clock)

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		self._stopwatch.time(lambda: self._inner.record(frame_index, tracked))

	def __enter__(self) -> TimedTrackSink:
		self._inner.__enter__()
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self._inner.__exit__(exc_type, exc_val, exc_tb)

"""Timing decorators: wrap each pipeline port to measure its per-frame latency.

Each decorator implements the port it wraps, forwards the call unchanged, and
reports a ``StepTiming`` to a ``TimingSink``. Both steps (detect, track) run
exactly once per frame, so each decorator counts its own calls as the frame
ordinal — they stay aligned without sharing state. See vault/15_step_timing.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from tratrac.domain.detection import Detection, TrackedDetection
from tratrac.domain.frame import Frame
from tratrac.domain.ports import Detector, TimingSink, Tracker
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

"""Tests for progress events and reporters."""

from __future__ import annotations

import io
import math

from tratrac.application.progress import NullProgressReporter
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.progress import (
	FrameProcessed,
	ProcessingFailed,
	ProcessingFinished,
	ProcessingStarted,
	ProgressEvent,
)
from tratrac.infrastructure.progress.console import ConsoleProgressReporter


class TestFrameProcessed:
	def test_fraction_and_percent(self) -> None:
		event = FrameProcessed(
			frame_index=49, frames_done=50, total_frames=100, timestamp_seconds=1.0, active_tracks=3
		)
		assert math.isclose(event.fraction, 0.5)
		assert math.isclose(event.percent, 50.0)

	def test_fraction_uses_frames_done_not_absolute_index(self) -> None:
		# Regression: with an analysis window the index is absolute (e.g. 10659)
		# while the total is the windowed count, so the index must not drive it.
		event = FrameProcessed(
			frame_index=10659,
			frames_done=1,
			total_frames=1300,
			timestamp_seconds=355.0,
			active_tracks=4,
		)
		assert math.isclose(event.fraction, 1 / 1300)

	def test_last_frame_reads_complete(self) -> None:
		event = FrameProcessed(
			frame_index=99,
			frames_done=100,
			total_frames=100,
			timestamp_seconds=1.0,
			active_tracks=0,
		)
		assert math.isclose(event.fraction, 1.0)

	def test_unknown_total_yields_zero_fraction(self) -> None:
		event = FrameProcessed(
			frame_index=10, frames_done=11, total_frames=0, timestamp_seconds=1.0, active_tracks=0
		)
		assert event.fraction == 0.0

	def test_negative_total_yields_zero_fraction(self) -> None:
		# OpenCV's frame count can come back as -1 for some containers/streams.
		event = FrameProcessed(
			frame_index=10, frames_done=11, total_frames=-1, timestamp_seconds=1.0, active_tracks=0
		)
		assert event.fraction == 0.0

	def test_fraction_clamped_to_one_when_count_under_reports(self) -> None:
		event = FrameProcessed(
			frame_index=150,
			frames_done=150,
			total_frames=100,
			timestamp_seconds=1.0,
			active_tracks=0,
		)
		assert event.fraction == 1.0


class TestNullProgressReporter:
	def test_discards_every_event_silently(self) -> None:
		reporter = NullProgressReporter()
		# None of these should raise or produce output.
		reporter.receive(ProcessingFinished(frames_processed=1))
		reporter.receive(
			FrameProcessed(
				frame_index=0, frames_done=1, total_frames=1, timestamp_seconds=0.0, active_tracks=0
			)
		)


class _FakeBar:
	"""Records the tqdm calls the reporter makes."""

	def __init__(self, **kwargs: object) -> None:
		self.kwargs = kwargs
		self.total = kwargs.get("total")
		self.advanced = 0
		self.postfixes: list[dict[str, object]] = []
		self.closed = False

	def update(self, n: int) -> None:
		self.advanced += n

	def set_postfix(self, *args: object, **kwargs: object) -> None:
		self.postfixes.append(kwargs)

	def close(self) -> None:
		self.closed = True


class _BarSpy:
	"""A bar_factory that captures every bar it creates."""

	def __init__(self) -> None:
		self.bars: list[_FakeBar] = []

	def __call__(self, **kwargs: object) -> _FakeBar:
		bar = _FakeBar(**kwargs)
		self.bars.append(bar)
		return bar


def _frame(frames_done: int, total: int, tracks: int = 0) -> FrameProcessed:
	return FrameProcessed(
		frame_index=frames_done - 1,
		frames_done=frames_done,
		total_frames=total,
		timestamp_seconds=frames_done / 10.0,
		active_tracks=tracks,
	)


class TestConsoleProgressReporter:
	def test_started_creates_a_bar_with_the_run_total(self) -> None:
		spy = _BarSpy()
		reporter = ConsoleProgressReporter(bar_factory=spy)
		meta = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=300)
		reporter.receive(ProcessingStarted(metadata=meta))
		assert len(spy.bars) == 1
		assert spy.bars[0].total == 300

	def test_unknown_total_makes_an_indeterminate_bar(self) -> None:
		spy = _BarSpy()
		meta = VideoMetadata(width=64, height=48, fps=30.0, total_frames=0)
		ConsoleProgressReporter(bar_factory=spy).receive(ProcessingStarted(metadata=meta))
		assert spy.bars[0].total is None

	def test_frames_advance_the_bar_by_the_delta(self) -> None:
		spy = _BarSpy()
		reporter = ConsoleProgressReporter(bar_factory=spy)
		meta = VideoMetadata(width=64, height=48, fps=30.0, total_frames=10)
		reporter.receive(ProcessingStarted(metadata=meta))
		reporter.receive(_frame(1, 10, tracks=2))
		reporter.receive(_frame(3, 10, tracks=4))  # jumped two frames
		bar = spy.bars[0]
		assert bar.advanced == 3  # 1 + 2, not the absolute index
		assert bar.postfixes[-1]["tracks"] == 4

	def test_finished_closes_the_bar(self) -> None:
		spy = _BarSpy()
		reporter = ConsoleProgressReporter(bar_factory=spy)
		meta = VideoMetadata(width=64, height=48, fps=30.0, total_frames=5)
		reporter.receive(ProcessingStarted(metadata=meta))
		reporter.receive(ProcessingFinished(frames_processed=5))
		assert spy.bars[0].closed is True

	def test_failed_closes_the_bar_and_writes_the_error(self) -> None:
		buf = io.StringIO()
		spy = _BarSpy()
		reporter = ConsoleProgressReporter(stream=buf, bar_factory=spy)
		meta = VideoMetadata(width=64, height=48, fps=30.0, total_frames=5)
		reporter.receive(ProcessingStarted(metadata=meta))
		reporter.receive(ProcessingFailed(frame_index=7, error="boom"))
		assert spy.bars[0].closed is True
		assert "7" in buf.getvalue()
		assert "boom" in buf.getvalue()

	def test_frames_before_started_are_ignored(self) -> None:
		spy = _BarSpy()
		ConsoleProgressReporter(bar_factory=spy).receive(_frame(1, 10))
		assert spy.bars == []  # no bar exists yet, so nothing to update

	def test_ignores_unknown_future_event(self) -> None:
		spy = _BarSpy()
		reporter = ConsoleProgressReporter(bar_factory=spy)

		class _FutureEvent(ProgressEvent):
			__slots__ = ()

		reporter.receive(_FutureEvent())
		assert spy.bars == []

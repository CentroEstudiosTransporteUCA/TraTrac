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
			frame_index=49, total_frames=100, timestamp_seconds=1.0, active_tracks=3
		)
		assert math.isclose(event.fraction, 0.5)
		assert math.isclose(event.percent, 50.0)

	def test_last_frame_reads_complete(self) -> None:
		event = FrameProcessed(
			frame_index=99, total_frames=100, timestamp_seconds=1.0, active_tracks=0
		)
		assert math.isclose(event.fraction, 1.0)

	def test_unknown_total_yields_zero_fraction(self) -> None:
		event = FrameProcessed(
			frame_index=10, total_frames=0, timestamp_seconds=1.0, active_tracks=0
		)
		assert event.fraction == 0.0

	def test_negative_total_yields_zero_fraction(self) -> None:
		# OpenCV's frame count can come back as -1 for some containers/streams.
		event = FrameProcessed(
			frame_index=10, total_frames=-1, timestamp_seconds=1.0, active_tracks=0
		)
		assert event.fraction == 0.0

	def test_fraction_clamped_to_one_when_count_under_reports(self) -> None:
		event = FrameProcessed(
			frame_index=150, total_frames=100, timestamp_seconds=1.0, active_tracks=0
		)
		assert event.fraction == 1.0


class TestNullProgressReporter:
	def test_discards_every_event_silently(self) -> None:
		reporter = NullProgressReporter()
		# None of these should raise or produce output.
		reporter.receive(ProcessingFinished(frames_processed=1))
		reporter.receive(
			FrameProcessed(frame_index=0, total_frames=1, timestamp_seconds=0.0, active_tracks=0)
		)


class TestConsoleProgressReporter:
	def test_started_writes_a_header(self) -> None:
		buf = io.StringIO()
		reporter = ConsoleProgressReporter(stream=buf)
		meta = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=300)
		reporter.receive(ProcessingStarted(metadata=meta))
		out = buf.getvalue()
		assert "300 frames" in out
		assert "1920x1080" in out

	def test_finished_writes_a_summary(self) -> None:
		buf = io.StringIO()
		ConsoleProgressReporter(stream=buf).receive(ProcessingFinished(frames_processed=42))
		assert "42" in buf.getvalue()

	def test_failed_writes_frame_and_error(self) -> None:
		buf = io.StringIO()
		ConsoleProgressReporter(stream=buf).receive(ProcessingFailed(frame_index=7, error="boom"))
		out = buf.getvalue()
		assert "7" in out
		assert "boom" in out

	def test_throttling_skips_rapid_frames_but_always_draws_the_final(self) -> None:
		buf = io.StringIO()
		# A huge interval means every non-final frame after the first is throttled.
		reporter = ConsoleProgressReporter(stream=buf, min_interval_seconds=1_000.0)

		reporter.receive(
			FrameProcessed(frame_index=0, total_frames=10, timestamp_seconds=0.0, active_tracks=1)
		)
		after_first = buf.getvalue()
		assert after_first != ""  # first frame always draws (last_draw == -inf)

		reporter.receive(
			FrameProcessed(frame_index=1, total_frames=10, timestamp_seconds=0.1, active_tracks=1)
		)
		assert buf.getvalue() == after_first  # within the throttle window: no redraw

		reporter.receive(
			FrameProcessed(frame_index=9, total_frames=10, timestamp_seconds=0.9, active_tracks=0)
		)
		assert buf.getvalue() != after_first  # final frame (fraction == 1.0) forces a draw
		assert "100.0%" in buf.getvalue()

	def test_ignores_unknown_future_event(self) -> None:
		buf = io.StringIO()
		reporter = ConsoleProgressReporter(stream=buf)

		class _FutureEvent(ProgressEvent):
			__slots__ = ()

		reporter.receive(_FutureEvent())
		assert buf.getvalue() == ""

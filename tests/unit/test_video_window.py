"""Tests for FrameWindow seconds->frame-range arithmetic."""

from __future__ import annotations

import pytest

from tratrac.infrastructure.video.window import FrameWindow


class TestFromSeconds:
	def test_no_bounds_spans_whole_video(self) -> None:
		w = FrameWindow.from_seconds(
			fps=30.0, total_frames=300, start_seconds=None, end_seconds=None
		)
		assert w.start_frame == 0
		assert w.end_frame is None
		assert w.frame_count == 300

	def test_start_only_seeks_and_counts_remaining(self) -> None:
		w = FrameWindow.from_seconds(
			fps=30.0, total_frames=300, start_seconds=1.0, end_seconds=None
		)
		assert w.start_frame == 30
		assert w.end_frame is None
		assert w.frame_count == 270

	def test_end_is_inclusive_of_its_frame(self) -> None:
		w = FrameWindow.from_seconds(fps=10.0, total_frames=300, start_seconds=0.3, end_seconds=0.6)
		# frames 3..6 inclusive
		assert w.start_frame == 3
		assert w.end_frame == 6
		assert w.frame_count == 4

	def test_end_clamped_to_last_frame(self) -> None:
		w = FrameWindow.from_seconds(
			fps=10.0, total_frames=50, start_seconds=None, end_seconds=999.0
		)
		assert w.end_frame == 49
		assert w.frame_count == 50

	def test_unknown_total_skips_bounds(self) -> None:
		w = FrameWindow.from_seconds(fps=10.0, total_frames=0, start_seconds=1.0, end_seconds=None)
		assert w.start_frame == 10
		assert w.end_frame is None
		assert w.frame_count is None  # no denominator without a real total

	def test_start_beyond_duration_raises(self) -> None:
		with pytest.raises(ValueError, match="beyond the video duration"):
			FrameWindow.from_seconds(
				fps=10.0, total_frames=50, start_seconds=10.0, end_seconds=None
			)

	def test_includes_respects_inclusive_end(self) -> None:
		w = FrameWindow.from_seconds(fps=10.0, total_frames=300, start_seconds=0.3, end_seconds=0.6)
		assert w.includes(6)
		assert not w.includes(7)

	def test_includes_always_true_without_end(self) -> None:
		w = FrameWindow.from_seconds(fps=10.0, total_frames=0, start_seconds=None, end_seconds=None)
		assert w.includes(1_000_000)

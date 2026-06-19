"""Tests for the shared DecimationGrid cadence helper."""

from __future__ import annotations

import pytest

from tratrac.infrastructure.cadence import DecimationGrid


def _accepted(grid: DecimationGrid, timestamps: list[float]) -> list[float]:
	return [t for t in timestamps if grid.accepts(t)]


class TestDecimationGrid:
	def test_first_timestamp_always_accepted(self) -> None:
		grid = DecimationGrid(min_interval_seconds=0.2, fps=10.0)
		assert grid.accepts(0.0) is True

	def test_thins_to_the_interval(self) -> None:
		# 10 fps frames (0.1 s apart), 0.2 s interval -> keep every other.
		grid = DecimationGrid(min_interval_seconds=0.2, fps=10.0)
		timestamps = [i / 10.0 for i in range(10)]
		assert _accepted(grid, timestamps) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])

	def test_interval_below_frame_duration_accepts_every_frame(self) -> None:
		grid = DecimationGrid(min_interval_seconds=0.05, fps=10.0)
		timestamps = [i / 10.0 for i in range(5)]
		assert _accepted(grid, timestamps) == pytest.approx(timestamps)

	def test_half_frame_snap_takes_the_nearest_frame(self) -> None:
		# Grid points 0.0, 0.25, 0.5; eps = 0.05. The frame at 0.2 (>= 0.25 - eps)
		# snaps to the 0.25 point, and 0.5 snaps to the 0.5 point -> no rounding-up drift.
		grid = DecimationGrid(min_interval_seconds=0.25, fps=10.0)
		assert _accepted(grid, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx([0.0, 0.2, 0.5])

	def test_reset_re_anchors(self) -> None:
		grid = DecimationGrid(min_interval_seconds=0.2, fps=10.0)
		assert grid.accepts(5.0) is True
		grid.reset()
		assert grid.accepts(9.0) is True  # first after reset always accepted

	def test_rejects_non_positive_interval(self) -> None:
		with pytest.raises(ValueError, match="min_interval_seconds"):
			DecimationGrid(min_interval_seconds=0.0, fps=10.0)

	def test_rejects_non_positive_fps(self) -> None:
		with pytest.raises(ValueError, match="fps"):
			DecimationGrid(min_interval_seconds=0.2, fps=0.0)

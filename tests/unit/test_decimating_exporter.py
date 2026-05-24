"""Tests for the timestep-decimating exporter decorator."""

from __future__ import annotations

from types import TracebackType

import pytest

from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.decimating import DecimatingTrajectoryExporter


class _RecordingExporter:
	"""Inner exporter that records the lifecycle calls and emitted timestamps."""

	def __init__(self) -> None:
		self.events: list[str] = []
		self.timestamps: list[float] = []

	def emit_frame(self, timestamp_seconds: float, states: list[VehicleState]) -> None:
		self.events.append("emit")
		self.timestamps.append(timestamp_seconds)

	def __enter__(self) -> _RecordingExporter:
		self.events.append("enter")
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		self.events.append("exit")


def _feed(decorator: DecimatingTrajectoryExporter, fps: float, n_frames: int) -> None:
	"""Push ``n_frames`` contiguous frames at ``fps`` through the decorator."""
	for i in range(n_frames):
		decorator.emit_frame(i / fps, [])


class TestConstruction:
	def test_rejects_non_positive_interval(self) -> None:
		with pytest.raises(ValueError, match="min_interval_seconds"):
			DecimatingTrajectoryExporter(_RecordingExporter(), min_interval_seconds=0.0, fps=30.0)

	def test_rejects_non_positive_fps(self) -> None:
		with pytest.raises(ValueError, match="fps"):
			DecimatingTrajectoryExporter(_RecordingExporter(), min_interval_seconds=0.1, fps=0.0)


class TestDecimation:
	def test_first_frame_is_always_emitted(self) -> None:
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=10.0, fps=30.0)
		dec.emit_frame(0.0, [])
		assert inner.timestamps == [0.0]

	def test_emits_on_the_interval_grid(self) -> None:
		# 30 fps, 0.1 s interval -> every 3rd frame: indices 0, 3, 6, 9.
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=0.1, fps=30.0)
		_feed(dec, fps=30.0, n_frames=10)
		assert inner.timestamps == pytest.approx([0 / 30, 3 / 30, 6 / 30, 9 / 30])

	def test_interval_at_or_below_frame_duration_emits_every_frame(self) -> None:
		# Interval finer than the 1/30 s frame spacing degrades to every frame.
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=0.01, fps=30.0)
		_feed(dec, fps=30.0, n_frames=5)
		assert inner.timestamps == pytest.approx([i / 30 for i in range(5)])

	def test_grid_is_anchored_at_the_first_timestamp(self) -> None:
		# A windowed run starts mid-video (first ts = 10.0); the grid anchors there.
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=0.1, fps=30.0)
		for i in range(10):
			dec.emit_frame(10.0 + i / 30, [])
		assert inner.timestamps == pytest.approx([10.0 + k / 30 for k in (0, 3, 6, 9)])

	def test_spacing_never_drops_below_the_interval(self) -> None:
		inner = _RecordingExporter()
		interval = 0.2
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=interval, fps=25.0)
		_feed(dec, fps=25.0, n_frames=100)
		gaps = [b - a for a, b in zip(inner.timestamps, inner.timestamps[1:], strict=False)]
		# Half-frame snapping means a gap can fall just under the nominal interval;
		# it must never fall below interval minus one frame.
		assert all(gap >= interval - 1.0 / 25.0 for gap in gaps)


class TestContextManager:
	def test_delegates_enter_and_exit(self) -> None:
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=0.1, fps=30.0)
		with dec:
			dec.emit_frame(0.0, [])
		assert inner.events == ["enter", "emit", "exit"]

	def test_resets_grid_on_reentry(self) -> None:
		# After a full context use, re-entering must emit the first frame again.
		inner = _RecordingExporter()
		dec = DecimatingTrajectoryExporter(inner, min_interval_seconds=10.0, fps=30.0)
		with dec:
			dec.emit_frame(0.0, [])
		with dec:
			dec.emit_frame(0.0, [])
		assert inner.timestamps == [0.0, 0.0]

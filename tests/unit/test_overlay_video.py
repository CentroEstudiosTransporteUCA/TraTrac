"""Tests for the overlay-video exporter.

cv2 is replaced by injected seams (``open_writer``, ``draw``) so these exercise
the adapter's orchestration — frame copy, trail accumulation, lifecycle — with no
codec or cv2 dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Transform2D, Vector2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.overlay_video import OverlayVideoExporter, TrailsView

_META = VideoMetadata(width=100, height=100, fps=10.0, total_frames=3)


def _frame(index: int = 0, value: int = 0) -> Frame:
	return Frame(index=index, pixels=np.full((100, 100, 3), value, dtype=np.uint8))


def _state(vehicle_id: int = 1, centroid: Point2D = Point2D(50.0, 50.0)) -> VehicleState:
	return VehicleState(
		vehicle_id=vehicle_id,
		timestamp_seconds=0.0,
		centroid=centroid,
		heading=Heading(1.0, 0.0),
		dimensions=Dimensions(length=4.0, width=2.0),
		velocity=Vector2D(0.0, 0.0),
		acceleration=0.0,
	)


class _FakeWriter:
	def __init__(self) -> None:
		self.frames: list[NDArray[np.uint8]] = []
		self.released = False

	def write(self, pixels: NDArray[np.uint8]) -> None:
		self.frames.append(pixels)

	def release(self) -> None:
		self.released = True


class _RecordingDraw:
	"""Captures each draw call and stamps a sentinel pixel to prove it got a copy."""

	def __init__(self) -> None:
		self.calls: list[
			tuple[list[VehicleState], float, dict[int, list[tuple[int, int]]], Transform2D]
		] = []

	def __call__(
		self,
		canvas: NDArray[np.uint8],
		states: list[VehicleState],
		scale: float,
		trails: TrailsView,
		to_raw: Transform2D,
	) -> None:
		canvas[0, 0, 0] = 255
		self.calls.append((list(states), scale, {k: list(v) for k, v in trails.items()}, to_raw))


def _exporter(
	writer: _FakeWriter,
	draw: _RecordingDraw,
	*,
	scale: float = 1.0,
	trail_length: int = 0,
	transform_source: object = None,
) -> OverlayVideoExporter:
	return OverlayVideoExporter(
		Path("unused.mp4"),
		_META,
		scale=scale,
		trail_length=trail_length,
		transform_source=transform_source,  # type: ignore[arg-type]
		open_writer=lambda _path, _meta: writer,
		draw=draw,
	)


class TestConstruction:
	def test_rejects_non_positive_scale(self) -> None:
		with pytest.raises(ValueError, match="Scale must be positive"):
			OverlayVideoExporter(Path("x.mp4"), _META, scale=0.0)

	def test_rejects_negative_trail_length(self) -> None:
		with pytest.raises(ValueError, match="trail_length"):
			OverlayVideoExporter(Path("x.mp4"), _META, scale=1.0, trail_length=-1)


class TestLifecycle:
	def test_enter_opens_and_exit_releases_writer(self) -> None:
		writer = _FakeWriter()
		with _exporter(writer, _RecordingDraw()):
			pass
		assert writer.released is True

	def test_emit_before_enter_raises(self) -> None:
		exporter = _exporter(_FakeWriter(), _RecordingDraw())
		with pytest.raises(RuntimeError, match="context manager"):
			exporter.emit_frame(0.0, [_state()], _frame())


class TestEmit:
	def test_writes_one_output_frame_per_emit(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw) as exporter:
			exporter.emit_frame(0.0, [_state()], _frame())
			exporter.emit_frame(0.1, [_state()], _frame())
		assert len(writer.frames) == 2

	def test_draws_on_a_copy_leaving_the_source_frame_untouched(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		frame = _frame(value=0)
		with _exporter(writer, draw) as exporter:
			exporter.emit_frame(0.0, [_state()], frame)
		# The draw seam stamped (0,0,0)=255 on the canvas it received...
		assert writer.frames[0][0, 0, 0] == 255
		# ...but the pipeline's frame must be unchanged (a composite peer may read it).
		assert frame.pixels[0, 0, 0] == 0

	def test_forwards_states_and_scale_to_draw(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		states = [_state(vehicle_id=7)]
		with _exporter(writer, draw, scale=0.5) as exporter:
			exporter.emit_frame(0.0, states, _frame())
		drawn_states, scale, _trails, _to_raw = draw.calls[0]
		assert [s.vehicle_id for s in drawn_states] == [7]
		assert scale == 0.5

	def test_centroid_converted_to_pixels_by_scale(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		# centroid 50 world units at 0.5 m/px -> pixel 100. No y-flip on the image.
		with _exporter(writer, draw, scale=0.5) as exporter:
			exporter.emit_frame(0.0, [_state(centroid=Point2D(50.0, 25.0))], _frame())
		_states, _scale, trails, _to_raw = draw.calls[0]
		assert trails[1] == [(100, 50)]

	def test_trails_accumulate_across_frames(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw) as exporter:
			exporter.emit_frame(0.0, [_state(centroid=Point2D(10.0, 10.0))], _frame())
			exporter.emit_frame(0.1, [_state(centroid=Point2D(20.0, 20.0))], _frame())
		assert draw.calls[1][2][1] == [(10, 10), (20, 20)]

	def test_only_visible_vehicles_appear_in_the_trails_view(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw) as exporter:
			exporter.emit_frame(0.0, [_state(vehicle_id=1)], _frame())
			exporter.emit_frame(0.1, [_state(vehicle_id=2)], _frame())
		# Frame 2 has only vehicle 2; vehicle 1's dead track must not ghost.
		assert set(draw.calls[1][2]) == {2}

	def test_trail_length_caps_the_rolling_window(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw, trail_length=2) as exporter:
			for i in range(3):
				exporter.emit_frame(
					i / 10.0, [_state(centroid=Point2D(float(i), float(i)))], _frame()
				)
		assert draw.calls[2][2][1] == [(1, 1), (2, 2)]


class TestReuse:
	def test_trails_reset_between_context_uses(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		exporter = _exporter(writer, draw)
		with exporter:
			exporter.emit_frame(0.0, [_state(centroid=Point2D(10.0, 10.0))], _frame())
		with exporter:
			exporter.emit_frame(0.0, [_state(centroid=Point2D(20.0, 20.0))], _frame())
		# Second use must start fresh: one point, not two.
		assert draw.calls[1][2][1] == [(20, 20)]


class TestAnnotateHook:
	def test_annotate_runs_after_draw_on_the_same_canvas_with_frame_index(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		seen: list[tuple[int, Transform2D]] = []

		def annotate(canvas: NDArray[np.uint8], frame_index: int, to_raw: Transform2D) -> None:
			assert canvas[0, 0, 0] == 255  # draw already ran on this canvas
			canvas[0, 0, 1] = 255
			seen.append((frame_index, to_raw))

		exporter = OverlayVideoExporter(
			Path("unused.mp4"),
			_META,
			scale=1.0,
			open_writer=lambda _path, _meta: writer,
			draw=draw,
			annotate=annotate,
		)
		with exporter:
			exporter.emit_frame(0.0, [_state()], _frame(index=4))

		assert [frame_index for frame_index, _ in seen] == [4]
		assert seen[0][1] == Transform2D.identity()  # same to_raw the draw seam got
		assert writer.frames[0][0, 0, 1] == 255  # annotate's mark reached the written frame

	def test_default_annotate_is_a_noop(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw) as exporter:  # no annotate injected
			exporter.emit_frame(0.0, [_state()], _frame())
		assert len(writer.frames) == 1


class TestTransformSource:
	def test_default_passes_identity_inverse_to_draw(self) -> None:
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw) as exporter:
			exporter.emit_frame(0.0, [_state()], _frame())
		assert draw.calls[0][3] == Transform2D.identity()

	def test_passes_the_inverse_of_the_current_transform(self) -> None:
		# A stabilized-frame→raw map: the draw seam must receive the inverse so it can
		# place stabilized coordinates back onto the raw frame.
		transform = Transform2D(a=1.0, b=0.0, tx=30.0, c=0.0, d=1.0, ty=-12.0)
		writer, draw = _FakeWriter(), _RecordingDraw()
		with _exporter(writer, draw, transform_source=lambda: transform) as exporter:
			exporter.emit_frame(0.0, [_state()], _frame())
		to_raw = draw.calls[0][3]
		assert to_raw.tx == pytest.approx(-30.0)
		assert to_raw.ty == pytest.approx(12.0)

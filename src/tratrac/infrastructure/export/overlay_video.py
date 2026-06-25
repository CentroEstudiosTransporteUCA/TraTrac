"""OverlayVideoExporter: renders each frame with its vehicles drawn on top and
writes the result to a video file.

It is a **standalone** renderer, not a ``TrajectoryExporter`` — rendering is a
post-hoc step, so this is driven directly by ``tratrac-render`` (which reads a
``.trj`` back into states via ``read_trj``), not composed into the pipeline. Its
``emit_frame`` therefore takes the ``Frame`` to draw on, unlike the frameless data
port. It draws on the **raw** frame. See vault/20_video_export.md.

Coordinates: ``VehicleState`` positions are in world units of the stabilized
(global) frame (a uniform ``scale`` metres-per-pixel multiple of pixels — no
homography yet, MVP1.x). To draw on the raw frame we divide by ``scale`` (no SSAM
y-flip; the image is y-down) and then map back onto the raw frame via the
ego-motion transform supplied by ``transform_source`` (identity when stabilization
is off). This keeps the overlay on the full, uncropped frame even when the drone
has drifted far from its first frame. See vault/05_75_mvp1_9.md.

cv2 lives only behind injected seams (``open_writer``, ``draw``, and the
``transform_source`` that maps stabilized coordinates back to the raw frame), so
the adapter's orchestration — frame copy, trail accumulation, coordinate mapping,
lifecycle — is unit-testable without cv2 or a codec.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.domain.vehicle import VehicleState


class FrameWriter(Protocol):
	"""Sink for rendered BGR frames. A cv2 ``VideoWriter`` satisfies this."""

	def write(self, pixels: NDArray[np.uint8]) -> None: ...

	def release(self) -> None: ...


# Per-vehicle trail points (stabilized/global image pixels) for the visible vehicles.
TrailsView = Mapping[int, Sequence[tuple[int, int]]]
OpenWriterFn = Callable[[Path, VideoMetadata], FrameWriter]
# Supplies the current frame's stabilized-frame→? transform; the overlay inverts it
# to map stabilized coordinates back onto the raw frame it is drawing on.
TransformSource = Callable[[], Transform2D]
DrawFn = Callable[[NDArray[np.uint8], list[VehicleState], float, TrailsView, Transform2D], None]


class OverlayVideoExporter:
	"""Writes a video of each frame with bumpers, IDs, and per-track trails drawn.

	Used as a context manager: entering opens the video writer, exiting releases
	it. ``emit_frame`` draws the frame's vehicles onto a copy of its pixels and
	writes one output frame. Trails accumulate per track across frames; with
	``trail_length`` 0 the whole path is kept, a positive value caps it to a
	rolling window of that many frames.
	"""

	def __init__(
		self,
		path: Path,
		metadata: VideoMetadata,
		*,
		scale: float,
		trail_length: int = 0,
		transform_source: TransformSource | None = None,
		open_writer: OpenWriterFn | None = None,
		draw: DrawFn | None = None,
	) -> None:
		if scale <= 0.0:
			raise ValueError(f"Scale must be positive, got {scale}.")
		if trail_length < 0:
			raise ValueError(f"trail_length must be >= 0 (0 = whole path), got {trail_length}.")
		self._path = path
		self._metadata = metadata
		self._scale = scale
		self._trail_maxlen = trail_length if trail_length > 0 else None
		# Default: identity, i.e. states are already in raw-frame coordinates (no
		# stabilization). When stabilization is on, the CLI supplies the ego-motion's
		# current transform so the overlay can map states back onto the raw frame.
		self._transform_source: TransformSource = (
			transform_source if transform_source is not None else Transform2D.identity
		)
		self._open_writer: OpenWriterFn = (
			open_writer if open_writer is not None else _cv2_open_writer
		)
		self._draw: DrawFn = draw if draw is not None else _cv2_draw
		self._writer: FrameWriter | None = None
		self._trails: dict[int, deque[tuple[int, int]]] = {}

	def __enter__(self) -> OverlayVideoExporter:
		self._writer = self._open_writer(self._path, self._metadata)
		# Reset trails so the exporter is reusable across context-manager uses.
		self._trails = defaultdict(self._new_trail)
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._writer is not None:
			self._writer.release()
			self._writer = None

	def emit_frame(
		self, timestamp_seconds: float, states: list[VehicleState], frame: Frame
	) -> None:
		del timestamp_seconds  # the video carries time implicitly via frame order
		writer = self._require_writer()
		# Copy so cv2's in-place drawing never mutates the frame the pipeline owns
		# (other exporters in a composite may read the same Frame).
		canvas: NDArray[np.uint8] = frame.pixels.copy()
		# Trails are stored in stabilized (global) pixels; mapping the whole path
		# through the current frame's inverse transform shows the world path from the
		# current camera pose. Identity when stabilization is off.
		to_raw = self._transform_source().inverse()
		visible: dict[int, Sequence[tuple[int, int]]] = {}
		for state in states:
			centroid = (
				round(state.centroid.x / self._scale),
				round(state.centroid.y / self._scale),
			)
			trail = self._trails[state.vehicle_id]
			trail.append(centroid)
			# Snapshot as a list: decouples the draw seam from the live deque and
			# keeps only currently-visible tracks (dead tracks stop ghosting).
			visible[state.vehicle_id] = list(trail)
		self._draw(canvas, states, self._scale, visible, to_raw)
		writer.write(canvas)

	def _new_trail(self) -> deque[tuple[int, int]]:
		return deque(maxlen=self._trail_maxlen)

	def _require_writer(self) -> FrameWriter:
		if self._writer is None:
			raise RuntimeError("OverlayVideoExporter must be used as a context manager.")
		return self._writer


def _color_for(vehicle_id: int) -> tuple[int, int, int]:
	"""Deterministic BGR colour from a vehicle id (golden-ratio hue spread)."""
	hue = (vehicle_id * 0.618033988749895) % 1.0
	r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
	return (int(b * 255), int(g * 255), int(r * 255))


def _world_to_raw(
	point_x: float, point_y: float, scale: float, to_raw: Transform2D
) -> tuple[int, int]:
	"""World units -> raw-frame pixels. Divide by scale (no y-flip; image is y-down),
	then map stabilized coordinates back onto the raw frame via ``to_raw``."""
	raw = to_raw.apply(Point2D(point_x / scale, point_y / scale))
	return (round(raw.x), round(raw.y))


def _cv2_draw(
	canvas: NDArray[np.uint8],
	states: list[VehicleState],
	scale: float,
	trails: TrailsView,
	to_raw: Transform2D,
) -> None:
	"""Draw bumpers, orientation line, ID+speed label, and trails onto ``canvas``."""
	import cv2  # lazy: keeps the module (and unit tests) free of the cv2 import

	for state in states:
		color = _color_for(state.vehicle_id)
		fx, fy = _world_to_raw(state.front_bumper.x, state.front_bumper.y, scale, to_raw)
		rx, ry = _world_to_raw(state.rear_bumper.x, state.rear_bumper.y, scale, to_raw)
		cv2.line(canvas, (rx, ry), (fx, fy), color, 2)
		cv2.circle(canvas, (fx, fy), 5, color, -1)
		cv2.circle(canvas, (rx, ry), 5, color, 2)
		cv2.putText(
			canvas,
			f"v{state.vehicle_id}  {state.speed:.0f}",
			(fx + 8, max(fy - 8, 14)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)

	for vehicle_id, points in trails.items():
		if len(points) < 2:
			continue
		color = _color_for(vehicle_id)
		mapped = [to_raw.apply(Point2D(float(px), float(py))) for px, py in points]
		pixels = [(round(p.x), round(p.y)) for p in mapped]
		for i in range(1, len(pixels)):
			cv2.line(canvas, pixels[i - 1], pixels[i], color, 3)


def _cv2_open_writer(path: Path, metadata: VideoMetadata) -> FrameWriter:
	"""Open an mp4v ``cv2.VideoWriter`` matching the source's size and fps."""
	import cv2

	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(path), fourcc, metadata.fps, (metadata.width, metadata.height))
	if not writer.isOpened():
		raise RuntimeError(f"Could not open a video writer for {path}.")
	return writer

"""OverlayVideoExporter: renders each frame with its vehicles drawn on top and
writes the result to a video file.

Produces an overlay video — each frame with its vehicles drawn on top
(front/rear bumpers, orientation line, ``vN speed`` label, per-track trails) — after
a run, via the standalone ``tratrac-render`` tool (``cli_render.py``), which reads the
run's SSAM ``.trj`` (plus, for an ego-motion or projected run, the transforms file) and
draws over the source clip. This is a debug/visualization output, not a new analytics
format: neither the SSAM export (A) nor the extended internal export (B) of the
dual-export architecture (``src/tratrac/domain/ARCHITECTURE.md``), just a rendering of
the trajectories. New analytics data still goes into (B), never here.

Why post-hoc: originally the overlay ran inside the per-frame loop — an
``OverlayVideoExporter`` composed onto the ``.trj`` exporter, so each run copied the
frame, drew on it, and encoded a video frame on every iteration, paying a large
per-frame cost for output that's fully derivable from the ``.trj`` afterward. So it was
pulled out: the run now only detects, tracks, and exports the ``.trj`` (+ optional
sidecars), no per-frame drawing or encoding; ``tratrac-render`` does the drawing as a
separate step, reusing this same drawing engine, with only the *driver* changed
(vehicle states come from reading a ``.trj``, not from live tracking);
``tratrac-postprocess`` likewise dropped its in-process ``--video-out``. This makes
rendering a normal downstream step alongside ``validate_trj``/``plot_run``, not a tax
on every run. The old in-pipeline ``CompositeTrajectoryExporter`` was removed along
with it: with rendering gone from both the live run and the smoother, nothing composes
exporters anymore.

It is a **standalone** renderer, not a ``TrajectoryExporter`` — the ``TrajectoryExporter``
port is ``emit_frame(timestamp_seconds, states)``, a pure data port with no pixels; it
was briefly widened to carry a ``Frame`` so the overlay could compose into the
pipeline, but with rendering now post-hoc that parameter was removed and this class is
no longer a ``TrajectoryExporter``. Its own ``emit_frame(timestamp, states, frame)``
takes the pixels to draw on, driven directly by ``tratrac-render`` (which reads a
``.trj`` back into states via ``read_trj`` in ``ssam_trj.py`` — the one place a
``.trj`` is read back, viz-only, not reopening the export-only invariant; see that
module's docstring). So the data exporters (``SsamTrjExporter``,
``DecimatingTrajectoryExporter``, ``TimedExporter``) stay frameless, and this is the
one class that needs pixels and carries them explicitly.

Coordinates: ``VehicleState`` positions are in world units of the stabilized
(global) frame (a uniform ``scale`` metres-per-pixel multiple of pixels — no
homography yet, MVP1.x). To draw on the raw frame we divide by ``scale`` (no SSAM
y-flip; the image is y-down) and then map back onto the raw frame via the
ego-motion transform supplied by ``transform_source`` (identity when stabilization
is off). This keeps the overlay on the full, uncropped frame even when the drone
has drifted far from its first frame. Trails are stored in stabilized coordinates and
mapped through the *current* inverse each frame, showing the world path from the
current camera pose; validator violation marks (``--violations``) ride the same
transform, landing in the same raw-frame space as the trajectories. See
``infrastructure/video/ego_motion_orb.py``'s module docstring.

**This "divide by scale" recovery is stale for every ``.trj``, not just a
homography-projected one.** ``cli_postprocess.postprocess`` always projects every
observation through the transforms file's ``TransformTable`` before smoothing, and
always shifts the result to a 0-origin extent (``_normalize_world_recording``) — even a
plain GSD-scale run's ``.trj`` positions are no longer the source video's own pixel
coordinates times one constant, they're shifted by an amount this exporter has no way
to recover (it isn't stored in the ``.trj``, and can't be — see
``application/SMOOTHING.md``'s "Dual-space export" section for why the shift is only
ever undone by inverting the *exact* projector object a postprocess run built, which
``tratrac-render`` never has). ``tratrac-render`` reading ``--smoothed-record`` instead
of ``--trj`` (always raw image-space pixels, purpose-built for exactly this kind of
consumer) would fix this for every case, scale or homography alike — not done yet.

cv2 and PyAV live only behind injected seams (``open_writer``, ``draw``, ``annotate``,
and the ``transform_source`` that maps stabilized coordinates back to the raw frame),
so the adapter's orchestration — frame copy, per-track trail accumulation
(``trail_length`` 0 = whole path, N = rolling window of N frames; only
currently-visible tracks are drawn so dead tracks stop ghosting), coordinate mapping,
lifecycle — is unit-testable without either.

The default writer encodes with PyAV (libx264, CRF 23, preset ``medium``) rather than
cv2's ``VideoWriter``: this project's ``opencv-python`` build has no software H.264
encoder registered (``avc1``/``h264``/``X264`` tags all fail to open), so
``cv2.VideoWriter`` fell back to MPEG-4 Part 2 ("mp4v") with no bitrate/CRF control —
measured, overlay videos were routinely 3-5x the size of their source clip at
identical resolution/fps (864 MB source -> 2.65 GB overlay, mpeg4 at 23 Mbps vs. the
source's h264 at 7.6 Mbps). PyAV binds FFmpeg's libraries directly (no subprocess/pipe
to manage) and its wheels bundle libx264, so it isn't subject to the same gap. This
was also the first adoption of PyAV, which ``docs/TECH_STACK.md`` names as the target
video I/O library ("NVIDIA NVDEC + PyAV") but which nothing had wired in yet — that gap
had gone untracked, unlike the RT-DETR->YOLOv8 detector swap
(``infrastructure/detection/DETECTOR_CHOICE.md``), which documents its override
explicitly. Only the **encode** side moved to PyAV; ``OpenCvVideoSource`` (decode,
seeking, ``--process-fps`` frame-skipping) and the drawing primitives
(``cv2.line``/``circle``/``putText``) stay on cv2 — see Group B4 in GitHub Issues for
the deferred full-decode TorchCodec+NVDEC migration.

Violations in the same pass: ``tratrac-render --violations CSV`` (a
``validate_trj.py`` violations CSV, optionally filtered by ``--checks``) marks each
non-compliant instance in red in the same render pass as the trajectories — one
encode, "frame + trajectories + violations." ``cli_render.py`` reuses this class's
``annotate`` seam (a generic post-draw hook ``(canvas, frame_index, to_raw)``):
it buckets violation rows onto absolute frames by ``round(timestamp_s * fps)`` and
the hook draws each frame's marks on top of the trajectories, mapped to raw via the
same ``to_raw``. This replaced the old standalone ``scripts/render_violations.py``,
which required a second encode over the overlay.

``cli_render.py`` (``tratrac-render``) is the driver: reads the ``.trj`` (``read_trj``),
the optional transforms file, and the optional violations CSV, buckets states (and
violation marks) onto absolute video frames by ``round(timestamp * fps)`` (fps from
the *clip* — the ``.trj`` carries time but not fps), and drives a single instance of
this class. It opens the clip windowed to the ``.trj``'s covered span (the
absolute-seconds TIMESTEPs bound ``start_seconds``/``end_seconds``, plus a small tail
buffer), so a short analysis window on a long clip renders only that span instead of
re-encoding the whole video; an empty ``.trj`` renders nothing. Invocation:
``tratrac-render VIDEO --trj RUN.trj --out OVERLAY.mp4 [--transforms
TRANSFORMS.jsonl] [--trail N] [--force]`` — ``--out`` must not pre-exist without
``--force``; pass ``--transforms`` (the ``tratrac-preprocess estimate`` run's
transforms file that fed ``input.transforms_in``) so the global-frame trajectories map
back onto the raw video, omitting it only for a non-stabilized (no similarity rows)
run.

Interaction with decimation: ``tratrac-render`` draws states where the ``.trj`` has
them. If the run used ``export.timestep_precision`` (or ``input.process_fps``), the
``.trj`` only carries states on the emitted timesteps, so the overlay shows
bumpers/trails only on those frames and bare frames in between. For a smooth,
full-cadence overlay, render from a ``.trj`` produced with ``timestep_precision = 0``.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

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
# Optional post-draw hook: (canvas, frame index, global->raw transform). Runs after the
# trajectory draw, before the frame is written, so callers can annotate on top (e.g.
# tratrac-render drawing validator violation marks). The transform is the same one the
# trajectories were mapped with, so annotations land in the same raw-frame space.
AnnotateFn = Callable[[NDArray[np.uint8], int, Transform2D], None]


def _no_annotate(canvas: NDArray[np.uint8], frame_index: int, to_raw: Transform2D) -> None:
	"""Default annotation hook: draw nothing."""


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
		annotate: AnnotateFn | None = None,
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
			open_writer if open_writer is not None else _pyav_open_writer
		)
		self._draw: DrawFn = draw if draw is not None else _cv2_draw
		self._annotate: AnnotateFn = annotate if annotate is not None else _no_annotate
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
		# Post-draw annotations (e.g. violation marks) land on top of the trajectories,
		# in the same raw-frame space (mapped via the same to_raw).
		self._annotate(canvas, frame.index, to_raw)
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


class _PyAvFrameWriter:
	"""Adapts a PyAV output container + video stream to the ``FrameWriter`` protocol.

	``container``/``stream`` are PyAV objects; PyAV ships no type stubs, so they're
	held as ``Any`` (matching cv2's `follow_imports = "skip"` treatment elsewhere).
	"""

	def __init__(self, container: Any, stream: Any) -> None:
		self._container = container
		self._stream = stream

	def write(self, pixels: NDArray[np.uint8]) -> None:
		import av

		frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
		for packet in self._stream.encode(frame):
			self._container.mux(packet)

	def release(self) -> None:
		for packet in self._stream.encode():  # flush buffered frames
			self._container.mux(packet)
		self._container.close()


def _pyav_open_writer(path: Path, metadata: VideoMetadata) -> FrameWriter:
	"""Open a libx264 writer matching the source's size and fps.

	CRF 23 / preset "medium" is libx264's own quality-oriented default operating
	point, empirically close to typical drone-footage delivery bitrates (see this
	module's own docstring above) rather than a size or bitrate target of its own.
	"""
	import av

	container = av.open(str(path), mode="w")
	stream = container.add_stream("libx264", rate=Fraction(metadata.fps).limit_denominator(1000))
	stream.width = metadata.width
	stream.height = metadata.height
	stream.pix_fmt = "yuv420p"
	stream.options = {"crf": "23", "preset": "medium"}
	return _PyAvFrameWriter(container, stream)

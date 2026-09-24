"""SSAM .trj v1.04 binary exporter (and a viz-only reader).

MVP1 conventions: little-endian, metric units, image-space Y flipped into SSAM
Cartesian, Link ID = 0, Lane ID = 0. The ``scale`` (metres per pixel) is a required
constructor argument — there is no default; the caller supplies the resolved GSD.
``tratrac`` itself no longer defaults to or requires calibration directly — GSD scale
is resolved once, upstream, by ``tratrac-preprocess estimate``, and this exporter (used
only by ``tratrac-postprocess``) reads the resolved value out of the shared transforms
file (see ``application/config.py``'s module docstring, "What isn't here"). Scale = 1.0
pixel mode remains available at the library level for callers that knowingly want it.

``read_trj`` reads a written ``.trj`` back into ``VehicleState``s. It exists
*solely* for rendering/diagnostics (e.g. ``tratrac-render`` drawing trajectories
over a clip) and does **not** reopen the load-bearing invariant that the SSAM
``.trj`` is export-only, never re-ingested into the processing/analytics path
(see src/tratrac/domain/ARCHITECTURE.md and src/tratrac/application/SMOOTHING.md):
smoothing and analytics consume the raw track sidecar, not the lossy ``.trj``.
Reconstruction is float32-exact for the pixel-rounded drawing the renderer does.

Format basics: binary, record-based, extension ``.trj``/``.TRJ``. No header magic — the
first byte is the FORMAT record-type byte (0). The file ends when no more data is
available, no explicit EOF record. Endianness is declared in the FORMAT record, not
fixed. Type encodings: Byte (1 byte, unsigned), Integer (4 bytes, signed two's
complement), Float (4 bytes, signed IEEE 754 single precision) — multi-byte values
honor the FORMAT record's endianness byte, itself the ASCII character ``L`` (0x4C) or
``B`` (0x42).

Authoritative sources for the byte layout below, in this same directory: ``SSAM File
Format v1.04.pdf`` (original Siemens/Gardner Consulting spec, 2004) and ``Open Source
SSAM File Format v3.0.pdf`` (New Global Systems update, 2017, backward compatible with
1.04, adds elevation). If this docstring and the PDFs disagree, the PDFs win.

File layout::

    FORMAT
    DIMENSIONS
    TIMESTEP
        VEHICLE
        VEHICLE
        ...
    TIMESTEP
        VEHICLE
        ...
    ... (more timesteps until EOF)

Exactly one FORMAT record, exactly one DIMENSIONS record, then alternating: one
TIMESTEP record followed by a variable number of VEHICLE records. The reader uses the
leading record-type byte to dispatch.

Record FORMAT (id = 0). v1.04 layout, 6 bytes: Record Type (Byte, 0), Endian (Byte,
ASCII ``L``/``B``), Version (Float, 1.04). v3.0 layout, 7 bytes, adds Z Value Option
(Byte: 0/blank = no Z in VEHICLE records, non-zero = Z values appended to each one).

Record DIMENSIONS (id = 1), 22 bytes: Record Type (Byte, 1); Units (Byte, 0 = English
feet/ft-s/ft-s2, 1 = Metric m/m-s/m-s2); Scale (Float, distance per X or Y unit "per
pixel" — real distance = value x Scale); MinX/MinY/MaxX/MaxY (Integer, the observation
area's left/bottom/right/top edges). Constraints: observation area < 10 sq miles;
coordinate system is Cartesian with X right, Y up.

Record TIMESTEP (id = 2), 5 bytes: Record Type (Byte, 2), Timestep (Float, seconds
since start of simulation/observation). Sub-second precision (~1/10 s) is the practical
minimum; once-per-second is too coarse for conflict analysis.

Record VEHICLE (id = 3). v1.04 layout, 42 bytes: Record Type (Byte, 3); Vehicle ID
(Integer); Link ID (Integer, road link identifier where available); Lane ID (Byte,
lane identifier where available — 1 byte, max 255); Front X/Front Y/Rear X/Rear Y
(Float, scaled — real X = value x DIMENSIONS.Scale); Length/Width (Float, **unscaled**,
in DIMENSIONS.Units); Speed (Float, unscaled, units/sec); Acceleration (Float,
unscaled, units/sec2). v3.0 supplemental fields, appended after Acceleration when
FORMAT.Z Value Option != 0: Front Z, Rear Z (Float, same units as DIMENSIONS).
Recommended Z conventions when real elevations are unavailable: -1 = underpass, 0 =
ground level, +1 = overpass, +-1 per additional stacked level.

Scaling semantics, easy to get wrong: the DIMENSIONS.Scale field decouples grid units
from physical units. X and Y in VEHICLE records are in grid units (think: pixel
indices); real-world distance = grid value x Scale. Length, Width, Speed, Acceleration
are already in physical units (DIMENSIONS.Units) and ignore Scale. So if Scale = 0.25
and Units = Metric, Front X = 4 means the bumper is at 1.0 m on the real X axis.

Coordinate orientation vs. image space: SSAM is Cartesian, Y grows up; aerial video is
image pixels, Y grows down. The exporter flips Y: ``y_ssam = image_height - y_image``
(after any cropping/stabilization adjustments).

MVP1 approximation strategy: MVP1 emits valid v1.04 (no Z, no multi-plane). Endianness
``L`` (dev machine is x86_64); Units Metric (1); Scale 1.0 (one grid unit per "meter" —
MVP1 has no calibration, so pixels are pretended to be meters, syntactically valid but
physically meaningless until MVP2); DIMENSIONS bounds MinX=0, MinY=0,
MaxX=image_width, MaxY=image_height; Link ID 0 (no road network); Lane ID 0 (until Link
ID / Lane ID assignment lands, see ``application/ROAD_GRAPH.md``); Front/Rear X/Y image
pixels after Y-flip; Length/Width bounding-box dimensions in pixels (treated as meters
— known wrong until calibration); Speed windowed pixel-displacement/time (treated as
m/s — known wrong until MVP1.75 calibration). Acceleration is the **longitudinal**
acceleration = rate of change of speed (d|v|/dt), a scalar in units/sec2 — a windowed
finite-difference of the Speed signal above, **not** a vector projected onto the
heading, so a vehicle turning at constant speed reports ~0 where a heading projection
would fire spuriously as a lagging heading estimate chases the rotating velocity.
``VehicleState.acceleration`` carries this scalar directly. Timestep is
``frame_index / fps``, Float seconds.

MVP2 world mode (post-hoc projection, see ``application/WORLD_PROJECTION.md``): when
``tratrac-postprocess --transforms`` projects to world metres, Scale = 1.0 and the
coordinates are *already* metric — so the DIMENSIONS bounds are not the pixel grid.
They are the world-coordinate bounding box in metres (MinX=MinY=0 after the projected
points are translated to a non-negative origin; MaxX/MaxY = the padded world extent),
and the Y-flip is about that world MaxY. Reusing the image dimensions here would leave
a third-party reader interpreting metric coordinates against a pixel-sized canvas
flipped about the wrong axis.

Record sizes: FORMAT 6 B (v1.04) / 7 B (v3.0 with Z); DIMENSIONS 22 B / 22 B; TIMESTEP
5 B / 5 B; VEHICLE 42 B / 50 B. A 30 s clip at 30 FPS with ~50 vehicles per frame is
about ``(5 + 50 x 42) x 900 ~= 1.9 MB`` for v1.04.

Out of scope for this docstring: the SSAM **Path** file format (``.pth``) is mentioned
in v1.04 but not specified; the SSAM **conflict output** format (``.csa``) is not part
of trajectory input.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO

from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Dimensions, Heading, Point2D
from tratrac.domain.vehicle import VehicleState

_FORMAT_RECORD_TYPE = 0
_DIMENSIONS_RECORD_TYPE = 1
_TIMESTEP_RECORD_TYPE = 2
_VEHICLE_RECORD_TYPE = 3

_LITTLE_ENDIAN_FLAG = ord("L")
_METRIC_UNITS_FLAG = 1
_VERSION_1_04 = 1.04

_FORMAT_STRUCT = struct.Struct("<BBf")
_DIMENSIONS_STRUCT = struct.Struct("<BBfiiii")
_TIMESTEP_STRUCT = struct.Struct("<Bf")
_VEHICLE_STRUCT = struct.Struct("<BiiBffffffff")


class SsamTrjExporter:
	"""Writes SSAM .trj v1.04 binary trajectory files.

	Used as a context manager. Entering writes FORMAT + DIMENSIONS; exiting closes
	the file. ``emit_frame`` writes one TIMESTEP and one VEHICLE record per state.
	"""

	def __init__(self, path: Path, metadata: VideoMetadata, *, scale: float) -> None:
		if scale <= 0.0:
			raise ValueError(f"Scale must be positive, got {scale}.")
		self._path = path
		self._metadata = metadata
		self._scale = scale
		self._file: IO[bytes] | None = None

	def __enter__(self) -> SsamTrjExporter:
		self._file = self._path.open("wb")
		self._write_format_record()
		self._write_dimensions_record()
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		if self._file is not None:
			self._file.close()
			self._file = None

	def emit_frame(self, timestamp_seconds: float, states: list[VehicleState]) -> None:
		out = self._require_file()
		out.write(_TIMESTEP_STRUCT.pack(_TIMESTEP_RECORD_TYPE, timestamp_seconds))
		for state in states:
			self._write_vehicle_record(state)

	def _write_format_record(self) -> None:
		self._require_file().write(
			_FORMAT_STRUCT.pack(_FORMAT_RECORD_TYPE, _LITTLE_ENDIAN_FLAG, _VERSION_1_04)
		)

	def _write_dimensions_record(self) -> None:
		self._require_file().write(
			_DIMENSIONS_STRUCT.pack(
				_DIMENSIONS_RECORD_TYPE,
				_METRIC_UNITS_FLAG,
				self._scale,
				0,
				0,
				self._metadata.width,
				self._metadata.height,
			)
		)

	def _write_vehicle_record(self, state: VehicleState) -> None:
		front = state.front_bumper
		rear = state.rear_bumper
		scale = self._scale
		# image_height comes in as pixels; convert to world units (matches the
		# units of state.{centroid, dimensions, ...}) so the y-flip subtraction
		# stays unit-consistent. Then divide by scale to get back to the grid
		# coordinates SSAM stores.
		image_height_world = self._metadata.height * scale
		self._require_file().write(
			_VEHICLE_STRUCT.pack(
				_VEHICLE_RECORD_TYPE,
				state.vehicle_id,
				state.link_id,
				state.lane_id,
				front.x / scale,
				(image_height_world - front.y) / scale,
				rear.x / scale,
				(image_height_world - rear.y) / scale,
				state.dimensions.length,
				state.dimensions.width,
				state.speed,
				state.acceleration,
			)
		)

	def _require_file(self) -> IO[bytes]:
		if self._file is None:
			raise RuntimeError("SsamTrjExporter must be used as a context manager.")
		return self._file


@dataclass(frozen=True, slots=True)
class TrjFrame:
	"""One TIMESTEP read back: its time and the vehicle states at that instant."""

	timestamp_seconds: float
	states: list[VehicleState]


@dataclass(frozen=True, slots=True)
class TrjRecording:
	"""A ``.trj`` read back into memory. ``scale`` and ``width``/``height`` come from
	the DIMENSIONS record, enough to reconstruct world-unit states for rendering."""

	scale: float
	width: int
	height: int
	frames: list[TrjFrame]


def read_trj(path: Path) -> TrjRecording:
	"""Read an SSAM ``.trj`` v1.04 back into ``VehicleState``s (viz-only — see module docstring).

	Inverts ``_write_vehicle_record`` exactly: grid coordinates are multiplied by the
	DIMENSIONS scale and the SSAM Y-flip is undone, recovering world-unit front/rear
	bumpers, from which the centroid, heading, and dimensions are reconstructed.

	Raises ``ValueError`` (re-wrapped with the path) on a truncated file or an
	unexpected record ordering (e.g. a VEHICLE before any TIMESTEP).
	"""
	data = path.read_bytes()
	try:
		offset = _FORMAT_STRUCT.size  # FORMAT record; its contents are not needed back
		_, _, scale, _, _, width, height = _DIMENSIONS_STRUCT.unpack_from(data, offset)
		offset += _DIMENSIONS_STRUCT.size
		image_height_world = height * scale

		frames: list[TrjFrame] = []
		current: list[VehicleState] | None = None
		while offset < len(data):
			record_type = data[offset]
			if record_type == _TIMESTEP_RECORD_TYPE:
				_, timestamp = _TIMESTEP_STRUCT.unpack_from(data, offset)
				offset += _TIMESTEP_STRUCT.size
				current = []
				frames.append(TrjFrame(timestamp_seconds=timestamp, states=current))
			elif record_type == _VEHICLE_RECORD_TYPE:
				if current is None:
					raise ValueError("VEHICLE record before any TIMESTEP record.")
				fields = _VEHICLE_STRUCT.unpack_from(data, offset)
				offset += _VEHICLE_STRUCT.size
				current.append(_vehicle_state_from_record(fields, scale, image_height_world))
			else:
				raise ValueError(f"unknown record type {record_type} at byte {offset}.")
	except struct.error as exc:
		raise ValueError(f"{path} is a truncated or malformed .trj: {exc}") from exc
	except ValueError as exc:
		raise ValueError(f"{path} is not a valid .trj: {exc}") from exc
	return TrjRecording(scale=scale, width=width, height=height, frames=frames)


def _vehicle_state_from_record(
	fields: tuple[int, int, int, int, float, float, float, float, float, float, float, float],
	scale: float,
	image_height_world: float,
) -> VehicleState:
	"""Rebuild a ``VehicleState`` from one unpacked VEHICLE record (inverse of the writer)."""
	(_, vehicle_id, link_id, lane_id, fx, fy, rx, ry, length, width, speed, acceleration) = fields
	# Undo "divide by scale" and the SSAM Y-flip the writer applied.
	front = Point2D(fx * scale, image_height_world - fy * scale)
	rear = Point2D(rx * scale, image_height_world - ry * scale)
	centroid = Point2D((front.x + rear.x) / 2.0, (front.y + rear.y) / 2.0)
	axis = rear.displacement_to(front)
	# Dimensions.length > 0 guarantees front != rear, so the axis normalizes; the
	# fallback only guards a corrupt/degenerate file.
	heading = axis.normalized() if axis.magnitude > 0.0 else Heading(1.0, 0.0)
	return VehicleState(
		vehicle_id=vehicle_id,
		timestamp_seconds=0.0,  # the TIMESTEP carries time; the per-vehicle copy is unused
		centroid=centroid,
		heading=heading,
		dimensions=Dimensions(length=length, width=width),
		velocity=heading.as_vector_with_magnitude(speed),
		acceleration=acceleration,
		link_id=link_id,
		lane_id=lane_id,
	)

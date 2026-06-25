"""SSAM .trj v1.04 binary exporter (and a viz-only reader).

Spec lives in vault/04_ssam_format.md. MVP1 conventions are documented there:
little-endian, metric units, image-space Y flipped into SSAM Cartesian, Link
ID = 0, Lane ID = 0. The ``scale`` (metres per pixel) is a required constructor
argument — there is no default; the caller supplies the resolved GSD.

``read_trj`` reads a written ``.trj`` back into ``VehicleState``s. It exists
*solely* for rendering/diagnostics (e.g. ``tratrac-render`` drawing trajectories
over a clip) and does **not** reopen the load-bearing invariant that the SSAM
``.trj`` is export-only, never re-ingested into the processing/analytics path
(see vault/01 and vault/22): smoothing and analytics consume the raw track
sidecar, not the lossy ``.trj``. Reconstruction is float32-exact for the
pixel-rounded drawing the renderer does.
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

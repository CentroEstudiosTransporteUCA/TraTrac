"""SSAM .trj v1.04 binary exporter.

Spec lives in vault/04_ssam_format.md. MVP1 conventions are documented there:
little-endian, metric units, scale=1.0 by default, image-space Y flipped into
SSAM Cartesian, Link ID = 0, Lane ID = 0.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import TracebackType
from typing import IO

from tratrac.domain.frame import VideoMetadata
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

	def __init__(self, path: Path, metadata: VideoMetadata, *, scale: float = 1.0) -> None:
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
		# Y-flip image -> SSAM Cartesian; then divide by scale to get grid units.
		image_height = self._metadata.height
		scale = self._scale
		self._require_file().write(
			_VEHICLE_STRUCT.pack(
				_VEHICLE_RECORD_TYPE,
				state.vehicle_id,
				0,  # Link ID — MVP1 has no road network.
				0,  # Lane ID — MVP1 has no lane assignment.
				front.x / scale,
				(image_height - front.y) / scale,
				rear.x / scale,
				(image_height - rear.y) / scale,
				state.dimensions.length,
				state.dimensions.width,
				state.speed,
				state.forward_acceleration,
			)
		)

	def _require_file(self) -> IO[bytes]:
		if self._file is None:
			raise RuntimeError("SsamTrjExporter must be used as a context manager.")
		return self._file

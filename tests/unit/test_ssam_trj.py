"""Byte-level tests for the SSAM .trj v1.04 binary exporter.

The spec we are verifying against lives in vault/04_ssam_format.md.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter

_FORMAT_SIZE = 6
_DIMENSIONS_SIZE = 22
_TIMESTEP_SIZE = 5
_VEHICLE_SIZE = 42

_HEADER_SIZE = _FORMAT_SIZE + _DIMENSIONS_SIZE


def _vehicle(
	vehicle_id: int = 1,
	timestamp: float = 0.0,
	centroid: Point2D = Point2D(50.0, 50.0),
	heading: Heading = Heading(1.0, 0.0),
	length: float = 4.0,
	width: float = 2.0,
	velocity: Vector2D = Vector2D(0.0, 0.0),
	acceleration: Vector2D = Vector2D(0.0, 0.0),
) -> VehicleState:
	return VehicleState(
		vehicle_id=vehicle_id,
		timestamp_seconds=timestamp,
		centroid=centroid,
		heading=heading,
		dimensions=Dimensions(length=length, width=width),
		velocity=velocity,
		acceleration=acceleration,
	)


class TestHeader:
	def test_format_record_is_v1_04_little_endian(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=100, height=200, fps=30.0, total_frames=1)
		with SsamTrjExporter(path, meta):
			pass
		data = path.read_bytes()

		record_type, endian, version = struct.unpack("<BBf", data[:_FORMAT_SIZE])
		assert record_type == 0
		assert endian == ord("L")
		assert math.isclose(version, 1.04, abs_tol=1e-3)

	def test_dimensions_record_carries_image_bounds_and_metric_units(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=1)
		with SsamTrjExporter(path, meta):
			pass
		data = path.read_bytes()

		rec = struct.unpack("<BBfiiii", data[_FORMAT_SIZE:_HEADER_SIZE])
		record_type, units, scale, min_x, min_y, max_x, max_y = rec
		assert record_type == 1
		assert units == 1
		assert math.isclose(scale, 1.0)
		assert (min_x, min_y, max_x, max_y) == (0, 0, 1920, 1080)


class TestTimestep:
	def test_emit_frame_writes_timestep_record(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=100, height=200, fps=30.0, total_frames=1)
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(timestamp_seconds=2.5, states=[])
		data = path.read_bytes()

		record_type, ts = struct.unpack("<Bf", data[_HEADER_SIZE : _HEADER_SIZE + _TIMESTEP_SIZE])
		assert record_type == 2
		assert math.isclose(ts, 2.5)


class TestVehicleRecord:
	def test_byte_layout_for_east_facing_vehicle(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=1000, height=500, fps=10.0, total_frames=1)
		state = _vehicle(
			vehicle_id=42,
			centroid=Point2D(100.0, 200.0),
			heading=Heading(1.0, 0.0),
			length=4.0,
			width=2.0,
			velocity=Vector2D(3.0, 0.0),
			acceleration=Vector2D(0.5, 0.0),
		)
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(timestamp_seconds=0.0, states=[state])
		data = path.read_bytes()

		offset = _HEADER_SIZE + _TIMESTEP_SIZE
		rec = struct.unpack("<BiiBffffffff", data[offset : offset + _VEHICLE_SIZE])
		record_type, vid, link_id, lane_id, fx, fy, rx, ry, length, width, speed, accel = rec

		assert record_type == 3
		assert vid == 42
		assert link_id == 0
		assert lane_id == 0
		# Front bumper: image (102, 200) -> SSAM (102, 500 - 200) = (102, 300).
		assert math.isclose(fx, 102.0)
		assert math.isclose(fy, 300.0)
		# Rear bumper: image (98, 200) -> SSAM (98, 300).
		assert math.isclose(rx, 98.0)
		assert math.isclose(ry, 300.0)
		assert math.isclose(length, 4.0)
		assert math.isclose(width, 2.0)
		assert math.isclose(speed, 3.0)
		assert math.isclose(accel, 0.5)

	def test_y_axis_flipped_into_ssam_cartesian(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=1000, height=500, fps=10.0, total_frames=1)
		# Heading "up the screen" in image space (y decreasing) -> y increasing in SSAM.
		state = _vehicle(centroid=Point2D(100.0, 200.0), heading=Heading(0.0, -1.0), length=10.0)
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(timestamp_seconds=0.0, states=[state])
		data = path.read_bytes()

		offset = _HEADER_SIZE + _TIMESTEP_SIZE
		rec = struct.unpack("<BiiBffffffff", data[offset : offset + _VEHICLE_SIZE])
		_, _, _, _, fx, fy, rx, ry, *_rest = rec
		# Front bumper image = (100, 200 + (-1)*5) = (100, 195) -> SSAM (100, 500 - 195) = (100, 305).
		assert math.isclose(fx, 100.0)
		assert math.isclose(fy, 305.0)
		# Rear bumper image = (100, 200 + 5) = (100, 205) -> SSAM (100, 295).
		assert math.isclose(rx, 100.0)
		assert math.isclose(ry, 295.0)

	def test_scale_divides_coordinates_but_not_dimensions(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=400, height=400, fps=10.0, total_frames=1)
		state = _vehicle(centroid=Point2D(100.0, 100.0), heading=Heading(1.0, 0.0), length=4.0)
		with SsamTrjExporter(path, meta, scale=0.5) as exporter:
			exporter.emit_frame(timestamp_seconds=0.0, states=[state])
		data = path.read_bytes()

		# Verify Scale in DIMENSIONS record.
		_, _, scale, *_ = struct.unpack("<BBfiiii", data[_FORMAT_SIZE:_HEADER_SIZE])
		assert math.isclose(scale, 0.5)

		# Vehicle X = 102 / 0.5 = 204 grid units. Length stays 4.0 (unscaled).
		offset = _HEADER_SIZE + _TIMESTEP_SIZE
		rec = struct.unpack("<BiiBffffffff", data[offset : offset + _VEHICLE_SIZE])
		_, _, _, _, fx, _fy, _rx, _ry, length, *_rest = rec
		assert math.isclose(fx, 204.0)
		assert math.isclose(length, 4.0)

	def test_y_flip_consistent_at_non_unit_scale(self, tmp_path: Path) -> None:
		# Regression: an earlier version of the exporter computed y-flip as
		# (image_height_px - centroid_world) / scale, which only works at scale=1.
		# Verify that when state is in metres and scale is m/px, the SSAM reader
		# (file_value *Scale) recovers the right Cartesian metres-y.
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=200, height=200, fps=10.0, total_frames=1)
		# 200 px tall image *0.5 m/px = 100 m tall observation area.
		# Place a centroid at 30 m from the top of the image (image-space y).
		state = _vehicle(centroid=Point2D(50.0, 30.0), heading=Heading(1.0, 0.0), length=4.0)
		with SsamTrjExporter(path, meta, scale=0.5) as exporter:
			exporter.emit_frame(timestamp_seconds=0.0, states=[state])
		data = path.read_bytes()

		offset = _HEADER_SIZE + _TIMESTEP_SIZE
		rec = struct.unpack("<BiiBffffffff", data[offset : offset + _VEHICLE_SIZE])
		_, _, _, _, _fx, fy, _rx, _ry, *_rest = rec

		# Expected reader recovery: real_y_metres = file_y *Scale = (100 - 30) = 70 m.
		assert math.isclose(fy * 0.5, 70.0)


class TestLinkAndLaneIds:
	def test_link_and_lane_ids_written_to_vehicle_record(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=200, height=200, fps=10.0, total_frames=1)
		state = VehicleState(
			vehicle_id=42,
			timestamp_seconds=0.0,
			centroid=Point2D(100.0, 100.0),
			heading=Heading(1.0, 0.0),
			dimensions=Dimensions(length=4.0, width=2.0),
			velocity=Vector2D(0.0, 0.0),
			acceleration=Vector2D(0.0, 0.0),
			link_id=104,
			lane_id=2,
		)
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(timestamp_seconds=0.0, states=[state])
		data = path.read_bytes()

		offset = _HEADER_SIZE + _TIMESTEP_SIZE
		rec = struct.unpack("<BiiBffffffff", data[offset : offset + _VEHICLE_SIZE])
		_record_type, _vid, link_id, lane_id, *_rest = rec
		assert link_id == 104
		assert lane_id == 2


class TestFileLayout:
	def test_two_frames_three_vehicles_total_size(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=100, height=100, fps=30.0, total_frames=2)
		v1 = _vehicle(vehicle_id=1, centroid=Point2D(20.0, 30.0))
		v2 = _vehicle(vehicle_id=2, centroid=Point2D(40.0, 60.0))
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(0.0, [v1, v2])
			exporter.emit_frame(1.0 / 30.0, [v1])

		expected = _HEADER_SIZE + 2 * _TIMESTEP_SIZE + 3 * _VEHICLE_SIZE
		assert path.stat().st_size == expected

	def test_record_type_bytes_in_order(self, tmp_path: Path) -> None:
		path = tmp_path / "x.trj"
		meta = VideoMetadata(width=100, height=100, fps=10.0, total_frames=1)
		v = _vehicle()
		with SsamTrjExporter(path, meta) as exporter:
			exporter.emit_frame(0.0, [v])
		data = path.read_bytes()

		assert data[0] == 0  # FORMAT
		assert data[_FORMAT_SIZE] == 1  # DIMENSIONS
		assert data[_HEADER_SIZE] == 2  # TIMESTEP
		assert data[_HEADER_SIZE + _TIMESTEP_SIZE] == 3  # VEHICLE

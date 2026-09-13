"""Tests for the post-process pass: smooth_to_states, exclusion, and the tratrac-postprocess CLI."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tratrac.application.coordinate_transforms import (
	TransformTable,
	TranslationTransform,
	compose_invertible,
)
from tratrac.application.kalman import SmoothedSample
from tratrac.application.track_smoothing import (
	TrackSample,
	build_state,
	invert_state_to_image,
	smooth_to_states,
)
from tratrac.cli_postprocess import app
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox, Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink
from tratrac.infrastructure.transform.records import HomographyFunction, ScaleFunction, TransformRow
from tratrac.infrastructure.transform.sink import CoordinateTransformSink, whole_canvas


def _samples(n: int, *, vx: float, fps: float = 10.0) -> list[TrackSample]:
	# Eastward motion at vx px/frame-second, fixed bbox.
	return [
		TrackSample(
			frame_index=i,
			timestamp_seconds=i / fps,
			center=Point2D(vx * (i / fps), 50.0),
			width=4.0,
			height=2.0,
		)
		for i in range(n)
	]


class TestSmoothToStates:
	def test_empty_track(self) -> None:
		assert smooth_to_states(1, [], ScaleFunction(1.0), pos_noise=2.0, jerk=20.0) == []

	def test_produces_state_per_sample_with_metric_scaling(self) -> None:
		samples = _samples(30, vx=10.0)
		states = smooth_to_states(7, samples, ScaleFunction(0.5), pos_noise=1.0, jerk=10.0)
		assert len(states) == len(samples)
		assert all(s.vehicle_id == 7 for s in states)
		mid = states[15]
		# Position scaled to metric: x ~ 10 * 1.5 s * 0.5 m/px = 7.5 m.
		assert mid.centroid.x == pytest.approx(7.5, abs=0.3)
		# Heading points east (motion direction).
		assert mid.heading.dx > 0.9
		# Dimensions from bbox major/minor, scaled: length 4*0.5=2, width 2*0.5=1.
		assert mid.dimensions.length == 2.0
		assert mid.dimensions.width == 1.0

	def test_stationary_track_uses_bbox_heading(self) -> None:
		samples = [
			TrackSample(i, i / 10.0, Point2D(20.0, 20.0), width=4.0, height=2.0) for i in range(10)
		]
		states = smooth_to_states(1, samples, ScaleFunction(1.0), pos_noise=2.0, jerk=20.0)
		# No motion -> heading falls back to bbox major axis (width >= height -> east).
		assert states[-1].heading.dx == 1.0

	def test_stationary_track_prefers_obb_angle_over_bbox_heading(self) -> None:
		# OBB reports "facing north" (pi/2); bbox shape alone would say east.
		samples = [
			TrackSample(i, i / 10.0, Point2D(20.0, 20.0), width=4.0, height=2.0, angle=math.pi / 2)
			for i in range(10)
		]
		states = smooth_to_states(1, samples, ScaleFunction(1.0), pos_noise=2.0, jerk=20.0)
		assert states[-1].heading.dy == pytest.approx(1.0, abs=1e-6)

	def test_obb_angle_never_overrides_a_moving_headings_velocity(self) -> None:
		# Moving east at speed, but the OBB angle claims north — velocity wins.
		samples = [
			TrackSample(
				i,
				i / 10.0,
				Point2D(10.0 * (i / 10.0), 50.0),
				width=4.0,
				height=2.0,
				angle=math.pi / 2,
			)
			for i in range(30)
		]
		states = smooth_to_states(1, samples, ScaleFunction(1.0), pos_noise=1.0, jerk=10.0)
		assert states[15].heading.dx > 0.9

	def test_obb_angle_is_disambiguated_against_last_heading(self) -> None:
		# Stationary (speed exactly 0): an OBB angle of pi (west) is the same axis as the
		# last known heading (east) 180° flipped, and should be reversed back to east
		# rather than taken as-is — OBB reports an axis, not a disambiguated direction.
		state, remembered = build_state(
			track_id=1,
			timestamp_seconds=0.0,
			kinematics=SmoothedSample(px=0.0, py=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0),
			width=4.0,
			height=2.0,
			scale=ScaleFunction(1.0),
			last_heading=Heading(1.0, 0.0),
			angle=math.pi,
		)
		assert state.heading.dx == pytest.approx(1.0)
		assert remembered == Heading(1.0, 0.0)  # low-speed branch never updates "last good"

	def test_oriented_size_replaces_bbox_for_dimensions(self) -> None:
		samples = [
			TrackSample(
				i, i / 10.0, Point2D(20.0, 20.0), width=4.0, height=2.0, oriented_size=(9.0, 3.0)
			)
			for i in range(5)
		]
		states = smooth_to_states(1, samples, ScaleFunction(2.0), pos_noise=2.0, jerk=20.0)
		# Scaled by 2.0 m/px: length 9*2=18, width 3*2=6 — not the bbox's 4*2=8 / 2*2=4.
		assert states[-1].dimensions.length == pytest.approx(18.0)
		assert states[-1].dimensions.width == pytest.approx(6.0)


def _homography_table(matrix: np.ndarray, *, frame_index: int = 0) -> TransformTable:
	return TransformTable(
		[
			TransformRow(
				frame_index,
				whole_canvas(1000, 1000),
				HomographyFunction(tuple(float(v) for v in matrix.flatten())),
			)
		]
	)


def _scale_homography(s: float) -> np.ndarray:
	return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class TestInvertStateToImage:
	def test_undoes_a_pure_scale_projection(self) -> None:
		# world = 0.5 * image -> image = 2 * world, the inverse this test checks.
		projector = _homography_table(_scale_homography(0.5))
		world_state = VehicleState(
			vehicle_id=1,
			timestamp_seconds=0.0,
			centroid=Point2D(10.0, 20.0),
			heading=Heading(1.0, 0.0),
			dimensions=Dimensions(length=4.0, width=2.0),
			velocity=Vector2D(0.0, 0.0),
			acceleration=0.0,
		)
		centroid, angle, dimensions = invert_state_to_image(world_state, projector, frame_index=0)
		assert centroid.x == pytest.approx(20.0)
		assert centroid.y == pytest.approx(40.0)
		assert angle == pytest.approx(0.0)  # heading (1, 0) unchanged direction under pure scale
		assert dimensions.length == pytest.approx(8.0)
		assert dimensions.width == pytest.approx(4.0)

	def test_round_trips_through_shifted_and_general_homography(self) -> None:
		# A real (non-axis-aligned-preserving) homography plus the 0-origin shift
		# _project_to_world composes in — the combination cli_postprocess.py actually uses.
		matrix = np.array([[2.0, 0.3, 5.0], [0.1, 1.5, -2.0], [0.001, 0.0005, 1.0]])
		projector = compose_invertible(
			_homography_table(matrix, frame_index=5), TranslationTransform(37.0, -11.0)
		)

		pixel_centroid = Point2D(50.0, 80.0)
		pixel_heading = Heading.from_angle(0.7)
		pixel_dimensions = Dimensions(length=6.0, width=3.0)
		front = pixel_centroid.translate_by(pixel_heading.as_vector_with_magnitude(3.0))
		rear = pixel_centroid.translate_by(pixel_heading.reversed().as_vector_with_magnitude(3.0))

		world_front = projector.apply(front, frame_index=5)
		world_rear = projector.apply(rear, frame_index=5)
		world_centroid = Point2D(
			(world_front.x + world_rear.x) / 2.0, (world_front.y + world_rear.y) / 2.0
		)
		world_axis = world_rear.displacement_to(world_front)
		world_state = VehicleState(
			vehicle_id=2,
			timestamp_seconds=0.0,
			centroid=world_centroid,
			heading=world_axis.normalized(),
			dimensions=Dimensions(length=world_axis.magnitude, width=pixel_dimensions.width),
			velocity=Vector2D(0.0, 0.0),
			acceleration=0.0,
		)
		centroid, angle, dimensions = invert_state_to_image(world_state, projector, frame_index=5)
		assert centroid.x == pytest.approx(pixel_centroid.x, abs=1e-4)
		assert centroid.y == pytest.approx(pixel_centroid.y, abs=1e-4)
		assert angle == pytest.approx(0.7, abs=1e-4)
		assert dimensions.length == pytest.approx(pixel_dimensions.length, abs=1e-3)


def _write_transforms_with_scale(
	path: Path, *, width: int, height: int, n_frames: int, scale: float
) -> None:
	"""A minimal ``tratrac-preprocess estimate``-shaped transforms file: no ego-motion
	rows, one scale row per frame -- everything ``tratrac-postprocess --transforms``
	needs for an image-space (non-projected) run."""
	zone = whole_canvas(width, height)
	with CoordinateTransformSink(path, width=width, height=height) as sink:
		for frame_index in range(n_frames):
			sink.record_row(TransformRow(frame_index, zone, ScaleFunction(scale)))


def _write_tracks(path: Path, samples: list[TrackSample]) -> Path:
	"""Returns the transforms file's path (for ``--transforms``)."""
	meta = VideoMetadata(width=1920, height=1080, fps=10.0, total_frames=len(samples))
	with ParquetTrackSink(path, meta) as sink:
		for s in samples:
			det = TrackedDetection(
				track_id=1,
				detection=Detection(
					bbox=BoundingBox(
						x=s.center.x - s.width / 2,
						y=s.center.y - s.height / 2,
						width=s.width,
						height=s.height,
					),
					score=0.9,
					vehicle_class=VehicleClass.CAR,
				),
			)
			sink.record(s.frame_index, [det])
	transforms_path = path.with_name(path.name + ".transforms.jsonl")
	_write_transforms_with_scale(
		transforms_path, width=meta.width, height=meta.height, n_frames=len(samples), scale=1.0
	)
	return transforms_path


class TestSmoothCli:
	def test_smooths_tracks_into_parseable_trj(self, tmp_path: Path) -> None:
		tracks = tmp_path / "tracks.parquet"
		out = tmp_path / "smooth.trj"
		transforms_path = _write_tracks(tracks, _samples(20, vx=8.0))

		result = CliRunner().invoke(
			app, [str(tracks), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output
		assert out.exists()
		# FORMAT record: first byte is the record type; the file is non-empty binary.
		data = out.read_bytes()
		assert len(data) > 0
		(record_type,) = struct.unpack_from("<B", data, 0)
		assert record_type in (0, 1, 2, 3)  # a valid SSAM record-type tag

	def test_refuses_to_overwrite_without_force(self, tmp_path: Path) -> None:
		tracks = tmp_path / "tracks.parquet"
		out = tmp_path / "smooth.trj"
		transforms_path = _write_tracks(tracks, _samples(5, vx=8.0))
		out.write_text("existing")
		result = CliRunner().invoke(
			app, [str(tracks), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code != 0
		assert "force" in result.output.lower()


def _track(track_id: int, cx: float, cy: float) -> TrackedDetection:
	# bbox centred on (cx, cy).
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=cx - 2.0, y=cy - 2.0, width=4.0, height=4.0),
			score=0.9,
			vehicle_class=VehicleClass.CAR,
		),
	)


class TestExclusion:
	def test_drops_a_track_mostly_inside_a_zone(self, tmp_path: Path) -> None:
		record = tmp_path / "record.parquet"
		meta = VideoMetadata(width=400, height=400, fps=10.0, total_frames=5)
		with ParquetTrackSink(record, meta) as sink:
			for i in range(5):
				# track 1 sits inside the zone [0,50]^2; track 2 is far outside.
				sink.record(i, [_track(1, 10.0, 10.0), _track(2, 300.0, 300.0)])
		transforms_path = record.with_name(record.name + ".transforms.jsonl")
		_write_transforms_with_scale(
			transforms_path, width=meta.width, height=meta.height, n_frames=5, scale=1.0
		)

		zones = tmp_path / "zones.json"
		zones.write_text(
			json.dumps(
				{
					"exclusion_zones": [
						{"reference_frame": 0, "vertices": [[0, 0], [50, 0], [50, 50], [0, 50]]}
					]
				}
			)
		)
		out = tmp_path / "out.trj"
		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				"--exclusion-zones",
				str(zones),
			],
		)
		assert result.exit_code == 0, result.output
		assert "dropped 1 excluded tracks" in result.output
		assert out.exists()

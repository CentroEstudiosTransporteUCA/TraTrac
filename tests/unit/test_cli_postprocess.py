"""Integration tests for ``tratrac-postprocess``, focused on the MVP2 world-projection path.

Builds a small Parquet track record plus a transforms file (mimicking what
``tratrac-preprocess estimate``/``project`` would have written), runs the CLI, and reads the
emitted ``.trj`` back to assert that a homography-fitted transforms file projects to world
metres + ``DIMENSIONS.Scale = 1.0``, and that a scale-only one leaves the pre-MVP2 image-space
path unchanged. See src/tratrac/application/WORLD_PROJECTION.md.

The homography fixtures reuse ``cli_preprocess.py``'s own fitting helpers (``_fit_whole_scene``/
``_fit_per_plane``) rather than re-deriving the math by hand, so these tests exercise the real
production fitting code and stay focused on what ``tratrac-postprocess`` itself does with an
already-fitted transforms file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tratrac.application.coordinate_transforms import TransformTable
from tratrac.cli_postprocess import app
from tratrac.cli_preprocess import _fit_per_plane, _fit_whole_scene
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox, Point2D, Polygon
from tratrac.domain.world import Calibration, Correspondence
from tratrac.infrastructure.export.ssam_trj import read_trj
from tratrac.infrastructure.tracks.footprint_parquet import FootprintParquetSink
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink
from tratrac.infrastructure.tracks.smoothed_parquet import read_smoothed_tracks
from tratrac.infrastructure.transform.records import ScaleFunction, TransformRow
from tratrac.infrastructure.transform.sink import CoordinateTransformSink, whole_canvas

_META = VideoMetadata(width=200, height=200, fps=10.0, total_frames=10)

# A pure-scaling homography: world = image * 0.5 (0.5 metres per pixel), static camera.
_HALF_SCALE_CALIBRATION = {
	"correspondences": [
		{"reference_frame": 0, "image": [0, 0], "world": [0.0, 0.0]},
		{"reference_frame": 0, "image": [100, 0], "world": [50.0, 0.0]},
		{"reference_frame": 0, "image": [100, 100], "world": [50.0, 50.0]},
		{"reference_frame": 0, "image": [0, 100], "world": [0.0, 50.0]},
	]
}


def _tracked(x: float, y: float, *, track_id: int = 1) -> TrackedDetection:
	# bbox top-left (x, y), size 4x2 -> centre (x + 2, y + 1).
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=x, y=y, width=4.0, height=2.0),
			score=0.9,
			vehicle_class=VehicleClass.CAR,
		),
	)


def _write_scale_transforms(
	path: Path, *, width: int, height: int, n_frames: int, scale: float
) -> None:
	"""A minimal ``tratrac-preprocess estimate``-shaped file: no ego-motion rows, one scale
	row per frame."""
	zone = whole_canvas(width, height)
	with CoordinateTransformSink(path, width=width, height=height) as sink:
		for frame_index in range(n_frames):
			sink.record_row(TransformRow(frame_index, zone, ScaleFunction(scale)))


def _calibration_from_dict(data: dict[str, Any]) -> Calibration:
	return Calibration(
		correspondences=tuple(
			Correspondence(
				reference_frame=c.get("reference_frame", 0),
				image=Point2D(*c["image"]),
				world=Point2D(*c["world"]),
			)
			for c in data["correspondences"]
		)
	)


def _write_homography_transforms(
	path: Path,
	*,
	width: int,
	height: int,
	n_frames: int,
	calibration: dict[str, Any],
	plane_zones: Path | None = None,
) -> None:
	"""A ``tratrac-preprocess project``-shaped file: fits (via the real fitting helpers) and
	materializes homography rows for frames ``0..n_frames-1`` -- a static camera, so the
	ego-motion stage used to map correspondences is the identity (an empty table)."""
	frame_indices = list(range(n_frames))
	pose = TransformTable([])
	calibration_data = _calibration_from_dict(calibration)
	rows = (
		_fit_per_plane(calibration_data, pose, plane_zones, frame_indices)
		if plane_zones is not None
		else _fit_whole_scene(calibration_data, pose, frame_indices, width, height)
	)
	with CoordinateTransformSink(path, width=width, height=height) as sink:
		for row in rows:
			sink.record_row(row)


def _write_record(path: Path, *, scale: float) -> Path:
	"""A single track moving at constant velocity along x (so the CA smoother reproduces it).

	Returns the transforms file's path (for ``--transforms``)."""
	with ParquetTrackSink(path, _META) as sink:
		for frame in range(6):
			sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0)])
	transforms_path = path.with_name(path.name + ".transforms.jsonl")
	_write_scale_transforms(
		transforms_path, width=_META.width, height=_META.height, n_frames=6, scale=scale
	)
	return transforms_path


def _centroid_at(trj_path: Path, frame_index: int) -> tuple[float, float]:
	"""The (sole) track's centroid at an interior frame, away from filter transients."""
	state = read_trj(trj_path).frames[frame_index].states[0]
	return state.centroid.x, state.centroid.y


def _centroid_for(trj_path: Path, frame_index: int, vehicle_id: int) -> tuple[float, float]:
	"""One of several tracks' centroid at an interior frame, by vehicle id."""
	for state in read_trj(trj_path).frames[frame_index].states:
		if state.vehicle_id == vehicle_id:
			return state.centroid.x, state.centroid.y
	raise AssertionError(f"vehicle {vehicle_id} not found at frame {frame_index}")


class TestPostprocessWorldProjection:
	def test_scale_only_transforms_keep_pixel_scale(self, tmp_path: Path) -> None:
		"""A scale-only table goes through the same per-observation projection any other
		table does (see `cli_postprocess.py`'s module docstring) -- with `scale=1.0` it
		maps pixels to identical pixels, so the *displacement* between frames is still
		exactly the raw per-frame pixel motion. Absolute position isn't asserted: like any
		other projected run, the recording is shifted to a 0-origin extent, so it's no
		longer literally the source video's own pixel coordinates."""
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "image.trj"
		transforms_path = _write_record(record, scale=1.0)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		assert read_trj(out).scale == pytest.approx(1.0)
		(x2, y2), (x3, y3) = _centroid_at(out, 2), _centroid_at(out, 3)
		# 10 px/frame at scale=1.0 -> 10 units/frame; y is constant.
		assert x3 - x2 == pytest.approx(10.0, abs=0.2)
		assert y3 - y2 == pytest.approx(0.0, abs=0.2)

	def test_homography_projects_to_world_and_sets_unit_scale(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0)])
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration=_HALF_SCALE_CALIBRATION,
		)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output
		assert "projected to world coordinates" in result.output

		# DIMENSIONS.Scale becomes 1.0 — the coordinates are already metric.
		assert read_trj(out).scale == pytest.approx(1.0)
		# Coordinates are translated to a 0-origin (absolute world origin is discarded), so
		# we assert the metric *displacement*: image dx = 10 px/frame, * 0.5 -> 5 m/frame.
		(x2, _), (x3, _) = _centroid_at(out, 2), _centroid_at(out, 3)
		assert x3 - x2 == pytest.approx(5.0, abs=0.2)

	def test_homography_sizes_dimensions_to_world_extent(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		with ParquetTrackSink(record, _META) as sink:
			for frame_index in range(6):
				sink.record(frame_index, [_tracked(x=10.0 * frame_index + 20.0, y=50.0)])
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration=_HALF_SCALE_CALIBRATION,
		)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		recording = read_trj(out)
		# DIMENSIONS bounds describe the WORLD extent (metres, ~29x4), not the 200x200 pixel
		# grid — otherwise an external reader sees metric coords on a pixel-sized canvas.
		assert recording.width < 100
		assert recording.height < 100
		# Every stored coordinate is non-negative and inside those bounds (the Y-flip is now
		# about the world height, not the pixel height).
		for trj_frame in recording.frames:
			for state in trj_frame.states:
				assert 0.0 <= state.centroid.x <= recording.width
				assert 0.0 <= state.centroid.y <= recording.height

	def test_multi_anchor_calibration_fits_one_homography_per_anchor(self, tmp_path: Path) -> None:
		# Two anchors, half-scale near frame 0 and quarter-scale near frame 9 -- a static
		# camera, so reference_frame is purely a fit-time grouping label here.
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "multi.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		meta = VideoMetadata(width=300, height=300, fps=10.0, total_frames=10)
		with ParquetTrackSink(record, meta) as sink:
			for frame in range(10):
				sink.record(frame, [_tracked(x=10.0 * frame, y=50.0)])
		_write_homography_transforms(
			transforms_path,
			width=meta.width,
			height=meta.height,
			n_frames=10,
			calibration={
				"correspondences": [
					{"reference_frame": 0, "image": [0, 0], "world": [0.0, 0.0]},
					{"reference_frame": 0, "image": [100, 0], "world": [50.0, 0.0]},
					{"reference_frame": 0, "image": [100, 100], "world": [50.0, 50.0]},
					{"reference_frame": 0, "image": [0, 100], "world": [0.0, 50.0]},
					{"reference_frame": 9, "image": [0, 0], "world": [0.0, 0.0]},
					{"reference_frame": 9, "image": [100, 0], "world": [25.0, 0.0]},
					{"reference_frame": 9, "image": [100, 100], "world": [25.0, 25.0]},
					{"reference_frame": 9, "image": [0, 100], "world": [0.0, 25.0]},
				]
			},
		)

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				# Near-zero measurement noise + a very responsive process model: the smoother
				# tracks the projected measurements closely instead of blending across the
				# mid-track scale switch, so the per-anchor displacement stays measurable.
				"--pos-noise",
				"0.01",
				"--jerk",
				"1e6",
			],
		)
		assert result.exit_code == 0, result.output

		near_anchor_0 = _centroid_at(out, 1)[0] - _centroid_at(out, 0)[0]
		near_anchor_9 = _centroid_at(out, 9)[0] - _centroid_at(out, 8)[0]
		# 10 px/frame at 0.5 m/px near anchor 0, at 0.25 m/px near anchor 9.
		assert near_anchor_0 == pytest.approx(5.0, abs=0.5)
		assert near_anchor_9 == pytest.approx(2.5, abs=0.5)

	def test_plane_zones_fit_one_homography_per_plane(self, tmp_path: Path) -> None:
		# Track 1 stays on the "ground" plane (x in [0,100]); track 2 stays on the "bridge"
		# plane (x in [200,300]) — a stand-in for a grade-separated scene (MVP3). Each plane
		# gets its own scale: ground half-scale, bridge quarter-scale.
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "planes.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		planes = tmp_path / "planes.json"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(
					frame,
					[
						_tracked(x=10.0 * frame, y=50.0, track_id=1),
						_tracked(x=210.0 + 10.0 * frame, y=250.0, track_id=2),
					],
				)
		# Zones are padded a few pixels beyond the correspondence square so the corner
		# correspondences (exactly on [0,100]/[200,300]) fall unambiguously inside, not
		# exactly on the boundary (shapely's `contains` is boundary-exclusive by design).
		planes.write_text(
			json.dumps(
				{
					"plane_zones": [
						{
							"plane_id": 0,
							"vertices": [[-10, -10], [110, -10], [110, 110], [-10, 110]],
						},
						{
							"plane_id": 1,
							"vertices": [[190, 190], [310, 190], [310, 310], [190, 310]],
						},
					]
				}
			)
		)
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration={
				"correspondences": [
					{"image": [0, 0], "world": [0.0, 0.0]},
					{"image": [100, 0], "world": [50.0, 0.0]},
					{"image": [100, 100], "world": [50.0, 50.0]},
					{"image": [0, 100], "world": [0.0, 50.0]},
					{"image": [200, 200], "world": [50.0, 50.0]},
					{"image": [300, 200], "world": [75.0, 50.0]},
					{"image": [300, 300], "world": [75.0, 75.0]},
					{"image": [200, 300], "world": [50.0, 75.0]},
				]
			},
			plane_zones=planes,
		)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		ground_dx = _centroid_for(out, 3, 1)[0] - _centroid_for(out, 2, 1)[0]
		bridge_dx = _centroid_for(out, 3, 2)[0] - _centroid_for(out, 2, 2)[0]
		# 10 px/frame at 0.5 m/px on the ground plane, at 0.25 m/px on the bridge plane.
		assert ground_dx == pytest.approx(5.0, abs=0.5)
		assert bridge_dx == pytest.approx(2.5, abs=0.5)

	def test_multiple_scale_zones_apply_their_own_factor(self, tmp_path: Path) -> None:
		"""Two SCALE zones (not homography) with different factors -- e.g. a fisheye lens'
		radially-varying GSD, simplified to a left/right split. Regression test for the bug
		where `postprocess` picked one zone's factor via `any_function()` and applied it to
		every observation regardless of which zone it was actually in."""
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "multi_scale.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(
					frame,
					[
						_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1),  # left half
						_tracked(x=10.0 * frame + 120.0, y=50.0, track_id=2),  # right half
					],
				)
		left = (
			Point2D(0.0, 0.0),
			Point2D(100.0, 0.0),
			Point2D(100.0, float(_META.height)),
			Point2D(0.0, float(_META.height)),
		)
		right = (
			Point2D(100.0, 0.0),
			Point2D(float(_META.width), 0.0),
			Point2D(float(_META.width), float(_META.height)),
			Point2D(100.0, float(_META.height)),
		)
		with CoordinateTransformSink(
			transforms_path, width=_META.width, height=_META.height
		) as sink:
			for frame_index in range(6):
				sink.record_row(TransformRow(frame_index, left, ScaleFunction(0.5)))
				sink.record_row(TransformRow(frame_index, right, ScaleFunction(0.25)))

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		left_dx = _centroid_for(out, 3, 1)[0] - _centroid_for(out, 2, 1)[0]
		right_dx = _centroid_for(out, 3, 2)[0] - _centroid_for(out, 2, 2)[0]
		# 10 px/frame at 0.5 m/px in the left zone, at 0.25 m/px in the right zone -- if the
		# old any_function() bug picked one zone's factor for both, these would come out equal.
		assert left_dx == pytest.approx(5.0, abs=0.5)
		assert right_dx == pytest.approx(2.5, abs=0.5)

	def test_homography_scales_metric_dimensions(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		transforms_path = tmp_path / "transforms.jsonl"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0)])
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration=_HALF_SCALE_CALIBRATION,
		)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		# bbox 4x2 px projected through world = image * 0.5 -> 2.0 x 1.0 metres.
		state = read_trj(out).frames[3].states[0]
		assert state.dimensions.length == pytest.approx(2.0, abs=0.05)
		assert state.dimensions.width == pytest.approx(1.0, abs=0.05)


class TestPostprocessTransformsRequirements:
	def test_missing_transforms_flag_is_an_error(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "out.trj"
		_write_record(record, scale=1.0)

		result = CliRunner().invoke(app, [str(record), "--out", str(out)])
		assert result.exit_code != 0
		assert "transforms" in result.output

	def test_transforms_file_with_no_scale_or_homography_rows_is_an_error(
		self, tmp_path: Path
	) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "out.trj"
		_write_record(record, scale=1.0)
		# An ego-motion-only file (e.g. before `project` ever ran) has no projection-stage
		# rows at all -- there's nothing to scale or project with.
		empty_transforms = tmp_path / "no_projection.jsonl"
		with CoordinateTransformSink(empty_transforms, width=200, height=200):
			pass

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(empty_transforms), "--out", str(out)]
		)
		assert result.exit_code != 0
		assert "no scale or homography rows" in result.output


class TestPostprocessSmoothedRecord:
	"""``--smoothed-record``: always image-space, independent of world projection."""

	def test_requires_out_or_smoothed_record(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		transforms_path = _write_record(record, scale=1.0)

		# COLUMNS forces Rich's error panel to render unwrapped -- CliRunner's default width
		# detection in a non-interactive CI environment can otherwise hard-wrap this message
		# across lines, breaking a plain substring check.
		result = CliRunner().invoke(
			app,
			[str(record), "--transforms", str(transforms_path)],
			env={"COLUMNS": "200"},
		)
		assert result.exit_code != 0
		assert "smoothed-record" in result.output

	def test_smoothed_record_alone_needs_no_out(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		smoothed = tmp_path / "smoothed.parquet"
		transforms_path = _write_record(record, scale=1.0)

		result = CliRunner().invoke(
			app,
			[str(record), "--transforms", str(transforms_path), "--smoothed-record", str(smoothed)],
		)
		assert result.exit_code == 0, result.output
		assert smoothed.exists()

	def test_smoothed_record_without_homography_matches_image_positions(
		self, tmp_path: Path
	) -> None:
		record = tmp_path / "tracks.parquet"
		smoothed = tmp_path / "smoothed.parquet"
		transforms_path = _write_record(record, scale=1.0)

		result = CliRunner().invoke(
			app,
			[str(record), "--transforms", str(transforms_path), "--smoothed-record", str(smoothed)],
		)
		assert result.exit_code == 0, result.output

		recovered = read_smoothed_tracks(smoothed)
		state = next(o for o in recovered.observations if o.frame_index == 3)
		# frame 3 image centre: (10*3 + 20 + 2, 51) = (52, 51) -- inverting the forward
		# scale=1.0 projection (+ its 0-origin shift) recovers the exact original pixel
		# position, the same way a homography's inverse does below.
		assert state.cx == pytest.approx(52.0, abs=0.5)
		assert state.cy == pytest.approx(51.0, abs=0.5)

	def test_smoothed_record_with_homography_still_recovers_image_positions(
		self, tmp_path: Path
	) -> None:
		"""The real point of --smoothed-record: a homography projects .trj to world metres,
		but this output stays in the same raw pixels regardless -- inverting the homography
		(+ the 0-origin shift _project_to_world applies) recovers the original image position."""
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		smoothed = tmp_path / "smoothed.parquet"
		transforms_path = tmp_path / "transforms.jsonl"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0)])
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration=_HALF_SCALE_CALIBRATION,
		)

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				"--smoothed-record",
				str(smoothed),
			],
		)
		assert result.exit_code == 0, result.output

		# .trj is world-space (the existing, unchanged behavior)...
		assert read_trj(out).scale == pytest.approx(1.0)
		# ...but --smoothed-record recovers the same raw image position the unprojected
		# test above got, not the world-metric one.
		recovered = read_smoothed_tracks(smoothed)
		state = next(o for o in recovered.observations if o.frame_index == 3)
		assert state.cx == pytest.approx(52.0, abs=0.5)
		assert state.cy == pytest.approx(51.0, abs=0.5)

	def test_smoothed_record_with_a_noninvertible_multi_plane_homography_is_rejected(
		self, tmp_path: Path
	) -> None:
		record = tmp_path / "tracks.parquet"
		smoothed = tmp_path / "smoothed.parquet"
		transforms_path = tmp_path / "transforms.jsonl"
		planes = tmp_path / "planes.json"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(6):
				sink.record(
					frame,
					[
						_tracked(x=10.0 * frame, y=50.0, track_id=1),
						_tracked(x=210.0 + 10.0 * frame, y=250.0, track_id=2),
					],
				)
		planes.write_text(
			json.dumps(
				{
					"plane_zones": [
						{
							"plane_id": 0,
							"vertices": [[-10, -10], [110, -10], [110, 110], [-10, 110]],
						},
						{
							"plane_id": 1,
							"vertices": [[190, 190], [310, 190], [310, 310], [190, 310]],
						},
					]
				}
			)
		)
		_write_homography_transforms(
			transforms_path,
			width=_META.width,
			height=_META.height,
			n_frames=6,
			calibration={
				"correspondences": [
					{"image": [0, 0], "world": [0.0, 0.0]},
					{"image": [100, 0], "world": [50.0, 0.0]},
					{"image": [100, 100], "world": [50.0, 50.0]},
					{"image": [0, 100], "world": [0.0, 50.0]},
					{"image": [200, 200], "world": [50.0, 50.0]},
					{"image": [300, 200], "world": [75.0, 50.0]},
					{"image": [300, 300], "world": [75.0, 75.0]},
					{"image": [200, 300], "world": [50.0, 75.0]},
				]
			},
			plane_zones=planes,
		)

		# COLUMNS forces Rich's error panel to render unwrapped -- see the comment on
		# test_requires_out_or_smoothed_record above.
		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--smoothed-record",
				str(smoothed),
			],
			env={"COLUMNS": "200"},
		)
		assert result.exit_code != 0
		assert "smoothed-record" in result.output


class TestPostprocessReidMerge:
	def test_reid_merge_stitches_two_fragments_into_one_track(self, tmp_path: Path) -> None:
		# track 1 (frames 0-2) and track 2 (frames 5-7) are the same vehicle, split by an
		# occlusion (frames 3-4 missing) -- a fragment pair a real ReID merge decision would
		# resolve to {"2": 1}.
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "merged.trj"
		merge = tmp_path / "merge.json"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(3):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
			for frame in range(5, 8):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=2)])
		transforms_path = record.with_name(record.name + ".transforms.jsonl")
		_write_scale_transforms(
			transforms_path, width=_META.width, height=_META.height, n_frames=8, scale=1.0
		)
		merge.write_text(json.dumps({"merges": {"2": 1}}))

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				"--reid-merge",
				str(merge),
			],
		)
		assert result.exit_code == 0, result.output
		assert "merged 1 ReID track ids" in result.output

		vehicle_ids = {state.vehicle_id for frame in read_trj(out).frames for state in frame.states}
		assert vehicle_ids == {1}

	def test_without_reid_merge_fragments_stay_separate(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "unmerged.trj"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(3):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
			for frame in range(5, 8):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=2)])
		transforms_path = record.with_name(record.name + ".transforms.jsonl")
		_write_scale_transforms(
			transforms_path, width=_META.width, height=_META.height, n_frames=8, scale=1.0
		)

		result = CliRunner().invoke(
			app, [str(record), "--transforms", str(transforms_path), "--out", str(out)]
		)
		assert result.exit_code == 0, result.output

		vehicle_ids = {state.vehicle_id for frame in read_trj(out).frames for state in frame.states}
		assert vehicle_ids == {1, 2}


class TestPostprocessFootprint:
	def test_footprint_replaces_bbox_derived_dimensions(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "footprint.trj"
		footprint = tmp_path / "footprint.parquet"
		with ParquetTrackSink(record, _META) as sink:
			for frame in range(5):
				# bbox 4x2 -> dims would be 4.0 x 2.0 without a footprint.
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
		transforms_path = record.with_name(record.name + ".transforms.jsonl")
		_write_scale_transforms(
			transforms_path, width=_META.width, height=_META.height, n_frames=5, scale=1.0
		)
		# An axis-aligned footprint polygon spanning 8x4 -> larger than the bbox.
		with FootprintParquetSink(footprint) as sink:
			for frame in range(5):
				cx, cy = 10.0 * frame + 22.0, 51.0
				polygon = Polygon(
					(
						Point2D(cx - 4.0, cy - 2.0),
						Point2D(cx + 4.0, cy - 2.0),
						Point2D(cx + 4.0, cy + 2.0),
						Point2D(cx - 4.0, cy + 2.0),
					)
				)
				sink.record(frame, 1, polygon)

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				"--footprint",
				str(footprint),
			],
		)
		assert result.exit_code == 0, result.output
		assert "replaced dimensions for 5 observations from footprints" in result.output

		state = read_trj(out).frames[2].states[0]
		assert state.dimensions.length == pytest.approx(8.0, abs=0.1)
		assert state.dimensions.width == pytest.approx(4.0, abs=0.1)

	def test_uncovered_observations_keep_bbox_dimensions(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "footprint.trj"
		footprint = tmp_path / "footprint.parquet"
		transforms_path = _write_record(record, scale=1.0)
		with FootprintParquetSink(footprint):
			pass  # no footprint rows at all -> nothing covered

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--transforms",
				str(transforms_path),
				"--out",
				str(out),
				"--footprint",
				str(footprint),
			],
		)
		assert result.exit_code == 0, result.output

		state = read_trj(out).frames[3].states[0]
		# bbox 4x2 px, scale 1.0 -> unchanged dims (same as the no-footprint baseline).
		assert state.dimensions.length == pytest.approx(4.0, abs=0.1)
		assert state.dimensions.width == pytest.approx(2.0, abs=0.1)

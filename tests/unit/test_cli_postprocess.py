"""Integration tests for ``tratrac-postprocess``, focused on the MVP2 ``--calibration`` path.

Builds a small Parquet track record, runs the CLI, and reads the emitted ``.trj`` back to
assert that with a calibration the coordinates are world metres + ``DIMENSIONS.Scale = 1.0``,
and that without one the pre-MVP2 image-space path is unchanged. See src/tratrac/application/WORLD_PROJECTION.md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tratrac.cli_postprocess import app
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox, Point2D, Polygon
from tratrac.infrastructure.export.ssam_trj import read_trj
from tratrac.infrastructure.tracks.footprint_parquet import FootprintParquetSink
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink

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


def _write_record(path: Path, *, scale: float) -> None:
	"""A single track moving at constant velocity along x (so the CA smoother reproduces it)."""
	with ParquetTrackSink(path, _META, scale=scale) as sink:
		for frame in range(6):
			sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0)])


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


class TestPostprocessCalibration:
	def test_without_calibration_coordinates_stay_image_space(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "image.trj"
		_write_record(record, scale=1.0)

		result = CliRunner().invoke(app, [str(record), "--out", str(out)])
		assert result.exit_code == 0, result.output

		assert read_trj(out).scale == pytest.approx(1.0)
		cx, cy = _centroid_at(out, 3)
		# frame 3 image centre: (10*3 + 20 + 2, 51) = (52, 51)
		assert cx == pytest.approx(52.0, abs=0.5)
		assert cy == pytest.approx(51.0, abs=0.5)

	def test_with_calibration_projects_to_world_and_sets_unit_scale(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		calibration = tmp_path / "calibration.json"
		_write_record(record, scale=1.0)
		calibration.write_text(json.dumps(_HALF_SCALE_CALIBRATION))

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--calibration", str(calibration)]
		)
		assert result.exit_code == 0, result.output
		assert "projected to world coordinates" in result.output

		# DIMENSIONS.Scale becomes 1.0 — the coordinates are already metric.
		assert read_trj(out).scale == pytest.approx(1.0)
		# Coordinates are translated to a 0-origin (absolute world origin is discarded), so
		# we assert the metric *displacement*: image dx = 10 px/frame, * 0.5 -> 5 m/frame.
		(x2, _), (x3, _) = _centroid_at(out, 2), _centroid_at(out, 3)
		assert x3 - x2 == pytest.approx(5.0, abs=0.2)

	def test_with_calibration_sizes_dimensions_to_world_extent(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		calibration = tmp_path / "calibration.json"
		_write_record(record, scale=1.0)
		calibration.write_text(json.dumps(_HALF_SCALE_CALIBRATION))

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--calibration", str(calibration)]
		)
		assert result.exit_code == 0, result.output

		recording = read_trj(out)
		# DIMENSIONS bounds describe the WORLD extent (metres, ~29x4), not the 200x200 pixel
		# grid — otherwise an external reader sees metric coords on a pixel-sized canvas.
		assert recording.width < 100
		assert recording.height < 100
		# Every stored coordinate is non-negative and inside those bounds (the Y-flip is now
		# about the world height, not the pixel height).
		for frame in recording.frames:
			for state in frame.states:
				assert 0.0 <= state.centroid.x <= recording.width
				assert 0.0 <= state.centroid.y <= recording.height

	def test_multi_anchor_calibration_fits_one_homography_per_anchor(self, tmp_path: Path) -> None:
		# Two anchors, half-scale near frame 0 and quarter-scale near frame 9 (no --anchors
		# manifest -> every reference_frame's pose is identity, matching the single-anchor
		# tests above; reference_frame is purely a grouping label here).
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "multi.trj"
		calibration = tmp_path / "calibration.json"
		with ParquetTrackSink(
			record, VideoMetadata(width=300, height=300, fps=10.0, total_frames=10), scale=1.0
		) as sink:
			for frame in range(10):
				sink.record(frame, [_tracked(x=10.0 * frame, y=50.0)])
		calibration.write_text(
			json.dumps(
				{
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
				}
			)
		)

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--out",
				str(out),
				"--calibration",
				str(calibration),
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
		calibration = tmp_path / "calibration.json"
		planes = tmp_path / "planes.json"
		with ParquetTrackSink(record, _META, scale=1.0) as sink:
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
		calibration.write_text(
			json.dumps(
				{
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
				}
			)
		)

		result = CliRunner().invoke(
			app,
			[
				str(record),
				"--out",
				str(out),
				"--calibration",
				str(calibration),
				"--plane-zones",
				str(planes),
			],
		)
		assert result.exit_code == 0, result.output

		ground_dx = _centroid_for(out, 3, 1)[0] - _centroid_for(out, 2, 1)[0]
		bridge_dx = _centroid_for(out, 3, 2)[0] - _centroid_for(out, 2, 2)[0]
		# 10 px/frame at 0.5 m/px on the ground plane, at 0.25 m/px on the bridge plane.
		assert ground_dx == pytest.approx(5.0, abs=0.5)
		assert bridge_dx == pytest.approx(2.5, abs=0.5)

	def test_plane_zones_without_calibration_is_an_error(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "out.trj"
		planes = tmp_path / "planes.json"
		_write_record(record, scale=1.0)
		planes.write_text(
			json.dumps({"plane_zones": [{"plane_id": 0, "vertices": [[0, 0], [1, 0], [0, 1]]}]})
		)

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--plane-zones", str(planes)]
		)
		assert result.exit_code != 0
		# rich highlights each "--flag" with interleaved ANSI codes, splitting a plain
		# "--calibration" substring, hence the dash-less match (mirrors the --force test above).
		assert "calibration" in result.output

	def test_calibration_scales_metric_dimensions(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "world.trj"
		calibration = tmp_path / "calibration.json"
		_write_record(record, scale=1.0)
		calibration.write_text(json.dumps(_HALF_SCALE_CALIBRATION))

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--calibration", str(calibration)]
		)
		assert result.exit_code == 0, result.output

		# bbox 4x2 px projected through world = image * 0.5 -> 2.0 x 1.0 metres.
		state = read_trj(out).frames[3].states[0]
		assert state.dimensions.length == pytest.approx(2.0, abs=0.05)
		assert state.dimensions.width == pytest.approx(1.0, abs=0.05)


class TestPostprocessReidMerge:
	def test_reid_merge_stitches_two_fragments_into_one_track(self, tmp_path: Path) -> None:
		# track 1 (frames 0-2) and track 2 (frames 5-7) are the same vehicle, split by an
		# occlusion (frames 3-4 missing) -- a fragment pair a real ReID merge decision would
		# resolve to {"2": 1}.
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "merged.trj"
		merge = tmp_path / "merge.json"
		with ParquetTrackSink(record, _META, scale=1.0) as sink:
			for frame in range(3):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
			for frame in range(5, 8):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=2)])
		merge.write_text(json.dumps({"merges": {"2": 1}}))

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--reid-merge", str(merge)]
		)
		assert result.exit_code == 0, result.output
		assert "merged 1 ReID track ids" in result.output

		vehicle_ids = {state.vehicle_id for frame in read_trj(out).frames for state in frame.states}
		assert vehicle_ids == {1}

	def test_without_reid_merge_fragments_stay_separate(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "unmerged.trj"
		with ParquetTrackSink(record, _META, scale=1.0) as sink:
			for frame in range(3):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
			for frame in range(5, 8):
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=2)])

		result = CliRunner().invoke(app, [str(record), "--out", str(out)])
		assert result.exit_code == 0, result.output

		vehicle_ids = {state.vehicle_id for frame in read_trj(out).frames for state in frame.states}
		assert vehicle_ids == {1, 2}


class TestPostprocessFootprint:
	def test_footprint_replaces_bbox_derived_dimensions(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "footprint.trj"
		footprint = tmp_path / "footprint.parquet"
		with ParquetTrackSink(record, _META, scale=1.0) as sink:
			for frame in range(5):
				# bbox 4x2 -> dims would be 4.0 x 2.0 without a footprint.
				sink.record(frame, [_tracked(x=10.0 * frame + 20.0, y=50.0, track_id=1)])
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
			app, [str(record), "--out", str(out), "--footprint", str(footprint)]
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
		_write_record(record, scale=1.0)
		with FootprintParquetSink(footprint):
			pass  # no footprint rows at all -> nothing covered

		result = CliRunner().invoke(
			app, [str(record), "--out", str(out), "--footprint", str(footprint)]
		)
		assert result.exit_code == 0, result.output

		state = read_trj(out).frames[3].states[0]
		# bbox 4x2 px, scale 1.0 -> unchanged dims (same as the no-footprint baseline).
		assert state.dimensions.length == pytest.approx(4.0, abs=0.1)
		assert state.dimensions.width == pytest.approx(2.0, abs=0.1)

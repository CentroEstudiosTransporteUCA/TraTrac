"""Integration tests for ``tratrac-postprocess``, focused on the MVP2 ``--calibration`` path.

Builds a small Parquet track record, runs the CLI, and reads the emitted ``.trj`` back to
assert that with a calibration the coordinates are world metres + ``DIMENSIONS.Scale = 1.0``,
and that without one the pre-MVP2 image-space path is unchanged. See vault/06_mvp2.md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tratrac.cli_postprocess import app
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.export.ssam_trj import read_trj
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


def _tracked(x: float, y: float) -> TrackedDetection:
	# bbox top-left (x, y), size 4x2 -> centre (x + 2, y + 1).
	return TrackedDetection(
		track_id=1,
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


def _interior_centroid(trj_path: Path) -> tuple[float, float]:
	"""The track's centroid at an interior frame (frame 3), away from filter transients."""
	recording = read_trj(trj_path)
	frame = recording.frames[3]
	state = frame.states[0]
	return state.centroid.x, state.centroid.y


class TestPostprocessCalibration:
	def test_without_calibration_coordinates_stay_image_space(self, tmp_path: Path) -> None:
		record = tmp_path / "tracks.parquet"
		out = tmp_path / "image.trj"
		_write_record(record, scale=1.0)

		result = CliRunner().invoke(app, [str(record), "--out", str(out)])
		assert result.exit_code == 0, result.output

		assert read_trj(out).scale == pytest.approx(1.0)
		cx, cy = _interior_centroid(out)
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
		cx, cy = _interior_centroid(out)
		# image centre (52, 51) * 0.5 -> world (26, 25.5)
		assert cx == pytest.approx(26.0, abs=0.5)
		assert cy == pytest.approx(25.5, abs=0.5)

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

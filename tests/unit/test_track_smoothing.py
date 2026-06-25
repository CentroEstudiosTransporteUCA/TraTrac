"""Tests for the post-process pass: smooth_to_states, exclusion, and the tratrac-postprocess CLI."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tratrac.application.track_smoothing import TrackSample, smooth_to_states
from tratrac.cli_postprocess import app
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox, Point2D
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink


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
		assert smooth_to_states(1, [], 1.0, pos_noise=2.0, jerk=20.0) == []

	def test_produces_state_per_sample_with_metric_scaling(self) -> None:
		samples = _samples(30, vx=10.0)
		states = smooth_to_states(7, samples, 0.5, pos_noise=1.0, jerk=10.0)
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
		states = smooth_to_states(1, samples, 1.0, pos_noise=2.0, jerk=20.0)
		# No motion -> heading falls back to bbox major axis (width >= height -> east).
		assert states[-1].heading.dx == 1.0


def _write_tracks(path: Path, samples: list[TrackSample]) -> None:
	meta = VideoMetadata(width=1920, height=1080, fps=10.0, total_frames=len(samples))
	with ParquetTrackSink(path, meta, scale=1.0) as sink:
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


class TestSmoothCli:
	def test_smooths_tracks_into_parseable_trj(self, tmp_path: Path) -> None:
		tracks = tmp_path / "tracks.parquet"
		out = tmp_path / "smooth.trj"
		_write_tracks(tracks, _samples(20, vx=8.0))

		result = CliRunner().invoke(app, [str(tracks), "--out", str(out)])
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
		_write_tracks(tracks, _samples(5, vx=8.0))
		out.write_text("existing")
		result = CliRunner().invoke(app, [str(tracks), "--out", str(out)])
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
		with ParquetTrackSink(record, meta, scale=1.0) as sink:
			for i in range(5):
				# track 1 sits inside the zone [0,50]^2; track 2 is far outside.
				sink.record(i, [_track(1, 10.0, 10.0), _track(2, 300.0, 300.0)])

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
			app, [str(record), "--out", str(out), "--exclusion-zones", str(zones)]
		)
		assert result.exit_code == 0, result.output
		assert "dropped 1 excluded tracks" in result.output
		assert out.exists()

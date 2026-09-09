"""Tests for the track record (the run's primary output, read by tratrac-smooth)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink, read_tracks

_META = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=900)


def _tracked(
	track_id: int,
	x: float,
	y: float,
	w: float,
	h: float,
	*,
	angle: float | None = None,
	oriented_size: tuple[float, float] | None = None,
) -> TrackedDetection:
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=x, y=y, width=w, height=h),
			score=0.8,
			vehicle_class=VehicleClass.CAR,
			angle=angle,
			oriented_size=oriented_size,
		),
	)


class TestParquetTrackSinkRoundTrip:
	def test_round_trips_metadata_and_observations(self, tmp_path: Path) -> None:
		path = tmp_path / "tracks.parquet"
		with ParquetTrackSink(path, _META, scale=0.05) as sink:
			sink.record(0, [_tracked(1, 10.0, 20.0, 4.0, 2.0)])
			sink.record(1, [_tracked(1, 12.0, 20.0, 4.0, 2.0), _tracked(2, 100.0, 50.0, 6.0, 3.0)])

		recording = read_tracks(path)
		assert recording.metadata == _META
		assert recording.scale == 0.05
		assert len(recording.observations) == 3
		first = recording.observations[0]
		# bbox center: (x + w/2, y + h/2) = (12, 21)
		assert (first.frame_index, first.track_id) == (0, 1)
		assert (first.cx, first.cy) == (12.0, 21.0)
		assert first.vehicle_class is VehicleClass.CAR
		assert first.score == 0.8
		assert first.angle is None
		assert first.obb_w is None
		assert first.obb_h is None

	def test_round_trips_obb_fields(self, tmp_path: Path) -> None:
		path = tmp_path / "obb.parquet"
		with ParquetTrackSink(path, _META, scale=0.05) as sink:
			sink.record(
				0,
				[
					_tracked(1, 10.0, 20.0, 4.0, 2.0, angle=0.5, oriented_size=(4.2, 2.1)),
					_tracked(2, 100.0, 50.0, 6.0, 3.0),  # AABB-only, no OBB fields
				],
			)

		observations = {o.track_id: o for o in read_tracks(path).observations}
		assert observations[1].angle == pytest.approx(0.5)
		assert observations[1].obb_w == pytest.approx(4.2)
		assert observations[1].obb_h == pytest.approx(2.1)
		assert observations[2].angle is None
		assert observations[2].obb_w is None
		assert observations[2].obb_h is None

	def test_reads_record_written_before_obb_columns_existed(self, tmp_path: Path) -> None:
		path = tmp_path / "old.parquet"
		schema = pa.schema(
			[
				("frame", pa.int64()),
				("track_id", pa.int64()),
				("cx", pa.float64()),
				("cy", pa.float64()),
				("w", pa.float64()),
				("h", pa.float64()),
				("vehicle_class", pa.string()),
				("score", pa.float64()),
			]
		).with_metadata(
			{
				b"fps": b"30.0",
				b"width": b"1920",
				b"height": b"1080",
				b"total_frames": b"900",
				b"meters_per_pixel": b"0.05",
			}
		)
		table = pa.table(
			{
				"frame": [0],
				"track_id": [1],
				"cx": [12.0],
				"cy": [21.0],
				"w": [4.0],
				"h": [2.0],
				"vehicle_class": ["car"],
				"score": [0.8],
			},
			schema=schema,
		)
		pq.write_table(table, path)

		observation = read_tracks(path).observations[0]
		assert observation.angle is None
		assert observation.obb_w is None
		assert observation.obb_h is None

	def test_empty_record_round_trips_with_metadata(self, tmp_path: Path) -> None:
		path = tmp_path / "empty.parquet"
		with ParquetTrackSink(path, _META, scale=0.05):
			pass  # no observations (e.g. a clip with no detections)
		recording = read_tracks(path)
		assert recording.metadata == _META
		assert recording.observations == []

	def test_record_outside_context_raises(self, tmp_path: Path) -> None:
		sink = ParquetTrackSink(tmp_path / "t.parquet", _META, scale=1.0)
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(0, [_tracked(1, 0.0, 0.0, 2.0, 2.0)])

	def test_non_parquet_file_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "bogus.parquet"
		path.write_text("not a parquet file")
		with pytest.raises(ValueError, match="not a readable Parquet"):
			read_tracks(path)

	def test_missing_schema_metadata_raises(self, tmp_path: Path) -> None:
		# A valid parquet with the right columns but no run metadata in the schema.
		path = tmp_path / "no_meta.parquet"
		table = pa.table({col: [] for col in ("frame", "track_id", "cx", "cy", "w", "h")})
		pq.write_table(table, path)
		with pytest.raises(ValueError, match="schema metadata"):
			read_tracks(path)

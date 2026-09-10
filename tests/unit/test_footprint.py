"""Tests for the footprint sidecar (Group D1). No segmentation model involved — this is pure
Parquet round-trip, mirroring tests/unit/test_tracks.py's shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from tratrac.domain.geometry import Point2D, Polygon
from tratrac.infrastructure.tracks.footprint_parquet import (
	FootprintParquetSink,
	read_footprints,
)


def _triangle(x: float, y: float) -> Polygon:
	return Polygon((Point2D(x, y), Point2D(x + 4.0, y), Point2D(x + 2.0, y + 3.0)))


class TestFootprintParquetRoundTrip:
	def test_round_trips_polygons(self, tmp_path: Path) -> None:
		path = tmp_path / "footprints.parquet"
		with FootprintParquetSink(path) as sink:
			sink.record(0, 1, _triangle(10.0, 20.0))
			sink.record(1, 1, _triangle(12.0, 20.0))
			sink.record(1, 2, _triangle(100.0, 50.0))

		observations = read_footprints(path)
		assert len(observations) == 3
		first = observations[0]
		assert (first.frame_index, first.track_id) == (0, 1)
		assert first.polygon.vertices[0] == Point2D(10.0, 20.0)
		assert len(first.polygon.vertices) == 3

	def test_empty_sidecar_round_trips(self, tmp_path: Path) -> None:
		path = tmp_path / "empty.parquet"
		with FootprintParquetSink(path):
			pass
		assert read_footprints(path) == []

	def test_record_outside_context_raises(self, tmp_path: Path) -> None:
		sink = FootprintParquetSink(tmp_path / "t.parquet")
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(0, 1, _triangle(0.0, 0.0))

	def test_non_parquet_file_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "bogus.parquet"
		path.write_text("not a parquet file")
		with pytest.raises(ValueError, match="not a readable Parquet"):
			read_footprints(path)

	def test_preserves_polygons_with_more_than_three_vertices(self, tmp_path: Path) -> None:
		path = tmp_path / "footprints.parquet"
		pentagon = Polygon(
			(
				Point2D(0.0, 0.0),
				Point2D(4.0, 0.0),
				Point2D(5.0, 2.0),
				Point2D(2.0, 4.0),
				Point2D(-1.0, 2.0),
			)
		)
		with FootprintParquetSink(path) as sink:
			sink.record(5, 3, pentagon)
		[observation] = read_footprints(path)
		assert len(observation.polygon.vertices) == 5
		assert observation.polygon.vertices[2] == Point2D(5.0, 2.0)

"""Round-trip tests for the smoothed track record (image-space, `--smoothed-record`)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tratrac.domain.frame import VideoMetadata
from tratrac.infrastructure.tracks.smoothed_parquet import (
	SmoothedObservation,
	SmoothedTrackParquetSink,
	read_smoothed_tracks,
)

_META = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=900)


class TestSmoothedTrackParquetRoundTrip:
	def test_round_trips_metadata_and_observations(self, tmp_path: Path) -> None:
		path = tmp_path / "smoothed.parquet"
		with SmoothedTrackParquetSink(path, _META) as sink:
			sink.record(
				SmoothedObservation(
					frame_index=0,
					track_id=1,
					cx=100.0,
					cy=200.0,
					angle=math.pi / 4,
					length=8.0,
					width=4.0,
					link_id=3,
					lane_id=2,
				)
			)
			sink.record(
				SmoothedObservation(
					frame_index=1,
					track_id=1,
					cx=105.0,
					cy=200.0,
					angle=0.0,
					length=8.0,
					width=4.0,
				)
			)

		recording = read_smoothed_tracks(path)
		assert recording.metadata == _META
		assert len(recording.observations) == 2
		first = recording.observations[0]
		assert (first.frame_index, first.track_id) == (0, 1)
		assert (first.cx, first.cy) == (100.0, 200.0)
		assert first.angle == math.pi / 4
		assert (first.length, first.width) == (8.0, 4.0)
		assert (first.link_id, first.lane_id) == (3, 2)
		# link_id/lane_id default to 0 when not passed.
		second = recording.observations[1]
		assert (second.link_id, second.lane_id) == (0, 0)

	def test_writer_requires_context_manager(self, tmp_path: Path) -> None:
		sink = SmoothedTrackParquetSink(tmp_path / "x.parquet", _META)
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(SmoothedObservation(0, 1, 0.0, 0.0, 0.0, 1.0, 1.0))

	def test_read_missing_file_raises_value_error(self, tmp_path: Path) -> None:
		with pytest.raises(ValueError, match="not a readable"):
			read_smoothed_tracks(tmp_path / "missing.parquet")

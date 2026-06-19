"""Tests for the track-observation sidecar (export B) and the RecordingTracker decorator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import Frame, VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.tracking.recording import RecordingTracker
from tratrac.infrastructure.tracks.csv import CsvTrackSink, read_tracks

_META = VideoMetadata(width=1920, height=1080, fps=30.0, total_frames=900)
_PIXELS = np.zeros((4, 4, 3), dtype=np.uint8)


def _tracked(track_id: int, x: float, y: float, w: float, h: float) -> TrackedDetection:
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=x, y=y, width=w, height=h),
			score=0.8,
			vehicle_class=VehicleClass.CAR,
		),
	)


class TestCsvTrackSinkRoundTrip:
	def test_round_trips_metadata_and_observations(self, tmp_path: Path) -> None:
		path = tmp_path / "tracks.csv"
		with CsvTrackSink(path, _META, scale=0.05) as sink:
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

	def test_record_outside_context_raises(self, tmp_path: Path) -> None:
		sink = CsvTrackSink(tmp_path / "t.csv", _META, scale=1.0)
		with pytest.raises(RuntimeError, match="context manager"):
			sink.record(0, [_tracked(1, 0.0, 0.0, 2.0, 2.0)])

	def test_missing_header_raises(self, tmp_path: Path) -> None:
		path = tmp_path / "tracks.csv"
		path.write_text("frame,track_id,cx,cy,w,h,vehicle_class,score\n0,1,1,1,2,2,car,0.5\n")
		with pytest.raises(ValueError, match="header line"):
			read_tracks(path)


class _FakeTracker:
	"""A Tracker stub returning a fixed tracked list."""

	def __init__(self, tracked: list[TrackedDetection]) -> None:
		self._tracked = tracked

	def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedDetection]:
		return self._tracked


class _ListSink:
	def __init__(self) -> None:
		self.calls: list[tuple[int, int]] = []

	def record(self, frame_index: int, tracked: list[TrackedDetection]) -> None:
		self.calls.append((frame_index, len(tracked)))


class TestRecordingTracker:
	def test_tees_tracked_and_returns_unchanged(self) -> None:
		tracked = [_tracked(1, 0.0, 0.0, 2.0, 2.0)]
		sink = _ListSink()
		decorated = RecordingTracker(_FakeTracker(tracked), sink)
		out = decorated.update(Frame(index=7, pixels=_PIXELS), [])
		assert out is tracked  # passes the tracker's output straight through
		assert sink.calls == [(7, 1)]  # teed with the frame index

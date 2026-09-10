"""Integration: iter_track_frames re-opens a video windowed to a track record's span and
pairs each frame with its recorded observations via a real decode.

Writes a tiny synthetic clip (no detector, no network) and a matching Parquet track record,
modeled on tests/integration/test_video_window.py + tests/integration/test_render.py. Skips if
the local OpenCV build can't write the fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.replay.track_frames import iter_track_frames
from tratrac.infrastructure.tracks.parquet import ParquetTrackSink, read_tracks

_WIDTH = 64
_HEIGHT = 48
_FPS = 10
_N_FRAMES = 10


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
	path = tmp_path / "clip.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
	rng = np.random.default_rng(seed=11)
	for _ in range(_N_FRAMES):
		writer.write(rng.integers(0, 256, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8))
	writer.release()
	if not path.exists():
		pytest.skip(f"Could not write fixture video (codec issue): {path}")
	return path


def _tracked(track_id: int) -> TrackedDetection:
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(
			bbox=BoundingBox(x=1.0, y=1.0, width=4.0, height=2.0),
			score=0.9,
			vehicle_class=VehicleClass.CAR,
		),
	)


def _write_record(path: Path, *, observed_frames: range) -> None:
	meta = VideoMetadata(width=_WIDTH, height=_HEIGHT, fps=float(_FPS), total_frames=_N_FRAMES)
	with ParquetTrackSink(path, meta) as sink:
		for frame in observed_frames:
			sink.record(frame, [_tracked(track_id=1)])


def test_yields_observed_frames_with_matching_observations(
	synthetic_video: Path, tmp_path: Path
) -> None:
	record_path = tmp_path / "tracks.parquet"
	_write_record(record_path, observed_frames=range(2, 6))  # frames 2, 3, 4, 5
	recording = read_tracks(record_path)

	results = list(iter_track_frames(synthetic_video, recording, padding_seconds=0.0))

	indices = [frame.index for frame, _observations in results]
	assert indices[0] == 2
	assert indices[-1] == 5
	for frame, observations in results:
		assert len(observations) == 1
		assert observations[0].frame_index == frame.index
		assert observations[0].track_id == 1


def test_gap_between_observations_still_yields_with_empty_list(
	synthetic_video: Path, tmp_path: Path
) -> None:
	record_path = tmp_path / "tracks.parquet"
	meta = VideoMetadata(width=_WIDTH, height=_HEIGHT, fps=float(_FPS), total_frames=_N_FRAMES)
	with ParquetTrackSink(record_path, meta) as sink:
		sink.record(2, [_tracked(track_id=1)])
		sink.record(5, [_tracked(track_id=1)])  # frames 3, 4 are a gap
	recording = read_tracks(record_path)

	results = {
		frame.index: observations
		for frame, observations in iter_track_frames(
			synthetic_video, recording, padding_seconds=0.0
		)
	}
	assert results[3] == []
	assert results[4] == []
	assert len(results[2]) == 1
	assert len(results[5]) == 1


def test_padding_extends_past_the_last_observation(synthetic_video: Path, tmp_path: Path) -> None:
	record_path = tmp_path / "tracks.parquet"
	_write_record(record_path, observed_frames=range(0, 3))  # frames 0, 1, 2
	recording = read_tracks(record_path)

	# 0.3s padding at 10fps -> 3 extra frames -> up to frame 5.
	results = list(iter_track_frames(synthetic_video, recording, padding_seconds=0.3))
	indices = [frame.index for frame, _observations in results]
	assert indices[-1] >= 5
	assert results[-1][1] == []  # padded frames have no observations


def test_empty_recording_yields_nothing(synthetic_video: Path, tmp_path: Path) -> None:
	record_path = tmp_path / "empty.parquet"
	meta = VideoMetadata(width=_WIDTH, height=_HEIGHT, fps=float(_FPS), total_frames=_N_FRAMES)
	with ParquetTrackSink(record_path, meta):
		pass
	recording = read_tracks(record_path)

	assert list(iter_track_frames(synthetic_video, recording)) == []

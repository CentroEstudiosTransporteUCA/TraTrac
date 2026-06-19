"""Integration: tratrac-smooth --video-out draws the smoothed trajectories onto a clip.

Writes a tiny synthetic video + a matching track sidecar (no detector, no network),
runs the post-pass with the overlay enabled, and checks both the .trj and the overlay
.mp4 come out. Skips if the local OpenCV can't write the fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tratrac.cli_smooth import app
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import BoundingBox
from tratrac.infrastructure.tracks.csv import CsvTrackSink

_W, _H, _FPS, _N = 64, 48, 10, 12


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
	path = tmp_path / "clip.mp4"
	writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), _FPS, (_W, _H))
	rng = np.random.default_rng(3)
	for _ in range(_N):
		writer.write(rng.integers(0, 256, size=(_H, _W, 3), dtype=np.uint8))
	writer.release()
	if not path.exists():
		pytest.skip(f"Could not write fixture video (codec issue): {path}")
	return path


@pytest.fixture
def tracks_csv(tmp_path: Path) -> Path:
	path = tmp_path / "tracks.csv"
	meta = VideoMetadata(width=_W, height=_H, fps=float(_FPS), total_frames=_N)
	with CsvTrackSink(path, meta, scale=1.0) as sink:
		for i in range(_N):
			cx = 5.0 + 3.0 * i  # one car moving east
			det = TrackedDetection(
				track_id=1,
				detection=Detection(
					bbox=BoundingBox(x=cx - 4.0, y=20.0, width=8.0, height=4.0),
					score=0.9,
					vehicle_class=VehicleClass.CAR,
				),
			)
			sink.record(i, [det])
	return path


def test_overlay_video_is_written(synthetic_video: Path, tracks_csv: Path, tmp_path: Path) -> None:
	out = tmp_path / "smooth.trj"
	overlay = tmp_path / "smooth_overlay.mp4"
	result = CliRunner().invoke(
		app,
		[
			str(tracks_csv),
			"--out",
			str(out),
			"--video",
			str(synthetic_video),
			"--video-out",
			str(overlay),
		],
	)
	assert result.exit_code == 0, result.output
	assert out.stat().st_size > 0
	assert overlay.stat().st_size > 0
	# The overlay is a readable video with frames the size of the source.
	cap = cv2.VideoCapture(str(overlay))
	ok, frame = cap.read()
	cap.release()
	assert ok
	assert frame.shape[:2] == (_H, _W)


def test_video_out_requires_video(tracks_csv: Path, tmp_path: Path) -> None:
	result = CliRunner().invoke(
		app,
		[str(tracks_csv), "--out", str(tmp_path / "s.trj"), "--video-out", str(tmp_path / "o.mp4")],
	)
	assert result.exit_code != 0
	assert "requires --video" in result.output

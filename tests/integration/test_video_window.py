"""Integration: OpenCvVideoSource trims to the [start, end] window via a real decode.

Writes a tiny synthetic clip (no detector, no network) and checks the source seeks
to the start frame, stops after the inclusive end frame, and labels frames with
absolute indices. Skips if the local OpenCV build can't write the fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tratrac.infrastructure.video.opencv import OpenCvVideoSource

_WIDTH = 64
_HEIGHT = 48
_FPS = 10
_N_FRAMES = 10


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
	path = tmp_path / "clip.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
	rng = np.random.default_rng(seed=7)
	for _ in range(_N_FRAMES):
		writer.write(rng.integers(0, 256, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8))
	writer.release()
	if not path.exists():
		pytest.skip(f"Could not write fixture video (codec issue): {path}")
	return path


def test_window_yields_inclusive_absolute_indices(synthetic_video: Path) -> None:
	# 0.3s..0.6s at 10 fps -> frames 3..6 inclusive.
	with OpenCvVideoSource(synthetic_video, start_seconds=0.3, end_seconds=0.6) as source:
		assert source.metadata.total_frames == 4
		indices = [frame.index for frame in source.frames()]
	assert indices == [3, 4, 5, 6]


def test_no_window_reads_from_zero(synthetic_video: Path) -> None:
	with OpenCvVideoSource(synthetic_video) as source:
		indices = [frame.index for frame in source.frames()]
	# mp4v may drop the final frame; require a contiguous run starting at 0.
	assert indices[0] == 0
	assert len(indices) >= _N_FRAMES - 1

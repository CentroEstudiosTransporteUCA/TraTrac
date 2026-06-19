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


def test_process_fps_decimates_decode(synthetic_video: Path) -> None:
	# 10 fps source, cap to 5 fps -> keep every other absolute index, and report the
	# decimated count as total_frames (frames this run will process).
	with OpenCvVideoSource(synthetic_video, process_fps=5.0) as source:
		assert source.metadata.total_frames == 5
		indices = [frame.index for frame in source.frames()]
	# frame 8 is the last kept grid point; frame 9 (last) may be dropped by mp4v anyway.
	assert indices[:4] == [0, 2, 4, 6]
	assert indices[-1] in (6, 8)


def test_process_fps_composes_with_window(synthetic_video: Path) -> None:
	# Window 0.2..0.9s (frames 2..9) capped to 5 fps -> keep 2, 4, 6, 8.
	with OpenCvVideoSource(
		synthetic_video, start_seconds=0.2, end_seconds=0.9, process_fps=5.0
	) as src:
		indices = [frame.index for frame in src.frames()]
	assert indices[:3] == [2, 4, 6]

"""Integration: tratrac-render draws a .trj's trajectories onto its source clip.

Writes a tiny synthetic video + a matching .trj (via SsamTrjExporter, no detector,
no network), runs the renderer, and checks the overlay .mp4 comes out readable at the
source size. Skips if the local OpenCV can't write the fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tratrac.cli_render import app
from tratrac.domain.frame import VideoMetadata
from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter

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
def trj_file(tmp_path: Path) -> Path:
	path = tmp_path / "run.trj"
	meta = VideoMetadata(width=_W, height=_H, fps=float(_FPS), total_frames=_N)
	with SsamTrjExporter(path, meta, scale=1.0) as exporter:
		for i in range(_N):
			cx = 5.0 + 3.0 * i  # one car moving east
			state = VehicleState(
				vehicle_id=1,
				timestamp_seconds=i / _FPS,
				centroid=Point2D(cx, 22.0),
				heading=Heading(1.0, 0.0),
				dimensions=Dimensions(length=8.0, width=4.0),
				velocity=Vector2D(30.0, 0.0),
				acceleration=0.0,
			)
			exporter.emit_frame(i / _FPS, [state])
	return path


def test_render_writes_overlay(synthetic_video: Path, trj_file: Path, tmp_path: Path) -> None:
	out = tmp_path / "overlay.mp4"
	result = CliRunner().invoke(
		app, [str(synthetic_video), "--trj", str(trj_file), "--out", str(out)]
	)
	assert result.exit_code == 0, result.output
	assert out.stat().st_size > 0
	# The overlay is a readable video with frames the size of the source.
	cap = cv2.VideoCapture(str(out))
	ok, frame = cap.read()
	cap.release()
	assert ok
	assert frame.shape[:2] == (_H, _W)


def test_existing_out_needs_force(synthetic_video: Path, trj_file: Path, tmp_path: Path) -> None:
	out = tmp_path / "overlay.mp4"
	out.write_bytes(b"x")
	result = CliRunner().invoke(
		app, [str(synthetic_video), "--trj", str(trj_file), "--out", str(out)]
	)
	assert result.exit_code != 0
	assert "already exists" in result.output

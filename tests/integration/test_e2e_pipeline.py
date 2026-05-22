"""End-to-end smoke test: synthetic video -> full pipeline -> valid .trj.

Uses the real RT-DETR + BoT-SORT stack. The first run downloads the RT-DETR
checkpoint (~80 MB); subsequent runs hit the HF cache. Synthetic black frames
won't trigger detections, so the resulting .trj has FORMAT + DIMENSIONS + N
empty TIMESTEPs — exactly the wire we want to verify.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from tratrac.application.orientation import OrientationEstimator
from tratrac.application.pipeline import TrajectoryPipeline
from tratrac.infrastructure.detection.rt_detr import RtDetrDetector
from tratrac.infrastructure.export.ssam_trj import SsamTrjExporter
from tratrac.infrastructure.tracking.boxmot_bot_sort import BoxmotBotSortTracker
from tratrac.infrastructure.video.opencv import OpenCvVideoSource

_WIDTH = 128
_HEIGHT = 96
_FPS = 30
_N_FRAMES = 5

_FORMAT_SIZE = 6
_DIMENSIONS_SIZE = 22
_TIMESTEP_SIZE = 5


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
	"""Generate a tiny synthetic video. No real vehicles — we test wiring, not detection."""
	path = tmp_path / "synthetic.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
	rng = np.random.default_rng(seed=42)
	for _ in range(_N_FRAMES):
		# Random noise frames; RT-DETR shouldn't find vehicles here.
		frame = rng.integers(0, 256, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8)
		writer.write(frame)
	writer.release()
	if not path.exists():
		pytest.skip(f"Could not write fixture video (codec issue): {path}")
	return path


@pytest.mark.slow
def test_pipeline_produces_parseable_trj(synthetic_video: Path, tmp_path: Path) -> None:
	out = tmp_path / "out.trj"
	with OpenCvVideoSource(synthetic_video) as source:
		detector = RtDetrDetector(
			checkpoint="PekingU/rtdetr_r18vd", device="cpu", score_threshold=0.5
		)
		tracker = BoxmotBotSortTracker(source.metadata)
		exporter = SsamTrjExporter(out, source.metadata)
		pipeline = TrajectoryPipeline(
			video=source,
			detector=detector,
			tracker=tracker,
			exporter=exporter,
			orientation=OrientationEstimator(),
		)
		n_frames = pipeline.run()

	assert n_frames >= 1  # codec may drop the last frame; we want >=1 to confirm we ran.
	data = out.read_bytes()

	# FORMAT record.
	record_type, endian, version = struct.unpack("<BBf", data[:_FORMAT_SIZE])
	assert record_type == 0
	assert endian == ord("L")
	assert math.isclose(version, 1.04, abs_tol=1e-3)

	# DIMENSIONS record matches the synthetic video.
	header_end = _FORMAT_SIZE + _DIMENSIONS_SIZE
	rec_type, units, _scale, min_x, min_y, max_x, max_y = struct.unpack(
		"<BBfiiii", data[_FORMAT_SIZE:header_end]
	)
	assert rec_type == 1
	assert units == 1
	assert (min_x, min_y, max_x, max_y) == (0, 0, _WIDTH, _HEIGHT)

	# Walk the rest of the file: expect alternating TIMESTEP / (no VEHICLE).
	# We don't assert frame count exactly — mp4v may swallow the last frame.
	timestep_count = 0
	offset = header_end
	while offset < len(data):
		marker = data[offset]
		if marker == 2:  # TIMESTEP
			timestep_count += 1
			offset += _TIMESTEP_SIZE
		elif marker == 3:  # VEHICLE — none expected from noise frames, but tolerate
			from tratrac.infrastructure.export.ssam_trj import _VEHICLE_STRUCT

			offset += _VEHICLE_STRUCT.size
		else:
			pytest.fail(f"Unexpected record type byte {marker} at offset {offset}")
	assert timestep_count >= 1

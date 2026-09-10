"""Integration test for ``tratrac-stabilize``: walks a small synthetic clip end to end
(no detector) and checks the transforms file + anchor manifest it writes are valid and
consistent with each other. Writes a tiny synthetic clip (no network), modeled on
tests/integration/test_replay.py. Skips if the local OpenCV build can't write the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tratrac.cli_stabilize import app
from tratrac.infrastructure.anchors.manifest import read_manifest
from tratrac.infrastructure.transform.sink import read_transforms

_WIDTH = 200
_HEIGHT = 200
_FPS = 10
_N_FRAMES = 6


def _textured_image() -> np.ndarray:
	rng = np.random.default_rng(seed=3)
	return rng.integers(0, 256, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
	"""A textured background translating by a small, constant per-frame shift — enough for
	ORB to track without drifting past the anchor's overlap threshold."""
	path = tmp_path / "clip.mp4"
	fourcc = cv2.VideoWriter.fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
	base = _textured_image()
	for i in range(_N_FRAMES):
		matrix = np.array([[1.0, 0.0, 2.0 * i], [0.0, 1.0, 0.0]], dtype=np.float64)
		frame = cv2.warpAffine(base, matrix, (_WIDTH, _HEIGHT))
		writer.write(frame)
	writer.release()
	if not path.exists():
		pytest.skip(f"Could not write fixture video (codec issue): {path}")
	return path


@pytest.fixture
def whole_frame_zones(tmp_path: Path) -> Path:
	"""A background_zones.json covering the entire frame — masking isn't the point here."""
	path = tmp_path / "background_zones.json"
	path.write_text(
		json.dumps(
			{
				"background_zones": [
					{
						"reference_frame": 0,
						"vertices": [[0, 0], [_WIDTH, 0], [_WIDTH, _HEIGHT], [0, _HEIGHT]],
					}
				]
			}
		)
	)
	return path


def test_stabilize_writes_a_complete_transforms_file_and_anchor_manifest(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	out = tmp_path / "transforms.jsonl"
	anchors_dir = tmp_path / "anchors"

	result = CliRunner().invoke(
		app,
		[
			str(synthetic_video),
			"--background-zones",
			str(whole_frame_zones),
			"--out",
			str(out),
			"--anchors-dir",
			str(anchors_dir),
		],
	)
	assert result.exit_code == 0, result.output
	assert out.exists()
	assert not out.with_name(out.name + ".partial").exists()

	table = read_transforms(out)
	for frame_index in range(_N_FRAMES):
		assert table.at(frame_index) is not None

	references = read_manifest(anchors_dir / "manifest.json")
	assert len(references) >= 1
	# Every exported anchor PNG actually exists.
	for ref in references:
		assert (anchors_dir / ref.image_name).exists()
	# The anchor manifest's poses agree with the dense per-frame table at the same
	# frames -- both are recorded off the identical estimate() call (RecordingEgoMotionEstimator
	# and AnchorRecordingEgoMotionEstimator wrap the same underlying OrbEgoMotionEstimator).
	for ref in references:
		assert table.at(ref.frame_index) == ref.pose


def test_refuses_to_overwrite_without_force(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	out = tmp_path / "transforms.jsonl"
	out.write_text("existing")
	anchors_dir = tmp_path / "anchors"

	result = CliRunner().invoke(
		app,
		[
			str(synthetic_video),
			"--background-zones",
			str(whole_frame_zones),
			"--out",
			str(out),
			"--anchors-dir",
			str(anchors_dir),
		],
	)
	assert result.exit_code != 0
	assert "force" in result.output.lower()

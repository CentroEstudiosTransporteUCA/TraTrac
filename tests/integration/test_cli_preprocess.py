"""Integration test for ``tratrac-preprocess``: walks a small synthetic clip end to end
(no detector) and checks the transforms file + anchor PNGs it writes are valid and
consistent with each other. Writes a tiny synthetic clip (no network), modeled on
tests/integration/test_replay.py. Skips if the local OpenCV build can't write the fixture.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tratrac.cli_preprocess import app
from tratrac.infrastructure.transform.sink import read_ego_motion

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


def test_preprocess_writes_a_complete_transforms_file_and_anchor_pngs(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	out = tmp_path / "transforms.jsonl"
	anchors_dir = tmp_path / "anchors"

	result = CliRunner().invoke(
		app,
		[
			"estimate",
			str(synthetic_video),
			"--background-zones",
			str(whole_frame_zones),
			"--out",
			str(out),
			"--anchors-dir",
			str(anchors_dir),
			"--meters-per-pixel",
			"0.05",
		],
	)
	assert result.exit_code == 0, result.output
	assert out.exists()
	assert not out.with_name(out.name + ".partial").exists()

	table = read_ego_motion(out)
	for frame_index in range(_N_FRAMES):
		assert table.function_at(frame_index) is not None

	# No manifest is written -- an anchor's pose is looked up in the dense per-frame
	# table above by the frame index its filename already carries.
	assert not (anchors_dir / "manifest.json").exists()
	anchor_pngs = sorted(anchors_dir.glob("frame_*.png"))
	assert len(anchor_pngs) >= 1
	for png in anchor_pngs:
		match = re.fullmatch(r"frame_(\d+)\.png", png.name)
		assert match is not None
		assert table.function_at(int(match.group(1))) is not None


def test_project_fits_a_homography_and_appends_it_to_the_same_file(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	out = tmp_path / "transforms.jsonl"
	estimate_result = CliRunner().invoke(
		app,
		[
			"estimate",
			str(synthetic_video),
			"--background-zones",
			str(whole_frame_zones),
			"--out",
			str(out),
			"--meters-per-pixel",
			"0.05",
		],
	)
	assert estimate_result.exit_code == 0, estimate_result.output

	calibration = tmp_path / "calibration.json"
	# A pure 2x scale, world = 2 * image, so the fit is exact and easy to check.
	calibration.write_text(
		json.dumps(
			{
				"correspondences": [
					{"reference_frame": 0, "image": [0, 0], "world": [0, 0]},
					{"reference_frame": 0, "image": [10, 0], "world": [20, 0]},
					{"reference_frame": 0, "image": [10, 10], "world": [20, 20]},
					{"reference_frame": 0, "image": [0, 10], "world": [0, 20]},
				]
			}
		)
	)

	project_result = CliRunner().invoke(
		app, ["project", "--transforms", str(out), "--calibration", str(calibration)]
	)
	assert project_result.exit_code == 0, project_result.output

	from tratrac.domain.geometry import Point2D
	from tratrac.infrastructure.transform.records import HomographyFunction
	from tratrac.infrastructure.transform.sink import read_transform_table

	projection = read_transform_table(out, kinds=(HomographyFunction,))
	assert projection.kinds() == {HomographyFunction}
	for frame_index in range(_N_FRAMES):
		point = projection.apply(Point2D(5.0, 5.0), frame_index)
		assert point.x == pytest.approx(10.0, abs=1e-6)
		assert point.y == pytest.approx(10.0, abs=1e-6)

	# Re-running project (e.g. after tweaking calibration.json) must stay idempotent:
	# it replaces the projection stage, it doesn't pile up duplicate rows.
	rerun_result = CliRunner().invoke(
		app, ["project", "--transforms", str(out), "--calibration", str(calibration)]
	)
	assert rerun_result.exit_code == 0, rerun_result.output
	replaced = read_transform_table(out, kinds=(HomographyFunction,))
	assert replaced.apply(Point2D(5.0, 5.0), 0) == projection.apply(Point2D(5.0, 5.0), 0)


def test_project_drops_the_now_superseded_scale_rows(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	"""``estimate`` writes a whole-canvas scale row per frame; a whole-scene homography also
	lands on the whole-canvas zone. If ``project`` only stripped stale homography rows, the two
	would coexist for the same (frame, zone) and every downstream ``TransformTable.apply`` --
	e.g. tratrac-postprocess's ``--transforms`` read -- would raise "ambiguous"."""
	out = tmp_path / "transforms.jsonl"
	estimate_result = CliRunner().invoke(
		app,
		["estimate", str(synthetic_video), "--out", str(out), "--meters-per-pixel", "0.05"],
	)
	assert estimate_result.exit_code == 0, estimate_result.output

	calibration = tmp_path / "calibration.json"
	calibration.write_text(
		json.dumps(
			{
				"correspondences": [
					{"reference_frame": 0, "image": [0, 0], "world": [0, 0]},
					{"reference_frame": 0, "image": [10, 0], "world": [20, 0]},
					{"reference_frame": 0, "image": [10, 10], "world": [20, 20]},
					{"reference_frame": 0, "image": [0, 10], "world": [0, 20]},
				]
			}
		)
	)
	project_result = CliRunner().invoke(
		app, ["project", "--transforms", str(out), "--calibration", str(calibration)]
	)
	assert project_result.exit_code == 0, project_result.output

	from tratrac.domain.geometry import Point2D
	from tratrac.infrastructure.transform.records import HomographyFunction, ScaleFunction
	from tratrac.infrastructure.transform.sink import read_transform_table

	projection = read_transform_table(out, kinds=(ScaleFunction, HomographyFunction))
	assert projection.kinds() == {HomographyFunction}
	point = projection.apply(Point2D(5.0, 5.0), 0)
	assert point.x == pytest.approx(10.0, abs=1e-6)
	assert point.y == pytest.approx(10.0, abs=1e-6)


def test_refuses_to_overwrite_without_force(
	synthetic_video: Path, whole_frame_zones: Path, tmp_path: Path
) -> None:
	out = tmp_path / "transforms.jsonl"
	out.write_text("existing")
	anchors_dir = tmp_path / "anchors"

	result = CliRunner().invoke(
		app,
		[
			"estimate",
			str(synthetic_video),
			"--background-zones",
			str(whole_frame_zones),
			"--out",
			str(out),
			"--anchors-dir",
			str(anchors_dir),
			"--meters-per-pixel",
			"0.05",
		],
	)
	assert result.exit_code != 0
	assert "force" in result.output.lower()

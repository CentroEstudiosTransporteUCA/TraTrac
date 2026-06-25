"""Tests for the world-projection sidecar reader + cv2 homography fit (MVP2, vault/06).

Exercises ``load_calibration`` (JSON -> ``Calibration`` validation) and ``compute_homography``
(cv2 fit). cv2 is imported, but no model download — this stays a fast unit test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tratrac.domain.geometry import Point2D
from tratrac.domain.world import Calibration, Correspondence
from tratrac.infrastructure.world.calibration import compute_homography, load_calibration


def _write(tmp_path: Path, document: object) -> Path:
	path = tmp_path / "calibration.json"
	path.write_text(json.dumps(document))
	return path


def _square_correspondences() -> list[dict[str, object]]:
	"""Four image<->world pairs, image scaled by 2 (world = image / 2)."""
	return [
		{"reference_frame": 0, "image": [0, 0], "world": [0.0, 0.0]},
		{"reference_frame": 0, "image": [20, 0], "world": [10.0, 0.0]},
		{"reference_frame": 0, "image": [20, 20], "world": [10.0, 10.0]},
		{"reference_frame": 0, "image": [0, 20], "world": [0.0, 10.0]},
	]


class TestLoadCalibration:
	def test_parses_a_valid_document(self, tmp_path: Path) -> None:
		path = _write(tmp_path, {"correspondences": _square_correspondences()})
		calibration = load_calibration(path)
		assert isinstance(calibration, Calibration)
		assert len(calibration.correspondences) == 4
		first = calibration.correspondences[0]
		assert first == Correspondence(
			reference_frame=0, image=Point2D(0.0, 0.0), world=Point2D(0.0, 0.0)
		)

	def test_reference_frame_defaults_to_zero(self, tmp_path: Path) -> None:
		pairs = [{"image": [i, i], "world": [float(i), float(i)]} for i in range(4)]
		calibration = load_calibration(_write(tmp_path, {"correspondences": pairs}))
		assert all(c.reference_frame == 0 for c in calibration.correspondences)

	def test_rejects_fewer_than_four_correspondences(self, tmp_path: Path) -> None:
		path = _write(tmp_path, {"correspondences": _square_correspondences()[:3]})
		with pytest.raises(ValueError, match="at least 4"):
			load_calibration(path)

	def test_rejects_missing_correspondences_key(self, tmp_path: Path) -> None:
		with pytest.raises(ValueError, match="correspondences"):
			load_calibration(_write(tmp_path, {"points": []}))

	def test_rejects_non_array_correspondences(self, tmp_path: Path) -> None:
		with pytest.raises(ValueError, match="must be an array"):
			load_calibration(_write(tmp_path, {"correspondences": {}}))

	def test_rejects_a_correspondence_missing_world(self, tmp_path: Path) -> None:
		pairs = _square_correspondences()
		del pairs[1]["world"]
		with pytest.raises(ValueError, match=r"image.*world"):
			load_calibration(_write(tmp_path, {"correspondences": pairs}))

	def test_rejects_a_malformed_pair(self, tmp_path: Path) -> None:
		pairs = _square_correspondences()
		pairs[2]["image"] = [1, 2, 3]
		with pytest.raises(ValueError, match="number pair"):
			load_calibration(_write(tmp_path, {"correspondences": pairs}))

	def test_rejects_a_boolean_coordinate(self, tmp_path: Path) -> None:
		pairs = _square_correspondences()
		pairs[0]["world"] = [True, 0.0]
		with pytest.raises(ValueError, match="number pair"):
			load_calibration(_write(tmp_path, {"correspondences": pairs}))

	def test_rejects_a_negative_reference_frame(self, tmp_path: Path) -> None:
		pairs = _square_correspondences()
		pairs[0]["reference_frame"] = -1
		with pytest.raises(ValueError, match="reference_frame"):
			load_calibration(_write(tmp_path, {"correspondences": pairs}))

	def test_rejects_invalid_json(self, tmp_path: Path) -> None:
		path = tmp_path / "calibration.json"
		path.write_text("{not json")
		with pytest.raises(ValueError, match="not valid JSON"):
			load_calibration(path)


class TestComputeHomography:
	def test_exact_four_point_fit_maps_image_to_world(self) -> None:
		image = [Point2D(0, 0), Point2D(20, 0), Point2D(20, 20), Point2D(0, 20)]
		world = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
		matrix = compute_homography(image, world)
		# Centre pixel (10,10) -> world (5,5) under the world = image/2 map.
		projected = matrix @ [10.0, 10.0, 1.0]
		assert projected[0] / projected[2] == pytest.approx(5.0)
		assert projected[1] / projected[2] == pytest.approx(5.0)

	def test_overdetermined_fit_uses_ransac(self) -> None:
		image = [Point2D(0, 0), Point2D(20, 0), Point2D(20, 20), Point2D(0, 20), Point2D(10, 10)]
		world = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10), Point2D(5, 5)]
		matrix = compute_homography(image, world)
		projected = matrix @ [0.0, 20.0, 1.0]
		assert projected[0] / projected[2] == pytest.approx(0.0, abs=1e-6)
		assert projected[1] / projected[2] == pytest.approx(10.0)

	def test_mismatched_lengths_raise(self) -> None:
		with pytest.raises(ValueError, match="equal length"):
			compute_homography([Point2D(0, 0)] * 4, [Point2D(0, 0)] * 3)

	def test_too_few_points_raise(self) -> None:
		with pytest.raises(ValueError, match="at least 4"):
			compute_homography([Point2D(0, 0)] * 3, [Point2D(0, 0)] * 3)

	def test_collinear_points_raise_on_the_ransac_path(self) -> None:
		# >4 correspondences go through findHomography, which returns None (no consensus) on
		# fully collinear input. (The exact 4-point path does not detect degeneracy — see the
		# compute_homography docstring.)
		image = [Point2D(i, 0) for i in range(5)]
		world = [Point2D(i, 0) for i in range(5)]
		with pytest.raises(ValueError, match="could not fit"):
			compute_homography(image, world)

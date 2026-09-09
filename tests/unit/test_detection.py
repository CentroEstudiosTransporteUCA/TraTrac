"""Tests for the Detection domain type: score validation and the OBB angle/oriented_size pair."""

from __future__ import annotations

import pytest

from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.geometry import BoundingBox


def _bbox() -> BoundingBox:
	return BoundingBox(x=0.0, y=0.0, width=10.0, height=5.0)


class TestDetection:
	def test_score_out_of_range_raises(self) -> None:
		with pytest.raises(ValueError, match="score"):
			Detection(bbox=_bbox(), score=1.5, vehicle_class=VehicleClass.CAR)

	def test_defaults_to_no_orientation(self) -> None:
		detection = Detection(bbox=_bbox(), score=0.9, vehicle_class=VehicleClass.CAR)
		assert detection.angle is None
		assert detection.oriented_size is None

	def test_angle_and_oriented_size_together_is_valid(self) -> None:
		detection = Detection(
			bbox=_bbox(),
			score=0.9,
			vehicle_class=VehicleClass.CAR,
			angle=0.3,
			oriented_size=(12.0, 6.0),
		)
		assert detection.angle == 0.3
		assert detection.oriented_size == (12.0, 6.0)

	def test_angle_without_oriented_size_raises(self) -> None:
		with pytest.raises(ValueError, match="together"):
			Detection(bbox=_bbox(), score=0.9, vehicle_class=VehicleClass.CAR, angle=0.3)

	def test_oriented_size_without_angle_raises(self) -> None:
		with pytest.raises(ValueError, match="together"):
			Detection(
				bbox=_bbox(), score=0.9, vehicle_class=VehicleClass.CAR, oriented_size=(12.0, 6.0)
			)

	def test_non_positive_oriented_size_raises(self) -> None:
		with pytest.raises(ValueError, match="oriented_size"):
			Detection(
				bbox=_bbox(),
				score=0.9,
				vehicle_class=VehicleClass.CAR,
				angle=0.0,
				oriented_size=(0.0, 6.0),
			)


class TestTrackedDetection:
	def test_negative_track_id_raises(self) -> None:
		detection = Detection(bbox=_bbox(), score=0.9, vehicle_class=VehicleClass.CAR)
		with pytest.raises(ValueError, match="track_id"):
			TrackedDetection(track_id=-1, detection=detection)

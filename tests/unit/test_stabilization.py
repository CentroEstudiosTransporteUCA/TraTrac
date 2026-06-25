"""Tests for the pure coordinate-stabilization helper."""

from __future__ import annotations

import pytest

from tratrac.application.stabilization import (
	EgoMotionStabilizer,
	NullDetectionStabilizer,
	apply_transform,
)
from tratrac.domain.detection import Detection, VehicleClass
from tratrac.domain.geometry import BoundingBox, Transform2D


def _detection(x: float, y: float, w: float, h: float) -> Detection:
	return Detection(
		bbox=BoundingBox(x=x, y=y, width=w, height=h),
		score=0.7,
		vehicle_class=VehicleClass.CAR,
	)


def test_translation_moves_centre_keeps_size() -> None:
	shift = Transform2D(a=1.0, b=0.0, tx=10.0, c=0.0, d=1.0, ty=-5.0)
	out = apply_transform(_detection(0.0, 0.0, 4.0, 2.0), shift)
	assert out.bbox.center.x == pytest.approx(12.0)  # (0+2) + 10
	assert out.bbox.center.y == pytest.approx(-4.0)  # (0+1) - 5
	assert out.bbox.width == pytest.approx(4.0)
	assert out.bbox.height == pytest.approx(2.0)


def test_uniform_scale_resizes_box() -> None:
	scaled = Transform2D(a=2.0, b=0.0, tx=0.0, c=0.0, d=2.0, ty=0.0)
	out = apply_transform(_detection(10.0, 10.0, 4.0, 2.0), scaled)
	assert out.bbox.width == pytest.approx(8.0)
	assert out.bbox.height == pytest.approx(4.0)
	# Centre (12, 11) scaled by 2 -> (24, 22).
	assert out.bbox.center.x == pytest.approx(24.0)
	assert out.bbox.center.y == pytest.approx(22.0)


def test_identity_is_a_no_op() -> None:
	det = _detection(3.0, 4.0, 5.0, 6.0)
	out = apply_transform(det, Transform2D.identity())
	assert out.bbox == det.bbox


def test_preserves_score_and_class() -> None:
	det = _detection(0.0, 0.0, 4.0, 2.0)
	out = apply_transform(det, Transform2D.identity())
	assert out.score == det.score
	assert out.vehicle_class is det.vehicle_class


class TestEgoMotionStabilizer:
	def test_maps_every_detection_through_the_pose(self) -> None:
		shift = Transform2D(a=1.0, b=0.0, tx=10.0, c=0.0, d=1.0, ty=0.0)
		out = EgoMotionStabilizer().stabilize([_detection(0.0, 0.0, 4.0, 2.0)], shift)
		assert out[0].bbox.center.x == pytest.approx(12.0)  # (0+2) + 10


class TestNullDetectionStabilizer:
	def test_returns_detections_unchanged_ignoring_the_pose(self) -> None:
		dets = [_detection(0.0, 0.0, 4.0, 2.0)]
		shift = Transform2D(a=1.0, b=0.0, tx=10.0, c=0.0, d=1.0, ty=0.0)
		assert NullDetectionStabilizer().stabilize(dets, shift) is dets  # same list, untouched

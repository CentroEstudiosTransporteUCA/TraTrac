"""Tests for the orientation + kinematics estimator."""

from __future__ import annotations

import math

import pytest

from tratrac.application.orientation import OrientationEstimator
from tratrac.domain.detection import Detection, TrackedDetection, VehicleClass
from tratrac.domain.geometry import BoundingBox


def _tracked(track_id: int, bbox: BoundingBox, *, score: float = 0.9) -> TrackedDetection:
	return TrackedDetection(
		track_id=track_id,
		detection=Detection(bbox=bbox, score=score, vehicle_class=VehicleClass.CAR),
	)


class TestFirstObservation:
	def test_velocity_and_acceleration_are_zero(self) -> None:
		estimator = OrientationEstimator()
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		assert state.velocity.magnitude == 0.0
		assert state.acceleration.magnitude == 0.0

	def test_heading_falls_back_to_bbox_major_axis_east_when_wide(self) -> None:
		estimator = OrientationEstimator()
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		assert state.heading.dx == 1.0
		assert state.heading.dy == 0.0

	def test_heading_falls_back_to_bbox_major_axis_south_when_tall(self) -> None:
		estimator = OrientationEstimator()
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=2.0, height=8.0)),
			timestamp_seconds=0.0,
		)
		assert state.heading.dx == 0.0
		assert state.heading.dy == 1.0

	def test_dimensions_come_from_bbox_major_minor(self) -> None:
		estimator = OrientationEstimator()
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=10.0, y=20.0, width=6.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		assert state.dimensions.length == 6.0
		assert state.dimensions.width == 2.0


class TestVelocity:
	def test_two_observations_eastward_yield_east_heading(self) -> None:
		estimator = OrientationEstimator()
		estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=10.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=1.0,
		)
		# Centroid moved from (2, 1) to (12, 1) in 1 s => velocity (10, 0).
		assert math.isclose(state.velocity.dx, 10.0)
		assert math.isclose(state.velocity.dy, 0.0)
		assert math.isclose(state.heading.dx, 1.0)
		assert math.isclose(state.heading.dy, 0.0)

	def test_stationary_track_falls_back_to_bbox(self) -> None:
		estimator = OrientationEstimator()
		bbox = BoundingBox(x=0.0, y=0.0, width=2.0, height=6.0)
		estimator.estimate(_tracked(1, bbox), timestamp_seconds=0.0)
		state = estimator.estimate(_tracked(1, bbox), timestamp_seconds=1.0)
		assert state.velocity.magnitude == 0.0
		assert state.heading == state.heading.__class__(0.0, 1.0)


class TestAcceleration:
	def test_constant_velocity_yields_zero_acceleration(self) -> None:
		estimator = OrientationEstimator()
		for i, t in enumerate([0.0, 1.0, 2.0]):
			estimator.estimate(
				_tracked(1, BoundingBox(x=i * 10.0, y=0.0, width=4.0, height=2.0)),
				timestamp_seconds=t,
			)
		final = estimator.estimate(
			_tracked(1, BoundingBox(x=30.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=3.0,
		)
		# Velocity has been constant at 10 m/s. Acceleration should be ~0.
		assert math.isclose(final.acceleration.dx, 0.0, abs_tol=1e-6)
		assert math.isclose(final.acceleration.dy, 0.0, abs_tol=1e-6)

	def test_accelerating_motion_produces_positive_acceleration(self) -> None:
		estimator = OrientationEstimator()
		# x positions: 0, 1, 3, 6 — velocity rising 1, 2, 3 per unit time.
		positions = [0.0, 1.0, 3.0, 6.0]
		for i, x in enumerate(positions):
			estimator.estimate(
				_tracked(1, BoundingBox(x=x, y=0.0, width=2.0, height=1.0)),
				timestamp_seconds=float(i),
			)
		# Hit one more frame to materialize acceleration.
		final = estimator.estimate(
			_tracked(1, BoundingBox(x=10.0, y=0.0, width=2.0, height=1.0)),
			timestamp_seconds=4.0,
		)
		assert final.acceleration.dx > 0.0


class TestIdentity:
	def test_separate_tracks_do_not_share_history(self) -> None:
		estimator = OrientationEstimator()
		estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		# Track 2 has only one observation — should not be polluted by track 1's history.
		state = estimator.estimate(
			_tracked(2, BoundingBox(x=100.0, y=100.0, width=4.0, height=2.0)),
			timestamp_seconds=1.0,
		)
		assert state.velocity.magnitude == 0.0

	def test_vehicle_id_matches_track_id(self) -> None:
		estimator = OrientationEstimator()
		state = estimator.estimate(
			_tracked(42, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		assert state.vehicle_id == 42


class TestForget:
	def test_forget_drops_history(self) -> None:
		estimator = OrientationEstimator()
		estimator.estimate(
			_tracked(1, BoundingBox(x=0.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=0.0,
		)
		estimator.forget(1)
		# After forgetting, the next observation is treated as the first.
		state = estimator.estimate(
			_tracked(1, BoundingBox(x=10.0, y=0.0, width=4.0, height=2.0)),
			timestamp_seconds=1.0,
		)
		assert state.velocity.magnitude == 0.0


class TestConfig:
	def test_smoothing_window_must_be_at_least_two(self) -> None:
		with pytest.raises(ValueError, match="window"):
			OrientationEstimator(smoothing_window=1)

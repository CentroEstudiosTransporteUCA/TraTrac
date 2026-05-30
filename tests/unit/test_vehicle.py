"""Tests for VehicleState — bumper math is the only non-trivial behavior."""

from __future__ import annotations

import math

import pytest

from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState


def _state(
	*,
	centroid: Point2D = Point2D(100.0, 200.0),
	heading: Heading = Heading(1.0, 0.0),
	length: float = 4.0,
	width: float = 2.0,
	velocity: Vector2D = Vector2D(0.0, 0.0),
	acceleration: float = 0.0,
) -> VehicleState:
	return VehicleState(
		vehicle_id=1,
		timestamp_seconds=0.0,
		centroid=centroid,
		heading=heading,
		dimensions=Dimensions(length=length, width=width),
		velocity=velocity,
		acceleration=acceleration,
	)


class TestBumpers:
	def test_east_facing_vehicle_front_is_to_the_right(self) -> None:
		state = _state(heading=Heading(1.0, 0.0), length=4.0)
		assert state.front_bumper == Point2D(102.0, 200.0)
		assert state.rear_bumper == Point2D(98.0, 200.0)

	def test_north_facing_vehicle_front_is_upward(self) -> None:
		state = _state(heading=Heading(0.0, 1.0), length=6.0)
		assert state.front_bumper == Point2D(100.0, 203.0)
		assert state.rear_bumper == Point2D(100.0, 197.0)

	def test_diagonal_heading(self) -> None:
		state = _state(heading=Heading.from_angle(math.pi / 4), length=2.0 * math.sqrt(2))
		assert math.isclose(state.front_bumper.x, 101.0)
		assert math.isclose(state.front_bumper.y, 201.0)
		assert math.isclose(state.rear_bumper.x, 99.0)
		assert math.isclose(state.rear_bumper.y, 199.0)


class TestLinkAndLaneIds:
	def test_link_and_lane_default_to_zero(self) -> None:
		state = _state()
		assert state.link_id == 0
		assert state.lane_id == 0

	def test_link_id_negative_raises(self) -> None:
		with pytest.raises(ValueError, match="link_id"):
			VehicleState(
				vehicle_id=1,
				timestamp_seconds=0.0,
				centroid=Point2D(0.0, 0.0),
				heading=Heading(1.0, 0.0),
				dimensions=Dimensions(length=4.0, width=2.0),
				velocity=Vector2D(0.0, 0.0),
				acceleration=0.0,
				link_id=-1,
			)

	def test_lane_id_above_255_raises(self) -> None:
		with pytest.raises(ValueError, match="lane_id"):
			VehicleState(
				vehicle_id=1,
				timestamp_seconds=0.0,
				centroid=Point2D(0.0, 0.0),
				heading=Heading(1.0, 0.0),
				dimensions=Dimensions(length=4.0, width=2.0),
				velocity=Vector2D(0.0, 0.0),
				acceleration=0.0,
				lane_id=256,
			)

	def test_lane_id_negative_raises(self) -> None:
		with pytest.raises(ValueError, match="lane_id"):
			VehicleState(
				vehicle_id=1,
				timestamp_seconds=0.0,
				centroid=Point2D(0.0, 0.0),
				heading=Heading(1.0, 0.0),
				dimensions=Dimensions(length=4.0, width=2.0),
				velocity=Vector2D(0.0, 0.0),
				acceleration=0.0,
				lane_id=-1,
			)


class TestKinematics:
	def test_speed_is_velocity_magnitude(self) -> None:
		state = _state(velocity=Vector2D(3.0, 4.0))
		assert state.speed == 5.0

	def test_acceleration_is_the_stored_longitudinal_scalar(self) -> None:
		# Acceleration is the rate of change of speed (units/sec²), stored
		# directly — no heading projection. The estimator computes it.
		state = _state(acceleration=2.5)
		assert state.acceleration == 2.5

	def test_acceleration_is_negative_when_decelerating(self) -> None:
		state = _state(acceleration=-3.0)
		assert state.acceleration == -3.0

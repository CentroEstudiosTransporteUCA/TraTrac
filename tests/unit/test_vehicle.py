"""Tests for VehicleState — bumper math is the only non-trivial behavior."""

from __future__ import annotations

import math

from tratrac.domain.geometry import Dimensions, Heading, Point2D, Vector2D
from tratrac.domain.vehicle import VehicleState


def _state(
	*,
	centroid: Point2D = Point2D(100.0, 200.0),
	heading: Heading = Heading(1.0, 0.0),
	length: float = 4.0,
	width: float = 2.0,
	velocity: Vector2D = Vector2D(0.0, 0.0),
	acceleration: Vector2D = Vector2D(0.0, 0.0),
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


class TestKinematics:
	def test_speed_is_velocity_magnitude(self) -> None:
		state = _state(velocity=Vector2D(3.0, 4.0))
		assert state.speed == 5.0

	def test_forward_acceleration_projects_onto_heading(self) -> None:
		state = _state(heading=Heading(1.0, 0.0), acceleration=Vector2D(2.5, 99.0))
		assert state.forward_acceleration == 2.5

	def test_forward_acceleration_is_negative_when_decelerating(self) -> None:
		state = _state(heading=Heading(1.0, 0.0), acceleration=Vector2D(-3.0, 0.0))
		assert state.forward_acceleration == -3.0

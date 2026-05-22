"""Tests for domain.geometry value objects."""

from __future__ import annotations

import math

import pytest

from tratrac.domain.geometry import BoundingBox, Dimensions, Heading, Point2D, Vector2D


class TestVector2D:
	def test_magnitude_uses_hypot(self) -> None:
		assert Vector2D(3.0, 4.0).magnitude == 5.0

	def test_scaled_by_distributes(self) -> None:
		assert Vector2D(2.0, -1.0).scaled_by(3.0) == Vector2D(6.0, -3.0)

	def test_normalized_yields_unit_heading(self) -> None:
		heading = Vector2D(3.0, 4.0).normalized()
		assert math.isclose(math.hypot(heading.dx, heading.dy), 1.0)

	def test_normalizing_zero_vector_raises(self) -> None:
		with pytest.raises(ValueError, match="zero vector"):
			Vector2D(0.0, 0.0).normalized()


class TestHeading:
	def test_rejects_non_unit_input(self) -> None:
		with pytest.raises(ValueError, match="unit length"):
			Heading(2.0, 0.0)

	def test_from_angle_round_trips(self) -> None:
		heading = Heading.from_angle(math.pi / 4)
		assert math.isclose(heading.dx, math.sqrt(2) / 2)
		assert math.isclose(heading.dy, math.sqrt(2) / 2)

	def test_reversed_flips_both_components(self) -> None:
		h = Heading.from_angle(0.5)
		assert h.reversed() == Heading(-h.dx, -h.dy)

	def test_as_vector_with_magnitude_scales(self) -> None:
		h = Heading(1.0, 0.0)
		assert h.as_vector_with_magnitude(7.5) == Vector2D(7.5, 0.0)


class TestPoint2D:
	def test_translate_adds_displacement(self) -> None:
		assert Point2D(1.0, 2.0).translate_by(Vector2D(3.0, 4.0)) == Point2D(4.0, 6.0)

	def test_displacement_to_is_other_minus_self(self) -> None:
		assert Point2D(1.0, 1.0).displacement_to(Point2D(4.0, 5.0)) == Vector2D(3.0, 4.0)


class TestDimensions:
	def test_rejects_non_positive(self) -> None:
		with pytest.raises(ValueError, match="positive"):
			Dimensions(0.0, 1.0)
		with pytest.raises(ValueError, match="positive"):
			Dimensions(1.0, -0.5)


class TestBoundingBox:
	def test_center_is_midpoint(self) -> None:
		bbox = BoundingBox(x=10.0, y=20.0, width=40.0, height=80.0)
		assert bbox.center == Point2D(30.0, 60.0)

	def test_axis_lengths(self) -> None:
		bbox = BoundingBox(x=0.0, y=0.0, width=12.0, height=5.0)
		assert bbox.major_axis_length == 12.0
		assert bbox.minor_axis_length == 5.0

	def test_rejects_non_positive_dimensions(self) -> None:
		with pytest.raises(ValueError, match="positive"):
			BoundingBox(x=0.0, y=0.0, width=0.0, height=10.0)

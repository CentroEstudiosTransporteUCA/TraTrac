"""Tests for domain.geometry value objects."""

from __future__ import annotations

import math

import pytest

from tratrac.domain.geometry import (
	BoundingBox,
	Dimensions,
	Heading,
	Point2D,
	Transform2D,
	Vector2D,
	clipped_overlap_fraction,
	point_in_polygon,
)


class TestPointInPolygon:
	_SQUARE = (Point2D(0.0, 0.0), Point2D(10.0, 0.0), Point2D(10.0, 10.0), Point2D(0.0, 10.0))

	def test_inside_point(self) -> None:
		assert point_in_polygon(Point2D(5.0, 5.0), self._SQUARE) is True

	def test_outside_point(self) -> None:
		assert point_in_polygon(Point2D(15.0, 5.0), self._SQUARE) is False

	def test_concave_polygon(self) -> None:
		# A C-shaped concave polygon; a point in the notch is outside.
		c_shape = (
			Point2D(0.0, 0.0),
			Point2D(10.0, 0.0),
			Point2D(10.0, 3.0),
			Point2D(3.0, 3.0),
			Point2D(3.0, 7.0),
			Point2D(10.0, 7.0),
			Point2D(10.0, 10.0),
			Point2D(0.0, 10.0),
		)
		assert point_in_polygon(Point2D(7.0, 5.0), c_shape) is False  # in the notch
		assert point_in_polygon(Point2D(1.0, 5.0), c_shape) is True  # in the spine

	def test_degenerate_polygon_contains_nothing(self) -> None:
		assert point_in_polygon(Point2D(0.0, 0.0), (Point2D(0.0, 0.0), Point2D(1.0, 1.0))) is False


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


class TestTransform2D:
	def test_identity_is_a_no_op(self) -> None:
		assert Transform2D.identity().apply(Point2D(3.0, -7.0)) == Point2D(3.0, -7.0)

	def test_apply_translates(self) -> None:
		shift = Transform2D(a=1.0, b=0.0, tx=10.0, c=0.0, d=1.0, ty=-5.0)
		assert shift.apply(Point2D(2.0, 2.0)) == Point2D(12.0, -3.0)

	def test_apply_scales_and_translates(self) -> None:
		t = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=3.0, ty=-1.0)
		assert t.apply(Point2D(4.0, 5.0)) == Point2D(9.0, 14.0)

	def test_apply_rotates_quarter_turn(self) -> None:
		# +90°: (x, y) -> (-y, x). a=0,b=-1,c=1,d=0.
		rot = Transform2D(a=0.0, b=-1.0, tx=0.0, c=1.0, d=0.0, ty=0.0)
		out = rot.apply(Point2D(1.0, 0.0))
		assert math.isclose(out.x, 0.0, abs_tol=1e-9)
		assert math.isclose(out.y, 1.0, abs_tol=1e-9)

	def test_compose_applies_inner_first(self) -> None:
		# outer.compose(inner).apply(p) == outer.apply(inner.apply(p))
		inner = Transform2D(a=1.0, b=0.0, tx=3.0, c=0.0, d=1.0, ty=4.0)
		outer = Transform2D(a=2.0, b=0.0, tx=0.0, c=0.0, d=2.0, ty=0.0)
		p = Point2D(1.0, 1.0)
		assert outer.compose(inner).apply(p) == outer.apply(inner.apply(p))
		# Concretely: inner -> (4, 5), then outer scales x2 -> (8, 10).
		assert outer.compose(inner).apply(p) == Point2D(8.0, 10.0)

	def test_identity_is_neutral_under_compose(self) -> None:
		t = Transform2D(a=2.0, b=1.0, tx=3.0, c=-1.0, d=2.0, ty=4.0)
		ident = Transform2D.identity()
		p = Point2D(5.0, 6.0)
		assert t.compose(ident).apply(p) == t.apply(p)
		assert ident.compose(t).apply(p) == t.apply(p)

	def test_inverse_undoes_apply(self) -> None:
		# A rotation + scale + translation (a representative similarity).
		t = Transform2D(a=2.0, b=1.0, tx=3.0, c=-1.0, d=2.0, ty=4.0)
		p = Point2D(5.0, 6.0)
		back = t.inverse().apply(t.apply(p))
		assert back.x == pytest.approx(p.x)
		assert back.y == pytest.approx(p.y)

	def test_inverse_of_identity_is_identity(self) -> None:
		p = Point2D(5.0, 6.0)
		assert Transform2D.identity().inverse().apply(p) == p

	def test_inverse_raises_on_singular_linear_part(self) -> None:
		singular = Transform2D(a=1.0, b=2.0, tx=0.0, c=2.0, d=4.0, ty=0.0)
		with pytest.raises(ValueError, match="singular"):
			singular.inverse()

	def test_scale_of_pure_translation_is_one(self) -> None:
		shift = Transform2D(a=1.0, b=0.0, tx=10.0, c=0.0, d=1.0, ty=-5.0)
		assert shift.scale == pytest.approx(1.0)

	def test_scale_of_uniform_scaling(self) -> None:
		scaled = Transform2D(a=3.0, b=0.0, tx=0.0, c=0.0, d=3.0, ty=0.0)
		assert scaled.scale == pytest.approx(3.0)

	def test_scale_is_rotation_invariant(self) -> None:
		# A similarity with scale 2 rotated 30°: scale must still read 2.
		s, ang = 2.0, math.radians(30.0)
		sim = Transform2D(
			a=s * math.cos(ang),
			b=-s * math.sin(ang),
			tx=0.0,
			c=s * math.sin(ang),
			d=s * math.cos(ang),
			ty=0.0,
		)
		assert sim.scale == pytest.approx(2.0)


class TestClippedOverlapFraction:
	def test_identity_is_full_overlap(self) -> None:
		assert clipped_overlap_fraction(Transform2D.identity(), 100, 100) == pytest.approx(1.0)

	def test_half_width_translation_halves_overlap(self) -> None:
		# Shift the frame 50px right in a 100px-wide frame: 50px column still inside.
		shift = Transform2D(a=1.0, b=0.0, tx=50.0, c=0.0, d=1.0, ty=0.0)
		assert clipped_overlap_fraction(shift, 100, 100) == pytest.approx(0.5)

	def test_fully_disjoint_is_zero(self) -> None:
		shift = Transform2D(a=1.0, b=0.0, tx=200.0, c=0.0, d=1.0, ty=0.0)
		assert clipped_overlap_fraction(shift, 100, 100) == pytest.approx(0.0)

	def test_zoom_in_keeps_frame_inside_anchor(self) -> None:
		# Content scaled to half size lands within the anchor: a quarter of the area.
		half = Transform2D(a=0.5, b=0.0, tx=0.0, c=0.0, d=0.5, ty=0.0)
		assert clipped_overlap_fraction(half, 100, 100) == pytest.approx(0.25)

	def test_rejects_non_positive_dimensions(self) -> None:
		with pytest.raises(ValueError, match="positive"):
			clipped_overlap_fraction(Transform2D.identity(), 0, 100)

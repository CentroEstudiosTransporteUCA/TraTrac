"""Geometric value objects. Coordinate-frame agnostic — same types serve image and world space."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vector2D:
	"""A 2D displacement. Arbitrary magnitude."""

	dx: float
	dy: float

	@property
	def magnitude(self) -> float:
		return math.hypot(self.dx, self.dy)

	def scaled_by(self, factor: float) -> Vector2D:
		return Vector2D(self.dx * factor, self.dy * factor)

	def normalized(self) -> Heading:
		"""Return the unit-length direction vector. Raises if magnitude is zero."""
		m = self.magnitude
		if m == 0.0:
			raise ValueError("Cannot normalize a zero vector into a heading.")
		return Heading(self.dx / m, self.dy / m)


@dataclass(frozen=True, slots=True)
class Heading:
	"""A unit-length 2D direction. Always magnitude 1 (±1e-6)."""

	dx: float
	dy: float

	def __post_init__(self) -> None:
		m = math.hypot(self.dx, self.dy)
		if not math.isclose(m, 1.0, abs_tol=1e-6):
			raise ValueError(f"Heading must be unit length, got magnitude {m}.")

	@classmethod
	def from_angle(cls, radians: float) -> Heading:
		return cls(math.cos(radians), math.sin(radians))

	def as_vector_with_magnitude(self, distance: float) -> Vector2D:
		return Vector2D(self.dx * distance, self.dy * distance)

	def reversed(self) -> Heading:
		return Heading(-self.dx, -self.dy)


@dataclass(frozen=True, slots=True)
class Point2D:
	"""A 2D point in some coordinate frame. Frame is the caller's responsibility."""

	x: float
	y: float

	def translate_by(self, displacement: Vector2D) -> Point2D:
		return Point2D(self.x + displacement.dx, self.y + displacement.dy)

	def displacement_to(self, other: Point2D) -> Vector2D:
		return Vector2D(other.x - self.x, other.y - self.y)


@dataclass(frozen=True, slots=True)
class Dimensions:
	"""Vehicle bounding dimensions. Units track DIMENSIONS.Units in the SSAM file."""

	length: float
	width: float

	def __post_init__(self) -> None:
		if self.length <= 0 or self.width <= 0:
			raise ValueError(
				f"Dimensions must be positive: length={self.length} width={self.width}."
			)


@dataclass(frozen=True, slots=True)
class BoundingBox:
	"""Axis-aligned rectangle in image pixel space (top-left origin, y grows down)."""

	x: float
	y: float
	width: float
	height: float

	def __post_init__(self) -> None:
		if self.width <= 0 or self.height <= 0:
			raise ValueError(f"BoundingBox must be positive: w={self.width} h={self.height}.")

	@property
	def center(self) -> Point2D:
		return Point2D(self.x + self.width / 2.0, self.y + self.height / 2.0)

	@property
	def major_axis_length(self) -> float:
		return max(self.width, self.height)

	@property
	def minor_axis_length(self) -> float:
		return min(self.width, self.height)

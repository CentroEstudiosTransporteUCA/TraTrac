"""Geometric value objects. Coordinate-frame agnostic — same types serve image and world space."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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
class Transform2D:
	"""A 2D affine transform, stored as the six coefficients of a 2x3 matrix.

	Maps a point ``(x, y)`` to ``(a·x + b·y + tx, c·x + d·y + ty)``. Coordinate-frame
	agnostic like the other geometry types. Used (MVP1.9, see ``vault/05_75_mvp1_9.md``)
	to carry the camera ego-motion estimated per frame: the transform mapping a
	frame's pixels into a fixed stabilization reference frame. The estimator fits a
	4-DOF similarity (translation + rotation + uniform scale); the storage is the
	general affine form, so a full affine fits here too if ever needed.
	"""

	a: float
	b: float
	tx: float
	c: float
	d: float
	ty: float

	@classmethod
	def identity(cls) -> Transform2D:
		return cls(a=1.0, b=0.0, tx=0.0, c=0.0, d=1.0, ty=0.0)

	def apply(self, point: Point2D) -> Point2D:
		return Point2D(
			self.a * point.x + self.b * point.y + self.tx,
			self.c * point.x + self.d * point.y + self.ty,
		)

	def compose(self, inner: Transform2D) -> Transform2D:
		"""Return the transform that applies ``inner`` first, then ``self``.

		``self.compose(inner).apply(p) == self.apply(inner.apply(p))`` — i.e. the
		matrix product ``self @ inner``. Used to accumulate per-step transforms into
		a single frame-to-reference transform.
		"""
		return Transform2D(
			a=self.a * inner.a + self.b * inner.c,
			b=self.a * inner.b + self.b * inner.d,
			tx=self.a * inner.tx + self.b * inner.ty + self.tx,
			c=self.c * inner.a + self.d * inner.c,
			d=self.c * inner.b + self.d * inner.d,
			ty=self.c * inner.tx + self.d * inner.ty + self.ty,
		)

	@property
	def scale(self) -> float:
		"""Uniform scale factor of the linear part (``sqrt`` of the determinant).

		Exact for a 4-DOF similarity, where the determinant is the squared scale.
		Used to resize a bounding box when mapping a detection between frames so
		zoom is normalised along with translation/rotation.
		"""
		return math.sqrt(abs(self.a * self.d - self.b * self.c))

	def inverse(self) -> Transform2D:
		"""Return the transform that undoes this one.

		``self.inverse().apply(self.apply(p)) == p`` (within float error). Defined
		for any affine whose linear part is non-singular; a 4-DOF similarity always
		qualifies (its determinant is the squared scale, > 0). Used to map a point
		from the stabilization reference frame back into a raw frame's coordinates.
		Raises ``ValueError`` if the linear part is singular.
		"""
		det = self.a * self.d - self.b * self.c
		if det == 0.0:
			raise ValueError("Cannot invert a transform with a singular linear part.")
		ia = self.d / det
		ib = -self.b / det
		ic = -self.c / det
		id_ = self.a / det
		return Transform2D(
			a=ia,
			b=ib,
			tx=-(ia * self.tx + ib * self.ty),
			c=ic,
			d=id_,
			ty=-(ic * self.tx + id_ * self.ty),
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


@dataclass(frozen=True, slots=True)
class Polygon:
	"""A simple polygon in some coordinate frame (image or stabilization).

	Vertices in order (winding either way). Backs image-space exclusion zones —
	regions whose detections are dropped before tracking (see
	``vault/21_exclusion_zones.md``). Pure vertex container: coverage of a bounding
	box is computed by an infrastructure ``DetectionMask`` that rasterizes the zones
	per frame, so concave polygons and overlapping zones union correctly.
	"""

	vertices: tuple[Point2D, ...]

	def __post_init__(self) -> None:
		if len(self.vertices) < 3:
			raise ValueError(f"Polygon needs at least 3 vertices, got {len(self.vertices)}.")


def clipped_overlap_fraction(transform: Transform2D, width: int, height: int) -> float:
	"""Fraction of a ``width``x``height`` reference rectangle still covered by a frame.

	``transform`` maps the *current* frame's pixel coordinates into the reference
	(anchor) frame. We map the current frame's rectangle corners through it and
	measure how much of that mapped quad falls inside the reference rectangle
	``[0, width] x [0, height]``, as a fraction of the reference area. Used by the
	keyframe stabilizer to decide when the camera has drifted far enough from the
	anchor that a new anchor is warranted. Pure geometry — no pixels, no cv2.
	"""
	if width <= 0 or height <= 0:
		raise ValueError(f"Rectangle dimensions must be positive: {width}x{height}.")
	w, h = float(width), float(height)
	corners = [Point2D(0.0, 0.0), Point2D(w, 0.0), Point2D(w, h), Point2D(0.0, h)]
	mapped = [transform.apply(c) for c in corners]
	clipped = _clip_to_rectangle(mapped, w, h)
	if len(clipped) < 3:
		return 0.0
	return min(1.0, _polygon_area(clipped) / (w * h))


def _polygon_area(polygon: list[Point2D]) -> float:
	"""Absolute area of a simple polygon via the shoelace formula."""
	total = 0.0
	for i in range(len(polygon)):
		a = polygon[i]
		b = polygon[(i + 1) % len(polygon)]
		total += a.x * b.y - b.x * a.y
	return abs(total) / 2.0


def _clip_to_rectangle(subject: list[Point2D], width: float, height: float) -> list[Point2D]:
	"""Sutherland-Hodgman clip of a convex polygon against ``[0,width]x[0,height]``.

	Each rectangle edge is a half-plane; the polygon is clipped edge by edge. The
	rectangle is convex, so the standard algorithm yields the exact intersection.
	"""
	# (keep-predicate, intersection-axis) per rectangle edge, walked CCW.
	edges: list[tuple[Callable[[Point2D], bool], Callable[[Point2D, Point2D], Point2D]]] = [
		(lambda p: p.x >= 0.0, lambda a, b: _intersect_x(a, b, 0.0)),
		(lambda p: p.x <= width, lambda a, b: _intersect_x(a, b, width)),
		(lambda p: p.y >= 0.0, lambda a, b: _intersect_y(a, b, 0.0)),
		(lambda p: p.y <= height, lambda a, b: _intersect_y(a, b, height)),
	]
	polygon = subject
	for inside, intersect in edges:
		if not polygon:
			return []
		clipped: list[Point2D] = []
		for i in range(len(polygon)):
			current = polygon[i]
			previous = polygon[i - 1]
			cur_in, prev_in = inside(current), inside(previous)
			if cur_in:
				if not prev_in:
					clipped.append(intersect(previous, current))
				clipped.append(current)
			elif prev_in:
				clipped.append(intersect(previous, current))
		polygon = clipped
	return polygon


def _intersect_x(a: Point2D, b: Point2D, x: float) -> Point2D:
	"""Point where segment a→b crosses the vertical line X = ``x``."""
	t = (x - a.x) / (b.x - a.x)
	return Point2D(x, a.y + t * (b.y - a.y))


def _intersect_y(a: Point2D, b: Point2D, y: float) -> Point2D:
	"""Point where segment a→b crosses the horizontal line Y = ``y``."""
	t = (y - a.y) / (b.y - a.y)
	return Point2D(a.x + t * (b.x - a.x), y)


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
	"""Whether ``point`` lies inside ``polygon`` (even-odd ray casting).

	Frame-agnostic like the other helpers; works for concave polygons. Boundary cases
	are not specially handled — sufficient for testing trajectory centroids against ROI
	polygons (see vault/21_exclusion_zones.md). A polygon of fewer than 3 vertices
	contains nothing.
	"""
	n = len(polygon)
	if n < 3:
		return False
	inside = False
	j = n - 1
	for i in range(n):
		vi, vj = polygon[i], polygon[j]
		if (vi.y > point.y) != (vj.y > point.y):
			x_cross = (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x
			if point.x < x_cross:
				inside = not inside
		j = i
	return inside

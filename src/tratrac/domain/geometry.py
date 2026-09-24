"""Geometric value objects. Coordinate-frame agnostic — same types serve image and world space."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import shapely


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

	def dot(self, other: Heading) -> float:
		"""Cosine of the angle between two unit headings (both are unit length by construction).

		Used to disambiguate a 0-360° OBB angle's front/back against the last known
		heading (GitHub Issues Group A7): negative means the candidate
		points the wrong way and should be reversed.
		"""
		return self.dx * other.dx + self.dy * other.dy


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
	agnostic like the other geometry types. Used (MVP1.9, see
	``infrastructure/video/ego_motion_orb.py``'s module docstring)
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
	``src/tratrac/application/EXCLUSION_ZONES.md``). Pure vertex container: coverage of a bounding
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
	anchor that a new anchor is warranted. Pure geometry (via ``shapely``) — no
	pixels, no cv2.
	"""
	if width <= 0 or height <= 0:
		raise ValueError(f"Rectangle dimensions must be positive: {width}x{height}.")
	w, h = float(width), float(height)
	corners = [Point2D(0.0, 0.0), Point2D(w, 0.0), Point2D(w, h), Point2D(0.0, h)]
	mapped = [transform.apply(c) for c in corners]
	raw_quad = shapely.Polygon([(p.x, p.y) for p in mapped])
	quad: shapely.geometry.base.BaseGeometry = (
		raw_quad if raw_quad.is_valid else shapely.make_valid(raw_quad)
	)
	reference = shapely.box(0.0, 0.0, w, h)
	overlap_area: float = quad.intersection(reference).area
	return min(1.0, overlap_area / (w * h))


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
	"""Whether ``point`` lies inside ``polygon`` (via ``shapely``'s prepared-geometry contains).

	Frame-agnostic like the other helpers; works for concave polygons. A polygon of
	fewer than 3 vertices contains nothing. See ``src/tratrac/application/EXCLUSION_ZONES.md``.
	"""
	if len(polygon) < 3:
		return False
	shape = shapely.Polygon([(p.x, p.y) for p in polygon])
	return bool(shape.contains(shapely.Point(point.x, point.y)))


def oriented_extent(polygon: Polygon, angle: float | None) -> tuple[float, float]:
	"""A ``(length, width)`` pair describing how far ``polygon`` extends along ``angle`` and its
	perpendicular — the real occupancy-derived dimensions a footprint (Group D1,
	GitHub Issues) buys over a bbox/OBB estimate.

	When ``angle`` is known (radians, standard math convention — matches
	``Detection.angle``/``Heading.from_angle``), each vertex is projected onto the heading axis
	and its perpendicular; the extent along each is that axis's ``max - min``. This is exact
	when the polygon's true orientation matches ``angle`` (e.g. the OBB angle that also produced
	the crop the mask came from) and only approximate otherwise (a mask's own principal axis
	need not exactly match a detector's OBB angle).

	When ``angle`` is ``None`` (no OBB angle available for this observation), falls back to the
	polygon's plain axis-aligned bounding box extent — not a fitted principal axis (e.g. minimum-
	area rectangle / PCA): simpler, and correct precisely in the already-common case where the
	vehicle happens to be axis-aligned, same honesty tradeoff ``_major_axis_heading`` already
	makes for the low-speed heading fallback (``application/track_smoothing.py``).
	"""
	xs = [v.x for v in polygon.vertices]
	ys = [v.y for v in polygon.vertices]
	if angle is None:
		return max(xs) - min(xs), max(ys) - min(ys)
	cos_a, sin_a = math.cos(angle), math.sin(angle)
	along = [x * cos_a + y * sin_a for x, y in zip(xs, ys, strict=True)]
	perp = [-x * sin_a + y * cos_a for x, y in zip(xs, ys, strict=True)]
	return max(along) - min(along), max(perp) - min(perp)


def oriented_box_to_aabb(cx: float, cy: float, w: float, h: float, angle: float) -> BoundingBox:
	"""The axis-aligned enclosing box of a rotated ``(cx, cy, w, h, angle)`` rectangle.

	Shared by every OBB-capable adapter (GitHub Issues Group A4/A5) that needs
	an unconditional ``Detection.bbox`` alongside the oriented box — ORB masking and IoU
	association still work in AABB space even when the detector/tracker reports orientation.
	"""
	cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
	half_w = (w * cos_a + h * sin_a) / 2.0
	half_h = (w * sin_a + h * cos_a) / 2.0
	return BoundingBox(x=cx - half_w, y=cy - half_h, width=2.0 * half_w, height=2.0 * half_h)

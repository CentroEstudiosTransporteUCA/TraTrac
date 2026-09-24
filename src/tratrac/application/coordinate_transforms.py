"""``TransformTable``: the one lookup-and-apply class every ``CoordinateTransform``
in the system is built from.

Replaces what used to be five separate impls (`IdentityTransform`,
`ScaleTransform`, `PerFrameTransform`, `PerAnchorTransform`,
`MultiPlaneTransform`) with one generic row-selection algorithm: group a
table's rows by `frame_index` (the primary key), filter to the row(s) whose
`zone` contains the query point, and delegate to that row's `function`
(`infrastructure/transform/records.py`'s `TransformFunction` objects, which do
the actual math). See `infrastructure/video/ego_motion_orb.py`'s module docstring.

A run's full transform composes **two** `TransformTable`s in sequence, never
one flat table: ego-motion (raw pixel -> global pixel) and scale-or-homography
(global pixel -> final metric/world unit) operate in different coordinate
spaces, so they're stages, not competing alternatives for one query.
`compose`/`compose_invertible` fold them (and, for a static/non-projected run,
an empty table standing in for "no stage") into the one object every caller
(`tratrac`'s pipeline, `tratrac-postprocess`) ever touches.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from tratrac.domain.geometry import Point2D, point_in_polygon
from tratrac.domain.ports import CoordinateTransform, InvertibleCoordinateTransform
from tratrac.infrastructure.transform.records import (
	IdentityFunction,
	InvertibleTransformFunction,
	TransformFunction,
	TransformRow,
)

_IDENTITY = IdentityFunction()


class TransformTable:
	"""Groups rows by `frame_index`, selects by `zone`, applies the row's `function`.

	An entirely empty table (no rows at all -- e.g. the ego-motion stage of a
	static-camera run) behaves as the identity for every query: there is no
	stage to apply, so the point passes through unchanged. A non-empty table
	queried at a `frame_index` it has no rows for is a data-integrity error
	(the file is expected to be complete for the whole clip) and raises, same
	contract every prior impl had.
	"""

	def __init__(self, rows: Iterable[TransformRow]) -> None:
		by_frame: dict[int, list[TransformRow]] = defaultdict(list)
		for row in rows:
			by_frame[row.frame_index].append(row)
		self._by_frame: dict[int, list[TransformRow]] = dict(by_frame)

	@property
	def is_empty(self) -> bool:
		"""Whether this table has no rows at all (the identity-fallback case) --
		distinct from a specific frame being absent from a *non-empty* table, which
		is a data-integrity error (see ``function_at``)."""
		return not self._by_frame

	@property
	def is_invertible(self) -> bool:
		"""Whether ``reverse`` is safe for every frame this table covers.

		An empty table (identity fallback) is trivially invertible. Otherwise every
		frame's row-set must be unambiguous (exactly one row -- no zone
		disambiguation needed to reverse) and that row's function itself
		invertible. A per-plane homography spanning more than one zone at any
		frame fails this, exactly mirroring `MultiPlaneTransform`'s prior
		reasoning: reversing needs to know which zone a *forward* point came
		from, which a world-space point can't tell you.
		"""
		if not self._by_frame:
			return True
		return all(
			len(rows) == 1 and isinstance(rows[0].function, InvertibleTransformFunction)
			for rows in self._by_frame.values()
		)

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		if not self._by_frame:
			return point
		row = self._select(point, frame_index)
		return row.function.apply(point)

	def function_at_point(self, point: Point2D, frame_index: int) -> TransformFunction:
		"""The one row's function whose zone ``point`` falls in at ``frame_index``.

		For a caller that needs to apply the *same* zone's function to several related
		points (e.g. a bounding box's centroid and its extent corners) without each one
		independently re-selecting a zone -- which could pick a different zone for a
		corner near a boundary, or raise "matches no zone" for a corner that strays
		outside a narrow zone even though the box's own position is squarely inside it.
		An empty table returns the identity function, matching ``apply``.
		"""
		if not self._by_frame:
			return _IDENTITY
		return self._select(point, frame_index).function

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		if not self._by_frame:
			return point
		rows = self._rows_for(frame_index)
		if len(rows) > 1:
			raise ValueError(
				f"cannot reverse frame {frame_index}: {len(rows)} zoned rows are ambiguous "
				"without a forward point to select by."
			)
		function = rows[0].function
		if not isinstance(function, InvertibleTransformFunction):
			raise ValueError(f"{type(function).__name__} at frame {frame_index} is not invertible.")
		return function.reverse(point)

	def kinds(self) -> frozenset[type]:
		"""The distinct ``TransformFunction`` classes present in this table.

		For a caller that needs to know *which* transformation kind a table holds
		before picking how to use it (e.g. ``tratrac-postprocess`` telling a
		plain-scale projection stage apart from a world-projection homography
		one) without inspecting individual rows.
		"""
		return frozenset(type(row.function) for rows in self._by_frame.values() for row in rows)

	def any_function(self) -> TransformFunction | None:
		"""Any one row's function, or ``None`` if the table is empty.

		For a caller that knows the table is homogeneous (e.g. every row a
		constant-factor ``ScaleFunction``) and just needs a representative
		instance, not a lookup by point/frame.
		"""
		for rows in self._by_frame.values():
			return rows[0].function
		return None

	def function_at(self, frame_index: int) -> TransformFunction | None:
		"""The raw ``TransformFunction`` recorded for ``frame_index``, or ``None`` if
		this table has nothing for it at all.

		For a consumer that needs the function object itself (e.g.
		``PrecomputedEgoMotionEstimator``, which needs the concrete
		``Transform2D`` a live pipeline step applies to a detection, not just a
		mapped point) rather than mapping one point through it. Raises
		``ValueError`` if more than one row shares this frame_index (ambiguous
		without a point to select a zone by).
		"""
		rows = self._by_frame.get(frame_index)
		if not rows:
			return None
		if len(rows) > 1:
			raise ValueError(
				f"frame {frame_index} has {len(rows)} rows; ambiguous without a point to select by."
			)
		return rows[0].function

	def _rows_for(self, frame_index: int) -> list[TransformRow]:
		rows = self._by_frame.get(frame_index)
		if not rows:
			raise KeyError(f"no transform recorded for frame {frame_index}.")
		return rows

	def _select(self, point: Point2D, frame_index: int) -> TransformRow:
		rows = self._rows_for(frame_index)
		matches = [row for row in rows if point_in_polygon(point, row.zone)]
		if not matches:
			raise ValueError(f"point {point} at frame {frame_index} matches no zone.")
		if len(matches) > 1:
			raise ValueError(
				f"point {point} at frame {frame_index} matches {len(matches)} zones ambiguously."
			)
		return matches[0]


@dataclass(frozen=True, slots=True)
class TranslationTransform:
	"""A constant 2D shift, ignoring ``frame_index``.

	Used to fold a post-hoc origin normalization (e.g. shifting a world
	projection to a non-negative origin) onto a ``TransformTable`` via
	``compose``/``compose_invertible``, replacing what used to be a
	transform-specific ``.shifted()`` method.
	"""

	dx: float
	dy: float

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return Point2D(point.x + self.dx, point.y + self.dy)

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		del frame_index
		return Point2D(point.x - self.dx, point.y - self.dy)


@dataclass(frozen=True, slots=True)
class ComposedTransform:
	"""Folds several ``CoordinateTransform`` stages into one, applied in listed order.

	Construct via ``compose`` (or ``compose_invertible`` for an all-invertible
	chain), not directly.
	"""

	stages: tuple[CoordinateTransform, ...]

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in self.stages:
			point = stage.apply(point, frame_index)
		return point


def compose(*stages: CoordinateTransform) -> CoordinateTransform:
	"""Fold ``stages`` into one ``CoordinateTransform``, applied in order."""
	return ComposedTransform(stages)


@dataclass(frozen=True, slots=True)
class _ComposedInvertibleTransform:
	stages: tuple[InvertibleCoordinateTransform, ...]

	def apply(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in self.stages:
			point = stage.apply(point, frame_index)
		return point

	def reverse(self, point: Point2D, frame_index: int) -> Point2D:
		for stage in reversed(self.stages):
			point = stage.reverse(point, frame_index)
		return point


def compose_invertible(*stages: InvertibleCoordinateTransform) -> InvertibleCoordinateTransform:
	"""Fold ``stages`` into one ``InvertibleCoordinateTransform``, applied in order."""
	return _ComposedInvertibleTransform(stages)


def local_scale_at(transform: CoordinateTransform, point: Point2D, frame_index: int = 0) -> float:
	"""Estimate metres-per-pixel at ``point`` (the transform's local scale there).

	Projects ``point`` and a neighbour one pixel away in x and measures the world gap.
	Used to convert pixel-tuned smoother noise into the world units the projection
	produces, so the Kalman behavior is preserved.
	"""
	here = transform.apply(point, frame_index)
	over = transform.apply(Point2D(point.x + 1.0, point.y), frame_index)
	return math.hypot(over.x - here.x, over.y - here.y)
